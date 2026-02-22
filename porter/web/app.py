"""FastAPI web application for AMD Model Porter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from porter.config import DEFAULT_DB_PATH, DEFAULT_DOCKER_IMAGE
from porter.database import Database
from porter.pipeline import Pipeline, PipelineEvent

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Global event queues for SSE
_event_queues: dict[int, asyncio.Queue] = {}


def create_app(db_path: str = DEFAULT_DB_PATH, docker_image: str = DEFAULT_DOCKER_IMAGE) -> FastAPI:
    app = FastAPI(title="AMD Model Porter")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    db = Database(db_path)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        jobs = db.list_jobs(limit=10)
        return templates.TemplateResponse("submit.html", {"request": request, "jobs": jobs})

    @app.post("/submit")
    async def submit(request: Request):
        form = await request.form()
        model_id = form.get("model_id", "").strip()
        if not model_id:
            return HTMLResponse("<p class='error'>Model ID is required</p>", status_code=400)

        job_id = db.create_job(model_id)
        _event_queues[job_id] = asyncio.Queue()

        asyncio.get_event_loop().run_in_executor(
            None, _run_pipeline, model_id, job_id, docker_image, db_path,
        )

        return HTMLResponse(
            f'<script>window.location.href="/job/{job_id}";</script>',
            headers={"HX-Redirect": f"/job/{job_id}"},
        )

    @app.get("/job/{job_id}", response_class=HTMLResponse)
    async def job_page(request: Request, job_id: int):
        job = db.get_job(job_id)
        if not job:
            return HTMLResponse("Job not found", status_code=404)
        benchmarks = db.get_benchmarks(job_id)
        return templates.TemplateResponse("progress.html", {
            "request": request, "job": job, "benchmarks": benchmarks,
        })

    @app.get("/job/{job_id}/stream")
    async def job_stream(job_id: int):
        async def event_generator():
            queue = _event_queues.get(job_id)
            if not queue:
                yield f"data: {json.dumps({'stage': 'done', 'message': 'No active stream'})}\n\n"
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60)
                    if event is None:
                        yield f"data: {json.dumps({'stage': 'done', 'message': 'Pipeline complete'})}\n\n"
                        break
                    yield f"data: {json.dumps({'stage': event.stage, 'message': event.message, 'level': event.level})}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'stage': 'heartbeat', 'message': 'waiting...'})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/results/{job_id}", response_class=HTMLResponse)
    async def results_page(request: Request, job_id: int):
        job = db.get_job(job_id)
        if not job:
            return HTMLResponse("Job not found", status_code=404)
        benchmarks = db.get_benchmarks(job_id)
        return templates.TemplateResponse("results.html", {
            "request": request, "job": job, "benchmarks": benchmarks,
        })

    @app.get("/leaderboard", response_class=HTMLResponse)
    async def leaderboard_page(request: Request):
        entries = db.get_leaderboard()
        return templates.TemplateResponse("leaderboard.html", {
            "request": request, "entries": entries,
        })

    return app


def _run_pipeline(model_id: str, job_id: int, docker_image: str, db_path: str):
    """Run pipeline in a background thread, pushing events to SSE queue."""
    queue = _event_queues.get(job_id)

    def on_event(event: PipelineEvent):
        if queue:
            asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, event)

    try:
        pipeline = Pipeline(docker_image=docker_image, db_path=db_path, on_event=on_event)
        pipeline.run(model_id)
    except Exception as e:
        if queue:
            ev = PipelineEvent(stage="error", message=str(e), level="error")
            asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, ev)
    finally:
        if queue:
            asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, None)
        _event_queues.pop(job_id, None)
