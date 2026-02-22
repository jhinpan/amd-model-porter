"""SQLite database for storing pipeline jobs and benchmark results."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from porter.benchmark import BenchmarkResult
from porter.config import DEFAULT_DB_PATH

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    profile_json TEXT,
    best_config TEXT,
    best_throughput REAL DEFAULT 0,
    fixes_applied TEXT,
    error_log TEXT,
    docker_run_cmd TEXT
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    config_id TEXT NOT NULL,
    config_summary TEXT,
    scenario TEXT NOT NULL,
    total_throughput REAL DEFAULT 0,
    output_throughput REAL DEFAULT 0,
    mean_ttft_ms REAL DEFAULT 0,
    mean_e2e_ms REAL DEFAULT 0,
    mean_itl_ms REAL DEFAULT 0,
    p99_ttft_ms REAL DEFAULT 0,
    p99_e2e_ms REAL DEFAULT 0,
    num_completed INTEGER DEFAULT 0,
    num_failed INTEGER DEFAULT 0,
    duration_seconds REAL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_job ON benchmarks(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_id);
"""


class Database:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_job(self, model_id: str) -> int:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO jobs (model_id, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (model_id, "pending", now, now),
            )
            return cur.lastrowid

    def update_job_status(self, job_id: int, status: str, **kwargs):
        sets = ["status = ?", "updated_at = ?"]
        params = [status, time.time()]
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            params.append(v)
        params.append(job_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)

    def get_job(self, job_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def insert_benchmark(
        self, job_id: int, config_id: str, config_summary: str, result: BenchmarkResult,
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO benchmarks
                   (job_id, config_id, config_summary, scenario,
                    total_throughput, output_throughput,
                    mean_ttft_ms, mean_e2e_ms, mean_itl_ms,
                    p99_ttft_ms, p99_e2e_ms,
                    num_completed, num_failed, duration_seconds, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, config_id, config_summary, result.scenario,
                    result.total_throughput, result.output_throughput,
                    result.mean_ttft_ms, result.mean_e2e_ms, result.mean_itl_ms,
                    result.p99_ttft_ms, result.p99_e2e_ms,
                    result.num_completed, result.num_failed,
                    result.duration_seconds, result.error, time.time(),
                ),
            )

    def get_benchmarks(self, job_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmarks WHERE job_id = ? ORDER BY config_id, scenario",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_leaderboard(self) -> list[dict]:
        """Return best throughput per model, ranked."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT j.model_id, j.best_throughput, j.best_config, j.status,
                          j.created_at, j.id as job_id
                   FROM jobs j
                   WHERE j.status = 'completed'
                   ORDER BY j.best_throughput DESC""",
            ).fetchall()
            return [dict(r) for r in rows]
