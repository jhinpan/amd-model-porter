"""Click CLI for AMD Model Porter."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from porter.config import DEFAULT_DB_PATH, DEFAULT_DOCKER_IMAGE

console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """AMD Model Porter — automated model porting for AMD GPUs."""
    _setup_logging(verbose)


@cli.command()
@click.argument("model_id")
def analyze(model_id: str):
    """Analyze a HuggingFace model's architecture and predict AMD issues."""
    from porter.analyzer import ModelAnalyzer

    analyzer = ModelAnalyzer()
    profile = analyzer.analyze(model_id)
    console.print(analyzer.format_report(profile))


@cli.command()
@click.argument("model_id")
@click.option("--image", default=DEFAULT_DOCKER_IMAGE, help="Docker image")
@click.option("--db", default=DEFAULT_DB_PATH, help="Database path")
def run(model_id: str, image: str, db: str):
    """Run the full pipeline: analyze → deploy → auto-fix → benchmark."""
    from porter.pipeline import Pipeline, PipelineEvent

    def on_event(event: PipelineEvent):
        style = {"error": "red", "warning": "yellow"}.get(event.level, "green")
        console.print(f"[{style}][{event.stage}][/{style}] {event.message}")

    pipeline = Pipeline(docker_image=image, db_path=db, on_event=on_event)
    result = pipeline.run(model_id)

    console.print()
    if result.success:
        console.print("[bold green]Pipeline succeeded![/bold green]")
        if result.docker_run_cmd:
            console.print("\n[bold]Optimal docker run command:[/bold]")
            console.print(result.docker_run_cmd)
    else:
        console.print(f"[bold red]Pipeline failed:[/bold red] {result.error}")

    console.print(f"\nDuration: {result.duration_seconds:.0f}s")
    console.print(f"Job ID: {result.job_id}")


@cli.command()
@click.option("--db", default=DEFAULT_DB_PATH, help="Database path")
@click.option("--limit", default=20, help="Max results")
def leaderboard(db: str, limit: int):
    """Show the model performance leaderboard."""
    from porter.database import Database

    database = Database(db)
    rows = database.get_leaderboard()

    table = Table(title="AMD Model Porter Leaderboard")
    table.add_column("Rank", style="bold")
    table.add_column("Model")
    table.add_column("Throughput (tok/s)", justify="right")
    table.add_column("Config")

    for i, row in enumerate(rows[:limit], 1):
        table.add_row(
            str(i), row["model_id"],
            f"{row['best_throughput']:.0f}",
            row["best_config"] or "",
        )

    console.print(table)


@cli.command()
@click.argument("model_id")
def download(model_id: str):
    """Download model weights to the shared storage."""
    from porter.docker_manager import DockerManager

    docker = DockerManager()
    container_name = f"porter-download-{model_id.replace('/', '-').lower()}"

    console.print(f"Creating temp container for download: {container_name}")
    docker.create_container(container_name)

    try:
        console.print(f"Downloading {model_id}...")
        path = docker.ensure_model_weights(model_id, container_name)
        console.print(f"[green]Downloaded to: {path}[/green]")
    finally:
        docker.remove_container(container_name)


@cli.command()
@click.option("--port", default=8080, help="Web UI port")
@click.option("--db", default=DEFAULT_DB_PATH, help="Database path")
@click.option("--image", default=DEFAULT_DOCKER_IMAGE, help="Docker image")
def web(port: int, db: str, image: str):
    """Start the web UI."""
    import uvicorn
    from porter.web.app import create_app

    app = create_app(db_path=db, docker_image=image)
    console.print(f"[bold]Starting web UI on http://0.0.0.0:{port}[/bold]")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    cli()
