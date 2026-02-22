"""Main pipeline orchestrator — ties all stages together.

Stage 0: Model Analysis
Stage 1: Docker Setup
Stage 2: Launch + Auto-Fix (+ optional Claude Code agent)
Stage 3: Grid Search Benchmark
Stage 4: Results Compilation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from porter.agent import ClaudeCodeAgent
from porter.analyzer import ModelAnalyzer, ModelProfile
from porter.benchmark import BenchmarkRunner
from porter.config import (
    DEFAULT_DB_PATH,
    DEFAULT_DOCKER_IMAGE,
    MODEL_WEIGHTS_CONTAINER_DIR,
    SGLANG_MAX_RETRIES,
)
from porter.database import Database
from porter.diagnoser import ErrorDiagnoser
from porter.docker_manager import DockerManager
from porter.grid_search import GridSearchEngine
from porter.server_config import ServerConfig

log = logging.getLogger(__name__)


@dataclass
class PipelineEvent:
    stage: str
    message: str
    timestamp: float = 0.0
    level: str = "info"

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PipelineResult:
    model_id: str
    job_id: int = 0
    success: bool = False
    profile: Optional[ModelProfile] = None
    best_config: Optional[ServerConfig] = None
    best_throughput: float = 0.0
    docker_run_cmd: str = ""
    fixes_applied: list[str] = field(default_factory=list)
    events: list[PipelineEvent] = field(default_factory=list)
    error: str = ""
    duration_seconds: float = 0.0


class Pipeline:
    """Orchestrate the full model porting pipeline."""

    def __init__(
        self,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        db_path: str = DEFAULT_DB_PATH,
        on_event: Optional[Callable[[PipelineEvent], None]] = None,
    ):
        self.docker = DockerManager(image=docker_image)
        self.db = Database(db_path)
        self.analyzer = ModelAnalyzer()
        self.diagnoser = ErrorDiagnoser()
        self._on_event = on_event

    def _emit(self, stage: str, message: str, level: str = "info") -> PipelineEvent:
        event = PipelineEvent(stage=stage, message=message, level=level)
        log.log(getattr(logging, level.upper(), logging.INFO), "[%s] %s", stage, message)
        if self._on_event:
            self._on_event(event)
        return event

    def run(self, model_id: str) -> PipelineResult:
        """Execute the full pipeline for a HuggingFace model."""
        start = time.time()
        result = PipelineResult(model_id=model_id)
        result.job_id = self.db.create_job(model_id)
        container_name = f"porter-{model_id.replace('/', '-').lower()}"

        try:
            self.db.update_job_status(result.job_id, "running")

            # Stage 0: Analysis
            profile = self._stage0_analyze(model_id, result)

            # Stage 1: Docker Setup
            self._stage1_docker(container_name, result)

            # Ensure model weights
            self._emit("setup", f"Checking model weights for {model_id}")
            resolved_path = self.docker.ensure_model_weights(model_id, container_name)
            profile.recommended_config.model_path = resolved_path

            # Stage 2: Launch + Auto-Fix
            working_config = self._stage2_launch(
                container_name, profile, result,
            )

            if not working_config:
                result.success = False
                result.error = "Failed to launch SGLang server after all retry attempts"
                self.db.update_job_status(result.job_id, "failed", error_log=result.error)
                return result

            # Stage 3: Grid Search
            self._stage3_grid_search(
                container_name, working_config, profile, result,
            )

            # Stage 4: Results
            self._stage4_results(result)

            result.success = True
            self.db.update_job_status(
                result.job_id, "completed",
                best_config=result.best_config.summary() if result.best_config else "",
                best_throughput=result.best_throughput,
                docker_run_cmd=result.docker_run_cmd,
                fixes_applied=json.dumps(result.fixes_applied),
            )

        except Exception as e:
            result.error = str(e)
            result.success = False
            self._emit("error", f"Pipeline failed: {e}", level="error")
            self.db.update_job_status(result.job_id, "failed", error_log=str(e))

        finally:
            result.duration_seconds = time.time() - start
            self._emit(
                "done",
                f"Pipeline {'succeeded' if result.success else 'failed'} "
                f"in {result.duration_seconds:.0f}s",
            )

        return result

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage0_analyze(self, model_id: str, result: PipelineResult) -> ModelProfile:
        self._emit("stage0", f"Analyzing {model_id}")
        profile = self.analyzer.analyze(model_id)
        result.profile = profile

        report = self.analyzer.format_report(profile)
        self._emit("stage0", report)

        self.db.update_job_status(result.job_id, "analyzing", profile_json=report)
        self._emit("stage0", f"Analysis complete. {len(profile.predicted_issues)} predicted issues.")
        return profile

    def _stage1_docker(self, container_name: str, result: PipelineResult) -> None:
        self._emit("stage1", "Setting up Docker container")

        if not self.docker.image_exists():
            self._emit("stage1", f"Pulling image: {self.docker.image}")
            if not self.docker.pull_image():
                raise RuntimeError("Failed to pull Docker image")

        self._emit("stage1", f"Creating container: {container_name}")
        self.docker.create_container(container_name)

        # Wait for setup script to complete
        self._emit("stage1", "Waiting for container setup (tmux, claude CLI)...")
        import time as _time
        for _ in range(60):
            r = self.docker.exec_cmd(container_name, "which claude 2>/dev/null && echo READY", timeout=10)
            if "READY" in r.stdout:
                break
            _time.sleep(5)

        self._emit("stage1", "Container ready")

    def _stage2_launch(
        self,
        container_name: str,
        profile: ModelProfile,
        result: PipelineResult,
    ) -> Optional[ServerConfig]:
        self._emit("stage2", "Starting launch + auto-fix loop")
        config = profile.recommended_config
        attempted_fixes: list[str] = []

        for attempt in range(1, SGLANG_MAX_RETRIES + 1):
            self._emit("stage2", f"Attempt {attempt}/{SGLANG_MAX_RETRIES}: {config.summary()}")

            # Handle special env flags from diagnoser
            if config.env_overrides.get("__PORTER_INSTALL_AITER"):
                self._emit("stage2", "Installing aiter in container")
                self.docker.exec_cmd(container_name, "pip install aiter", timeout=120)
                env = dict(config.env_overrides)
                env.pop("__PORTER_INSTALL_AITER", None)
                config = config.clone(env_overrides=env)

            if config.env_overrides.get("__PORTER_UPGRADE_TRANSFORMERS"):
                self._emit("stage2", "Upgrading transformers from source")
                self.docker.exec_cmd(
                    container_name,
                    "pip install git+https://github.com/huggingface/transformers.git",
                    timeout=300,
                )
                env = dict(config.env_overrides)
                env.pop("__PORTER_UPGRADE_TRANSFORMERS", None)
                config = config.clone(env_overrides=env)

            # Launch
            healthy = self.docker.launch_server(container_name, config)
            if healthy:
                # Correctness check
                ok, content = self.docker.verify_correctness(config.port)
                if ok:
                    self._emit("stage2", f"Server healthy and correct on attempt {attempt}")
                    result.fixes_applied = attempted_fixes
                    return config
                else:
                    self._emit("stage2", f"Correctness check failed: {content}", level="warning")
                    self.docker.kill_server(container_name)

            # Diagnose failure
            logs = self.docker.get_logs(container_name, tail=200)
            diagnoses = self.diagnoser.diagnose(logs)

            if not diagnoses:
                self._emit("stage2", "No known pattern matched — escalating to agent", level="warning")
                break

            # Apply highest-severity fix
            diag = diagnoses[0]
            self._emit("stage2", f"Diagnosed: [{diag.severity}] {diag.issue_id}: {diag.description}")

            should_escalate = config.env_overrides.get("__PORTER_ESCALATE_TO_AGENT")
            if should_escalate:
                break

            config = self.diagnoser.apply_fix(config, diag)
            attempted_fixes.append(f"{diag.issue_id}: {diag.fix_action}")
            self.docker.kill_server(container_name)

        # Escalate to Claude Code agent
        self._emit("stage2", "Launching Claude Code agent for debugging")
        agent = ClaudeCodeAgent(self.docker, container_name)
        agent_result = agent.run(
            profile=profile, config=config,
            error_logs=self.docker.get_logs(container_name, tail=300),
            attempted_fixes=attempted_fixes,
        )

        if agent_result.success:
            self._emit("stage2", f"Agent resolved issue in {agent_result.duration_seconds:.0f}s")
            result.fixes_applied = attempted_fixes + ["agent: " + agent_result.fix_description]
            return config

        self._emit("stage2", "All launch attempts exhausted", level="error")
        result.fixes_applied = attempted_fixes
        return None

    def _stage3_grid_search(
        self,
        container_name: str,
        working_config: ServerConfig,
        profile: ModelProfile,
        result: PipelineResult,
    ) -> None:
        self._emit("stage3", "Starting grid search benchmark")

        benchmark = BenchmarkRunner(self.docker, container_name)
        engine = GridSearchEngine(self.docker, container_name, benchmark, self.db)

        grid_results = engine.run(working_config, profile, job_id=result.job_id)

        if grid_results and grid_results[0].launched:
            best = grid_results[0]
            result.best_config = best.config
            result.best_throughput = best.weighted_score
            self._emit(
                "stage3",
                f"Best config: {best.config_id} ({best.config.summary()}) "
                f"— {best.weighted_score:.0f} weighted score",
            )
        else:
            result.best_config = working_config
            self._emit("stage3", "Grid search found no improvements over initial config")

    def _stage4_results(self, result: PipelineResult) -> None:
        self._emit("stage4", "Compiling results")

        if result.best_config:
            result.docker_run_cmd = self._build_docker_run_cmd(result)
            self._emit("stage4", f"Docker run command:\n{result.docker_run_cmd}")

        self._emit("stage4", f"Best throughput: {result.best_throughput:.0f} weighted score")
        if result.fixes_applied:
            self._emit("stage4", f"Fixes applied: {', '.join(result.fixes_applied)}")

    def _build_docker_run_cmd(self, result: PipelineResult) -> str:
        config = result.best_config
        env = config.build_env()
        env_str = " \\\n  ".join(f"-e {k}={v}" for k, v in sorted(env.items()))

        return f"""docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \\
  --shm-size=128g --network=host \\
  --device=/dev/kfd --device=/dev/dri --group-add video \\
  -v /mnt/dcgpuval/huggingface:/sgl-workspace/models \\
  {env_str} \\
  {self.docker.image} \\
  bash -lc '{config.build_shell_cmd()}'"""
