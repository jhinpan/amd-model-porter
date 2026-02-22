"""Benchmark runner wrapping SGLang's bench_serving.py."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from porter.config import BENCHMARK_SCENARIOS, BenchmarkScenario, MODEL_WEIGHTS_CONTAINER_DIR
from porter.docker_manager import DockerManager

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    scenario: str
    total_throughput: float = 0.0  # tok/s (input + output)
    output_throughput: float = 0.0  # tok/s (output only)
    mean_ttft_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    mean_itl_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    num_completed: int = 0
    num_failed: int = 0
    duration_seconds: float = 0.0
    raw_output: str = ""
    error: str = ""


class BenchmarkRunner:
    """Run SGLang bench_serving.py inside a Docker container."""

    def __init__(self, docker: DockerManager, container_name: str):
        self.docker = docker
        self.container = container_name

    def run_scenario(
        self,
        model_path: str,
        scenario: BenchmarkScenario,
        port: int = 30000,
        timeout: int = 600,
    ) -> BenchmarkResult:
        """Run a single benchmark scenario and parse results."""
        start = time.time()
        log.info("Running benchmark: %s", scenario.name)

        cmd = self._build_cmd(model_path, scenario, port)
        result = self.docker.exec_cmd(self.container, cmd, timeout=timeout)

        duration = time.time() - start
        output = result.stdout + result.stderr

        if result.returncode != 0:
            log.error("Benchmark %s failed: %s", scenario.name, output[-500:])
            return BenchmarkResult(
                scenario=scenario.name, error=output[-1000:], duration_seconds=duration,
            )

        parsed = self._parse_output(output)
        parsed.scenario = scenario.name
        parsed.duration_seconds = duration
        parsed.raw_output = output[-3000:]
        log.info(
            "Benchmark %s: %.0f tok/s total, %.0f tok/s output",
            scenario.name, parsed.total_throughput, parsed.output_throughput,
        )
        return parsed

    def run_all_scenarios(
        self, model_path: str, port: int = 30000,
    ) -> list[BenchmarkResult]:
        """Run all standard benchmark scenarios."""
        results = []
        for scenario in BENCHMARK_SCENARIOS:
            r = self.run_scenario(model_path, scenario, port)
            results.append(r)
        return results

    @staticmethod
    def _build_cmd(model_path: str, s: BenchmarkScenario, port: int) -> str:
        parts = [
            "HF_HUB_OFFLINE=1",
            "python3 -m sglang.bench_serving",
            "--backend sglang",
            f"--base-url http://127.0.0.1:{port}",
            "--dataset-name random",
            f"--num-prompts {s.num_prompts}",
            f"--random-input-len {s.input_len}",
            f"--random-output-len {s.output_len}",
            f"--request-rate {s.request_rate}",
            f"--max-concurrency {s.concurrency}",
            f"--model {model_path}",
            "--disable-stream",
        ]
        return " ".join(parts)

    @staticmethod
    def _parse_output(output: str) -> BenchmarkResult:
        """Parse bench_serving.py text output into a BenchmarkResult."""
        r = BenchmarkResult(scenario="")

        def _extract(pattern: str, text: str) -> float:
            m = re.search(pattern, text)
            return float(m.group(1)) if m else 0.0

        r.total_throughput = _extract(r"Total throughput:\s*([\d.]+)\s*tok", output)
        r.output_throughput = _extract(r"Output throughput:\s*([\d.]+)\s*tok", output)
        r.mean_ttft_ms = _extract(r"Mean TTFT.*?:\s*([\d.]+)\s*ms", output)
        r.mean_e2e_ms = _extract(r"Mean E2E.*?:\s*([\d.]+)\s*ms", output)
        r.mean_itl_ms = _extract(r"Mean ITL.*?:\s*([\d.]+)\s*ms", output)
        r.p99_ttft_ms = _extract(r"P99 TTFT.*?:\s*([\d.]+)\s*ms", output)
        r.p99_e2e_ms = _extract(r"P99 E2E.*?:\s*([\d.]+)\s*ms", output)
        r.num_completed = int(_extract(r"Completed requests:\s*(\d+)", output))
        r.num_failed = int(_extract(r"Failed requests:\s*(\d+)", output))

        # Fallback: look for throughput in different format
        if r.total_throughput == 0:
            r.total_throughput = _extract(r"Throughput:\s*([\d.]+)", output)

        return r
