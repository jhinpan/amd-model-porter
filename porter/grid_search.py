"""Stage 3 — Tiered grid search over SGLang configurations.

Tier 1: Backend sweep (triton, aiter)
Tier 2: Parallelism sweep (TP, EP, DP)
Tier 3: Memory tuning (mem-fraction, cache sizes, chunked-prefill)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from porter.analyzer import ModelProfile
from porter.benchmark import BenchmarkResult, BenchmarkRunner
from porter.config import (
    ATTENTION_BACKENDS,
    CHUNKED_PREFILL_SIZES,
    MEM_FRACTIONS,
    PARALLELISM_CONFIGS,
)
from porter.database import Database
from porter.docker_manager import DockerManager
from porter.server_config import ServerConfig

log = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    config_id: str
    config: ServerConfig
    results: list[BenchmarkResult] = field(default_factory=list)
    weighted_score: float = 0.0
    launched: bool = False
    error: str = ""


class GridSearchEngine:
    """Three-tier grid search over SGLang configurations."""

    SCORE_WEIGHTS = {"throughput": 0.50, "balanced": 0.30, "burst": 0.20}

    def __init__(
        self,
        docker: DockerManager,
        container_name: str,
        benchmark: BenchmarkRunner,
        db: Optional[Database] = None,
    ):
        self.docker = docker
        self.container = container_name
        self.benchmark = benchmark
        self.db = db

    def run(
        self,
        base_config: ServerConfig,
        profile: ModelProfile,
        job_id: Optional[int] = None,
    ) -> list[GridSearchResult]:
        """Run full 3-tier grid search. Returns all results sorted by score."""
        all_results: list[GridSearchResult] = []

        # Tier 1: Backend sweep
        log.info("=== Tier 1: Backend Sweep ===")
        tier1 = self._tier1_backend(base_config, profile)
        all_results.extend(tier1)
        best_t1 = self._pick_best(tier1)
        if not best_t1:
            log.error("No successful backend config in Tier 1")
            return all_results

        log.info("Tier 1 winner: %s (%.0f score)", best_t1.config_id, best_t1.weighted_score)

        # Tier 2: Parallelism sweep
        log.info("=== Tier 2: Parallelism Sweep ===")
        tier2 = self._tier2_parallelism(best_t1.config, profile)
        all_results.extend(tier2)
        best_t2 = self._pick_best(tier2)
        if not best_t2:
            best_t2 = best_t1

        log.info("Tier 2 winner: %s (%.0f score)", best_t2.config_id, best_t2.weighted_score)

        # Tier 3: Memory tuning
        log.info("=== Tier 3: Memory Tuning ===")
        tier3 = self._tier3_memory(best_t2.config, profile)
        all_results.extend(tier3)

        # Persist results
        if self.db and job_id:
            for r in all_results:
                for br in r.results:
                    self.db.insert_benchmark(job_id, r.config_id, r.config.summary(), br)

        all_results.sort(key=lambda r: r.weighted_score, reverse=True)
        return all_results

    # ------------------------------------------------------------------
    # Tiers
    # ------------------------------------------------------------------

    def _tier1_backend(
        self, base: ServerConfig, profile: ModelProfile,
    ) -> list[GridSearchResult]:
        results = []
        for backend in ATTENTION_BACKENDS:
            config_id = f"T1-{backend}"
            config = base.clone(
                attention_backend=backend, disable_cuda_graph=False,
            )
            r = self._evaluate(config_id, config)
            results.append(r)
        return results

    def _tier2_parallelism(
        self, base: ServerConfig, profile: ModelProfile,
    ) -> list[GridSearchResult]:
        results = []
        for i, pconfig in enumerate(PARALLELISM_CONFIGS):
            config_id = f"T2-{chr(65+i)}"
            overrides = {}
            overrides["tp"] = pconfig.get("tp", base.tp)
            if "ep" in pconfig:
                overrides["ep"] = pconfig["ep"]
            if "dp" in pconfig:
                overrides["dp"] = pconfig["dp"]
            config = base.clone(**overrides)
            r = self._evaluate(config_id, config)
            results.append(r)
        return results

    def _tier3_memory(
        self, base: ServerConfig, profile: ModelProfile,
    ) -> list[GridSearchResult]:
        results = []

        # Memory fraction sweep
        for frac in MEM_FRACTIONS:
            config_id = f"T3-mem{frac}"
            config = base.clone(mem_fraction_static=frac)
            r = self._evaluate(config_id, config)
            results.append(r)

        # Mamba cache size sweep (for hybrid attention models)
        if profile.has_deltanet or profile.has_hybrid_attention:
            for size in [32, 64, 128, 256]:
                config_id = f"T3-mamba{size}"
                config = base.clone(max_mamba_cache_size=size)
                r = self._evaluate(config_id, config)
                results.append(r)

        # CUDA graph on/off
        config_id = "T3-no-cuda-graph"
        r = self._evaluate(config_id, base.clone(disable_cuda_graph=True))
        results.append(r)

        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, config_id: str, config: ServerConfig) -> GridSearchResult:
        """Launch server with config, benchmark, return result."""
        log.info("Evaluating %s: %s", config_id, config.summary())

        launched = self.docker.launch_server(self.container, config)
        if not launched:
            logs = self.docker.get_logs(self.container, tail=50)
            log.warning("Config %s failed to launch", config_id)
            return GridSearchResult(
                config_id=config_id, config=config, launched=False, error=logs[-500:],
            )

        bench_results = self.benchmark.run_all_scenarios(config.model_path, config.port)
        score = self._compute_score(bench_results)

        self.docker.kill_server(self.container)
        return GridSearchResult(
            config_id=config_id, config=config, results=bench_results,
            weighted_score=score, launched=True,
        )

    def _compute_score(self, results: list[BenchmarkResult]) -> float:
        score = 0.0
        for r in results:
            weight = self.SCORE_WEIGHTS.get(r.scenario, 0)
            score += weight * r.total_throughput
        return score

    @staticmethod
    def _pick_best(results: list[GridSearchResult]) -> Optional[GridSearchResult]:
        launched = [r for r in results if r.launched and r.weighted_score > 0]
        if not launched:
            return None
        return max(launched, key=lambda r: r.weighted_score)
