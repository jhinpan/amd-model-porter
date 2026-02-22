"""Stage 2 — Error Diagnoser.

Pattern-matches SGLang launch failures against known AMD/ROCm issues
and applies deterministic fixes. Unknown errors are escalated to the
Claude Code agent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from porter.server_config import ServerConfig

log = logging.getLogger(__name__)

KNOWN_ISSUES_PATH = Path(__file__).parent / "known_issues.yaml"


@dataclass
class Diagnosis:
    issue_id: str
    pattern: str
    severity: str
    description: str
    fix_action: str
    matched_text: str = ""


class ErrorDiagnoser:
    """Match SGLang error logs against known issue patterns and generate fixes."""

    def __init__(self):
        self.issues = self._load_issues()
        self._compiled: list[tuple[re.Pattern, dict]] = [
            (re.compile(iss["pattern"], re.IGNORECASE | re.MULTILINE), iss)
            for iss in self.issues
        ]

    @staticmethod
    def _load_issues() -> list[dict]:
        if not KNOWN_ISSUES_PATH.exists():
            return []
        with open(KNOWN_ISSUES_PATH) as f:
            data = yaml.safe_load(f)
        return data.get("issues", [])

    def diagnose(self, logs: str) -> list[Diagnosis]:
        """Return all matching diagnoses for the given error logs."""
        results = []
        for regex, issue in self._compiled:
            m = regex.search(logs)
            if m:
                results.append(Diagnosis(
                    issue_id=issue["id"],
                    pattern=issue["pattern"],
                    severity=issue["severity"],
                    description=issue["description"],
                    fix_action=issue["fix"],
                    matched_text=m.group(0)[:200],
                ))
        results.sort(key=lambda d: {"high": 0, "medium": 1, "low": 2}.get(d.severity, 3))
        return results

    def apply_fix(self, config: ServerConfig, diagnosis: Diagnosis) -> ServerConfig:
        """Return a new ServerConfig with the fix for the given diagnosis applied."""
        action = diagnosis.fix_action
        handler = self._fix_handlers.get(action)
        if handler is None:
            log.warning("No handler for fix action: %s", action)
            return config
        log.info("Applying fix [%s]: %s", diagnosis.issue_id, action)
        return handler(config, diagnosis)

    # ------------------------------------------------------------------
    # Fix handlers — each returns a new ServerConfig
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_fallback_triton(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(attention_backend="triton")

    @staticmethod
    def _fix_disable_cuda_graph(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(disable_cuda_graph=True)

    @staticmethod
    def _fix_disable_cuda_graph_or_fp8(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        if config.quantization == "fp8":
            return config.clone(quantization=None, disable_cuda_graph=True)
        return config.clone(disable_cuda_graph=True)

    @staticmethod
    def _fix_reduce_memory(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        new_frac = max(0.60, config.mem_fraction_static - 0.10)
        return config.clone(mem_fraction_static=new_frac)

    @staticmethod
    def _fix_lower_mem_fraction(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        new_frac = max(0.60, config.mem_fraction_static - 0.05)
        return config.clone(mem_fraction_static=new_frac)

    @staticmethod
    def _fix_disable_fp8(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(quantization=None)

    @staticmethod
    def _fix_configure_nccl(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        env = dict(config.env_overrides)
        env["NCCL_SOCKET_IFNAME"] = "eth0"
        env["NCCL_DEBUG"] = "INFO"
        return config.clone(env_overrides=env)

    @staticmethod
    def _fix_install_aiter(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        # Flag that aiter needs installing — pipeline will docker exec pip install
        env = dict(config.env_overrides)
        env["__PORTER_INSTALL_AITER"] = "1"
        return config.clone(env_overrides=env)

    @staticmethod
    def _fix_verify_cuda_guard(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(attention_backend="triton")

    @staticmethod
    def _fix_verify_triton_fallback(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(attention_backend="triton")

    @staticmethod
    def _fix_v_head_dim_init(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(attention_backend="triton", disable_cuda_graph=True)

    @staticmethod
    def _fix_upgrade_transformers(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        env = dict(config.env_overrides)
        env["__PORTER_UPGRADE_TRANSFORMERS"] = "1"
        return config.clone(env_overrides=env)

    @staticmethod
    def _fix_set_nsa_tilelang(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        return config.clone(nsa_prefill_backend="tilelang", nsa_decode_backend="tilelang")

    @staticmethod
    def _fix_escalate(config: ServerConfig, _diag: Diagnosis) -> ServerConfig:
        env = dict(config.env_overrides)
        env["__PORTER_ESCALATE_TO_AGENT"] = "1"
        return config.clone(env_overrides=env)

    _fix_handlers: dict = {}


# Register handlers by action name
ErrorDiagnoser._fix_handlers = {
    "fallback_triton_backend": ErrorDiagnoser._fix_fallback_triton,
    "disable_cuda_graph": ErrorDiagnoser._fix_disable_cuda_graph,
    "disable_cuda_graph_or_fp8": ErrorDiagnoser._fix_disable_cuda_graph_or_fp8,
    "reduce_memory": ErrorDiagnoser._fix_reduce_memory,
    "lower_mem_fraction": ErrorDiagnoser._fix_lower_mem_fraction,
    "disable_fp8": ErrorDiagnoser._fix_disable_fp8,
    "configure_nccl": ErrorDiagnoser._fix_configure_nccl,
    "install_aiter": ErrorDiagnoser._fix_install_aiter,
    "verify_cuda_guard": ErrorDiagnoser._fix_verify_cuda_guard,
    "verify_triton_fallback": ErrorDiagnoser._fix_verify_triton_fallback,
    "fix_v_head_dim_init": ErrorDiagnoser._fix_v_head_dim_init,
    "upgrade_transformers": ErrorDiagnoser._fix_upgrade_transformers,
    "set_nsa_tilelang": ErrorDiagnoser._fix_set_nsa_tilelang,
    "escalate_to_agent": ErrorDiagnoser._fix_escalate,
}
