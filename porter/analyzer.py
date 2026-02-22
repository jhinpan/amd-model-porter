"""Stage 0 — Model Analyzer.

Downloads config.json from HuggingFace, parses architecture features,
estimates VRAM, predicts AMD-specific issues, and generates a recommended
ServerConfig for the first launch attempt.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from huggingface_hub import hf_hub_download, HfApi

from porter.config import GPU_SPECS, DEFAULT_GPU, MODEL_WEIGHTS_CONTAINER_DIR
from porter.server_config import ServerConfig

log = logging.getLogger(__name__)

KNOWN_ISSUES_PATH = Path(__file__).parent / "known_issues.yaml"


# ---------------------------------------------------------------------------
# ModelProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Parsed architecture summary of a HuggingFace model."""

    model_id: str
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)

    # Scale
    num_params_billion: float = 0.0
    num_active_params_billion: float = 0.0
    hidden_size: int = 0
    num_layers: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0

    # Attention
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    v_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    max_position_embeddings: int = 0

    # Features
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_token: int = 0
    num_shared_experts: int = 0
    first_k_dense_layers: int = 0

    has_mla: bool = False
    has_deltanet: bool = False
    has_hybrid_attention: bool = False
    has_nsa: bool = False  # Native Sparse Attention / DSA
    has_mtp: bool = False

    attention_type: str = "unknown"  # GQA, MHA, MLA, hybrid

    # VRAM estimate
    weight_size_gb: float = 0.0
    min_gpus: int = 1

    # Predictions
    predicted_issues: list[dict] = field(default_factory=list)
    recommended_config: Optional[ServerConfig] = None

    # Raw config for reference
    raw_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ModelAnalyzer:
    """Download and analyze a HuggingFace model's config."""

    def __init__(self, gpu_type: str = DEFAULT_GPU):
        self.gpu_spec = GPU_SPECS[gpu_type]
        self.known_issues = self._load_known_issues()

    @staticmethod
    def _load_known_issues() -> list[dict]:
        path = KNOWN_ISSUES_PATH
        if not path.exists():
            return []
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("issues", [])

    def analyze(self, model_id: str) -> ModelProfile:
        """Run full analysis on a HuggingFace model ID (e.g. 'zai-org/GLM-5-FP8')."""
        log.info("Analyzing model: %s", model_id)
        config = self._download_config(model_id)
        profile = self._parse_config(model_id, config)
        profile.predicted_issues = self._predict_issues(profile)
        profile.recommended_config = self._recommend_config(profile)
        return profile

    # ------------------------------------------------------------------
    # Config download
    # ------------------------------------------------------------------

    @staticmethod
    def _download_config(model_id: str) -> dict:
        """Download config.json from HuggingFace (no full model download)."""
        import requests as _requests

        # Try direct HTTP first (no auth needed for public models)
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        try:
            r = _requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

        # Fall back to huggingface_hub
        try:
            path = hf_hub_download(model_id, "config.json")
            with open(path) as f:
                return json.load(f)
        except Exception:
            log.warning("Could not download config.json, trying HF API")
            api = HfApi()
            info = api.model_info(model_id)
            if info.config:
                return info.config
            raise RuntimeError(f"Cannot fetch config for {model_id}")

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_config(self, model_id: str, config: dict) -> ModelProfile:
        p = ModelProfile(model_id=model_id, raw_config=config)

        p.model_type = config.get("model_type", "")
        p.architectures = config.get("architectures", [])

        p.hidden_size = config.get("hidden_size", 0)
        p.num_layers = config.get("num_hidden_layers", 0)
        p.intermediate_size = config.get("intermediate_size", 0)
        p.vocab_size = config.get("vocab_size", 0)
        p.max_position_embeddings = config.get("max_position_embeddings", 0)

        # Attention heads
        p.num_attention_heads = config.get("num_attention_heads", 0)
        p.num_kv_heads = config.get("num_key_value_heads", p.num_attention_heads)
        p.head_dim = config.get("head_dim", 0)
        if p.head_dim == 0 and p.num_attention_heads > 0:
            p.head_dim = p.hidden_size // p.num_attention_heads

        # MLA fields
        p.v_head_dim = config.get("v_head_dim")
        p.qk_nope_head_dim = config.get("qk_nope_head_dim")
        p.qk_rope_head_dim = config.get("qk_rope_head_dim")
        p.kv_lora_rank = config.get("kv_lora_rank")
        p.q_lora_rank = config.get("q_lora_rank")
        p.has_mla = p.kv_lora_rank is not None and p.kv_lora_rank > 0

        # MoE
        p.num_experts = config.get("n_routed_experts", 0) or config.get("num_local_experts", 0)
        p.num_experts_per_token = (
            config.get("num_experts_per_tok", 0) or config.get("num_selected_experts", 0)
        )
        p.num_shared_experts = config.get("n_shared_experts", 0)
        p.first_k_dense_layers = config.get("first_k_dense_replace", 0)
        p.is_moe = p.num_experts > 1

        # Hybrid attention (DeltaNet, DSA/NSA)
        p.has_deltanet = any(
            k in config for k in ("delta_net_config", "hybrid_attention_pattern")
        ) or "DeltaNet" in str(p.architectures)
        p.has_nsa = any(
            k in config for k in ("index_head_dim", "nsa_config", "indexer_rope_interleave")
        ) or "Dsa" in str(p.architectures)
        p.has_hybrid_attention = p.has_deltanet or p.has_nsa
        p.has_mtp = config.get("num_nextn_predict_layers", 0) > 0 or "mtp" in str(config).lower()

        # Attention type classification
        if p.has_mla:
            p.attention_type = "MLA"
        elif p.has_hybrid_attention:
            p.attention_type = "hybrid"
        elif p.num_kv_heads < p.num_attention_heads:
            p.attention_type = "GQA"
        else:
            p.attention_type = "MHA"

        # Parameter / VRAM estimation
        p.num_params_billion = self._estimate_params(config, p) / 1e9
        p.num_active_params_billion = self._estimate_active_params(config, p) / 1e9
        p.weight_size_gb = self._estimate_weight_size(p)
        p.min_gpus = self._estimate_min_gpus(p)

        return p

    @staticmethod
    def _estimate_params(config: dict, p: ModelProfile) -> float:
        """Rough total parameter count from config dimensions."""
        if "num_parameters" in config:
            return config["num_parameters"]
        H = p.hidden_size
        L = p.num_layers
        V = p.vocab_size
        I = p.intermediate_size
        if H == 0 or L == 0:
            return 0
        # Embedding + LM head
        embed = V * H * 2
        # Per-layer: attention + FFN (rough)
        attn_per_layer = 4 * H * H  # Q, K, V, O projections
        if p.is_moe:
            ffn_per_layer = p.num_experts * 3 * H * I + p.num_shared_experts * 3 * H * I
        else:
            ffn_per_layer = 3 * H * I  # gate + up + down
        return embed + L * (attn_per_layer + ffn_per_layer)

    @staticmethod
    def _estimate_active_params(config: dict, p: ModelProfile) -> float:
        if not p.is_moe:
            return p.num_params_billion * 1e9
        H = p.hidden_size
        L = p.num_layers
        V = p.vocab_size
        I = p.intermediate_size
        if H == 0 or L == 0:
            return 0
        embed = V * H * 2
        attn_per_layer = 4 * H * H
        ffn_active = p.num_experts_per_token * 3 * H * I + p.num_shared_experts * 3 * H * I
        return embed + L * (attn_per_layer + ffn_active)

    def _estimate_weight_size(self, p: ModelProfile) -> float:
        """Estimate weight size in GB (bf16 = 2 bytes per param, fp8 = 1 byte)."""
        is_fp8 = "fp8" in p.model_id.lower() or "FP8" in p.model_id
        bytes_per_param = 1.0 if is_fp8 else 2.0
        return p.num_params_billion * 1e9 * bytes_per_param / (1024**3)

    def _estimate_min_gpus(self, p: ModelProfile) -> int:
        """Minimum GPUs needed to fit model weights + some KV cache overhead."""
        usable_per_gpu = self.gpu_spec.hbm_gb * 0.75
        overhead_factor = 1.2  # 20% for KV cache, activations
        needed = p.weight_size_gb * overhead_factor
        return max(1, math.ceil(needed / usable_per_gpu))

    # ------------------------------------------------------------------
    # Issue prediction
    # ------------------------------------------------------------------

    def _predict_issues(self, p: ModelProfile) -> list[dict]:
        """Cross-reference model features against known issues."""
        predicted = []
        for issue in self.known_issues:
            affected = issue.get("models_affected", ["all"])
            if "all" in affected:
                match = True
            else:
                match = any([
                    "MoE" in affected and p.is_moe,
                    "MLA" in affected and p.has_mla,
                    "DeltaNet" in affected and p.has_deltanet,
                    "hybrid_attention" in affected and p.has_hybrid_attention,
                    "NSA" in affected and p.has_nsa,
                    "new_models" in affected,
                ])
            if match:
                predicted.append({
                    "id": issue["id"],
                    "severity": issue["severity"],
                    "description": issue["description"],
                    "fix": issue["fix"],
                })
        return predicted

    # ------------------------------------------------------------------
    # Config recommendation
    # ------------------------------------------------------------------

    def _recommend_config(self, p: ModelProfile) -> ServerConfig:
        """Generate conservative first-launch config based on analysis."""
        tp = max(p.min_gpus, 8)  # MI355X: usually need all 8
        tp = min(tp, 8)  # cap at 8 for single-node

        config = ServerConfig(
            model_path=f"{MODEL_WEIGHTS_CONTAINER_DIR}/{p.model_id.split('/')[-1]}",
            tp=tp,
            attention_backend="triton",
            disable_cuda_graph=True,
            mem_fraction_static=0.75,
            gpu_ids=list(range(tp)),
        )

        # Hybrid attention models need mamba cache size
        if p.has_deltanet or p.has_hybrid_attention:
            config.max_mamba_cache_size = 64

        # NSA/DSA models need explicit tilelang backends on ROCm
        if p.has_nsa:
            config.nsa_prefill_backend = "tilelang"
            config.nsa_decode_backend = "tilelang"

        # Lower mem fraction for very large models
        if p.weight_size_gb > 700:
            config.mem_fraction_static = 0.70

        return config

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def format_report(self, profile: ModelProfile) -> str:
        """Format a human-readable analysis report."""
        lines = [
            f"{'='*60}",
            f"Model Analysis: {profile.model_id}",
            f"{'='*60}",
            "",
            "Architecture:",
            f"  Type:        {profile.model_type}",
            f"  Class:       {', '.join(profile.architectures)}",
            f"  Params:      {profile.num_params_billion:.1f}B total"
            + (f", {profile.num_active_params_billion:.1f}B active" if profile.is_moe else ""),
            f"  Layers:      {profile.num_layers}",
            f"  Hidden:      {profile.hidden_size}",
            f"  Attn Heads:  {profile.num_attention_heads} Q / {profile.num_kv_heads} KV",
            f"  Head Dim:    {profile.head_dim}",
            f"  Attn Type:   {profile.attention_type}",
        ]

        if profile.has_mla:
            lines.extend([
                f"  v_head_dim:  {profile.v_head_dim}",
                f"  qk_nope:     {profile.qk_nope_head_dim}",
                f"  kv_lora:     {profile.kv_lora_rank}",
            ])

        if profile.is_moe:
            lines.extend([
                "",
                "MoE:",
                f"  Experts:     {profile.num_experts} total,"
                f" {profile.num_experts_per_token} active,"
                f" {profile.num_shared_experts} shared",
                f"  Dense layers:{profile.first_k_dense_layers}",
            ])

        features = []
        if profile.has_deltanet:
            features.append("DeltaNet")
        if profile.has_nsa:
            features.append("NSA/DSA")
        if profile.has_mtp:
            features.append("MTP")
        if features:
            lines.extend(["", f"  Features:    {', '.join(features)}"])

        lines.extend([
            "",
            "VRAM Estimate:",
            f"  Weights:     {profile.weight_size_gb:.0f} GB",
            f"  Min GPUs:    {profile.min_gpus}x {self.gpu_spec.name}",
            "",
        ])

        if profile.predicted_issues:
            lines.append(f"Predicted Issues ({len(profile.predicted_issues)}):")
            for issue in profile.predicted_issues:
                sev = issue['severity'].upper()
                lines.append(f"  [{sev}] {issue['id']}: {issue['description']}")
            lines.append("")

        if profile.recommended_config:
            lines.extend([
                "Recommended First Launch:",
                f"  {profile.recommended_config.summary()}",
                "",
            ])

        return "\n".join(lines)
