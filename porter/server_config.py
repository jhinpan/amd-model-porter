"""ServerConfig dataclass representing a complete SGLang launch configuration."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Optional

from porter.config import AMD_ENV_VARS, SGLANG_DEFAULT_PORT


@dataclass
class ServerConfig:
    """Full configuration for launching an SGLang server on AMD GPUs."""

    model_path: str
    tp: int = 8
    ep: Optional[int] = None
    dp: Optional[int] = None
    attention_backend: str = "triton"
    decode_attention_backend: Optional[str] = None
    trust_remote_code: bool = True
    disable_cuda_graph: bool = False
    cuda_graph_max_bs: Optional[int] = None
    mem_fraction_static: float = 0.80
    max_mamba_cache_size: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    quantization: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = SGLANG_DEFAULT_PORT
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    speculative_algorithm: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_draft_attention_backend: Optional[str] = None
    extra_args: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    gpu_ids: Optional[list[int]] = None

    def build_env(self) -> dict[str, str]:
        """Return the full environment variable dict for the server process."""
        from porter.config import MODEL_WEIGHTS_CONTAINER_DIR

        env = dict(AMD_ENV_VARS)
        if self.gpu_ids is not None:
            env["HIP_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
        else:
            env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.tp))
        env["HF_HUB_CACHE"] = f"{MODEL_WEIGHTS_CONTAINER_DIR}/hub"
        env.update(self.env_overrides)
        return env

    def build_cmd(self) -> list[str]:
        """Return the SGLang launch command as a list of tokens."""
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--tp", str(self.tp),
            "--attention-backend", self.attention_backend,
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.ep is not None:
            cmd.extend(["--ep", str(self.ep)])
        if self.dp is not None:
            cmd.extend(["--dp", str(self.dp)])
            cmd.append("--enable-dp-attention")
            cmd.append("--enable-dp-lm-head")
        if self.decode_attention_backend:
            cmd.extend(["--decode-attention-backend", self.decode_attention_backend])
        if self.disable_cuda_graph:
            cmd.append("--disable-cuda-graph")
        if self.cuda_graph_max_bs is not None:
            cmd.extend(["--cuda-graph-max-bs", str(self.cuda_graph_max_bs)])
        if self.max_mamba_cache_size is not None:
            cmd.extend(["--max-mamba-cache-size", str(self.max_mamba_cache_size)])
        if self.chunked_prefill_size is not None:
            cmd.extend(["--chunked-prefill-size", str(self.chunked_prefill_size)])
        if self.quantization:
            cmd.extend(["--quantization", self.quantization])
        if self.tool_call_parser:
            cmd.extend(["--tool-call-parser", self.tool_call_parser])
        if self.reasoning_parser:
            cmd.extend(["--reasoning-parser", self.reasoning_parser])
        if self.speculative_algorithm:
            cmd.extend(["--speculative-algorithm", self.speculative_algorithm])
        if self.speculative_num_steps is not None:
            cmd.extend(["--speculative-num-steps", str(self.speculative_num_steps)])
        if self.speculative_eagle_topk is not None:
            cmd.extend(["--speculative-eagle-topk", str(self.speculative_eagle_topk)])
        if self.speculative_num_draft_tokens is not None:
            cmd.extend(["--speculative-num-draft-tokens", str(self.speculative_num_draft_tokens)])
        if self.speculative_draft_attention_backend:
            cmd.extend([
                "--speculative-draft-attention-backend",
                self.speculative_draft_attention_backend,
            ])
        cmd.extend(self.extra_args)
        return cmd

    def build_shell_cmd(self) -> str:
        """Return the full shell command string including env vars."""
        env = self.build_env()
        env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
        cmd_str = " ".join(shlex.quote(t) for t in self.build_cmd())
        return f"{env_str} {cmd_str}"

    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        parts = [f"backend={self.attention_backend}", f"tp={self.tp}"]
        if self.ep:
            parts.append(f"ep={self.ep}")
        if self.dp:
            parts.append(f"dp={self.dp}")
        parts.append(f"mem={self.mem_fraction_static}")
        if self.max_mamba_cache_size:
            parts.append(f"mamba={self.max_mamba_cache_size}")
        if self.disable_cuda_graph:
            parts.append("no-cuda-graph")
        return ", ".join(parts)

    def clone(self, **overrides) -> "ServerConfig":
        """Return a copy with selective field overrides."""
        import dataclasses
        return dataclasses.replace(self, **overrides)
