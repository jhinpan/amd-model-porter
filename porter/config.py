"""Global constants for AMD Model Porter."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Docker images
# ---------------------------------------------------------------------------
DOCKER_IMAGES = {
    "mi355x-rocm700": "rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260221",
    "mi355x-rocm720": "rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260221",
}
DEFAULT_DOCKER_IMAGE = DOCKER_IMAGES["mi355x-rocm700"]

SETUP_GIST_URL = (
    "https://gist.githubusercontent.com/jhinpan/501ed5c3dae590dcb01f3b13ff191108"
    "/raw/setup-dev-env.sh"
)

# ---------------------------------------------------------------------------
# AMD environment variables (applied to every SGLang container)
# ---------------------------------------------------------------------------
AMD_ENV_VARS: dict[str, str] = {
    "SGLANG_USE_AITER": "1",
    "AITER_ONLINE_TUNE": "1",
    "HIP_FORCE_DEV_KERNARG": "1",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    "SGLANG_MOE_PADDING": "1",
    "NCCL_MIN_NCHANNELS": "112",
    "VLLM_FP8_PADDING": "1",
    "VLLM_FP8_ACT_PADDING": "1",
    "VLLM_FP8_WEIGHT_PADDING": "1",
}

# ---------------------------------------------------------------------------
# Hardware specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUSpec:
    name: str
    hbm_gb: float
    tdp_watts: int
    fp16_tflops: float

GPU_SPECS: dict[str, GPUSpec] = {
    "mi300x": GPUSpec("MI300X", hbm_gb=192, tdp_watts=750, fp16_tflops=1307),
    "mi355x": GPUSpec("MI355X", hbm_gb=288, tdp_watts=1400, fp16_tflops=1800),
}
DEFAULT_GPU = "mi355x"

# ---------------------------------------------------------------------------
# Model weight storage
# ---------------------------------------------------------------------------
MODEL_WEIGHTS_HOST_DIR = "/mnt/dcgpuval/huggingface"
MODEL_WEIGHTS_CONTAINER_DIR = "/sgl-workspace/models"

# ---------------------------------------------------------------------------
# Container docker run defaults
# ---------------------------------------------------------------------------
DOCKER_RUN_DEFAULTS: dict[str, str | list[str]] = {
    "cap_add": "SYS_PTRACE",
    "ipc": "host",
    "privileged": "true",
    "shm_size": "128g",
    "network": "host",
    "devices": ["/dev/kfd", "/dev/dri"],
    "group_add": "video",
}

# ---------------------------------------------------------------------------
# SGLang defaults
# ---------------------------------------------------------------------------
SGLANG_DEFAULT_PORT = 30000
SGLANG_HEALTH_ENDPOINT = "/health"
SGLANG_HEALTH_TIMEOUT = 300
SGLANG_MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Claude Code agent
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-opus-4-6"
ANTHROPIC_BASE_URL = "http://localhost:8082"
AGENT_TIMEOUT = 1800  # 30 min max for agent debugging

# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    num_prompts: int
    input_len: int
    output_len: int
    request_rate: str  # "inf" or float as string
    concurrency: int

BENCHMARK_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario("throughput", 256, 1024, 256, "inf", 256),
    BenchmarkScenario("balanced", 192, 1024, 256, "8", 64),
    BenchmarkScenario("burst", 300, 128, 128, "inf", 300),
]

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
ATTENTION_BACKENDS = ["triton", "aiter"]
PARALLELISM_CONFIGS = [
    {"tp": 8},
    {"tp": 8, "ep": 8},
    {"tp": 8, "ep": 4},
]
MEM_FRACTIONS = [0.75, 0.80, 0.85, 0.90]
CHUNKED_PREFILL_SIZES = [4096, 8192, 16384]

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = "porter_results.db"
