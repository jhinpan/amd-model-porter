# AMD Model Porter

Automated pipeline for porting HuggingFace models to AMD MI300X/MI355X GPUs via SGLang ROCm Docker.

## What It Does

1. **Analyzes** a HuggingFace model (architecture, MoE/MLA/GQA detection, VRAM estimation)
2. **Deploys** it inside a SGLang ROCm Docker container on AMD GPUs
3. **Auto-fixes** launch failures using a catalog of 18+ known AMD/ROCm error patterns
4. **Escalates** unknown errors to a Claude Code agent running inside the container
5. **Grid-searches** optimal backend, parallelism, and memory configs across 3 tiers
6. **Benchmarks** throughput, latency, and correctness
7. **Serves** results via a live web dashboard

## Quick Start

```bash
# Install
pip install -e .

# Analyze a model
porter analyze Qwen/Qwen3.5-397B-A17B

# Run full pipeline
porter run zai-org/GLM-5-FP8

# Start web UI
porter web --port 8080
```

## Requirements

- AMD MI300X or MI355X GPUs
- Docker with ROCm support
- SGLang ROCm Docker image (`rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260221`)
- Python 3.10+

## Architecture

```
User submits HF URL
        │
        ▼
┌─ Stage 0: Model Analysis ──────────────┐
│  Download config.json, detect arch,    │
│  predict AMD issues, recommend config  │
└────────────────────────────────────────┘
        │
        ▼
┌─ Stage 1: Docker Setup ───────────────┐
│  Pull image, create container,        │
│  mount weights, set AMD env vars      │
└───────────────────────────────────────┘
        │
        ▼
┌─ Stage 2: Launch + Auto-Fix ──────────┐
│  Start SGLang → health check →        │
│  diagnose errors → apply fix → retry  │
│  If unknown: Claude Code agent debug  │
└───────────────────────────────────────┘
        │
        ▼
┌─ Stage 3: Grid Search Benchmark ──────┐
│  Tier 1: Backend sweep                │
│  Tier 2: Parallelism sweep            │
│  Tier 3: Memory tuning               │
└───────────────────────────────────────┘
        │
        ▼
┌─ Stage 4: Results ────────────────────┐
│  Best docker run command, benchmark   │
│  table, fixes applied, web dashboard  │
└───────────────────────────────────────┘
```

## Configuration

Set these environment variables for the Claude Code agent integration:

```bash
export AMD_LLM_API_KEY="your-subscription-key"
python3 amd_proxy.py  # Start AMD LLM proxy on localhost:8082
```

## License

MIT
