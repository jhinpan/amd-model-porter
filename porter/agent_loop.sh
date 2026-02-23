#!/bin/bash
# porter_agent_loop.sh — Autonomous Claude Code debugging loop
# Runs inside a Docker container. Iterates: launch -> health check -> if fail: Claude Code fix -> repeat.
set -o pipefail

MODEL_PATH="${MODEL_PATH:-zai-org/GLM-5-FP8}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
PORT="${PORT:-30000}"
WORK_DIR="/tmp/porter_agent"
SGLANG_DIR="/sgl-workspace/sglang"

mkdir -p "$WORK_DIR"

# Default launch config (conservative: triton backend, no CUDA graphs)
cat > "$WORK_DIR/current_config.sh" << 'DEFAULTCFG'
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_CACHE=/sgl-workspace/models/hub \
HF_HUB_OFFLINE=1 \
SGLANG_USE_AITER=1 \
AITER_ONLINE_TUNE=1 \
HIP_FORCE_DEV_KERNARG=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
SGLANG_MOE_PADDING=1 \
NCCL_MIN_NCHANNELS=112 \
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5-FP8 \
  --tp 8 \
  --attention-backend triton \
  --nsa-prefill-backend tilelang \
  --nsa-decode-backend tilelang \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --disable-cuda-graph \
  --host 0.0.0.0 --port 30000
DEFAULTCFG

kill_server() {
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    sleep 3
    pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
    pkill -9 -f 'sglang::scheduler' 2>/dev/null || true
    sleep 2
}

launch_server() {
    kill_server
    echo "[LOOP] Launching server with config:"
    cat "$WORK_DIR/current_config.sh"
    echo ""
    nohup bash "$WORK_DIR/current_config.sh" > /tmp/sglang_server.log 2>&1 &
    echo $! > "$WORK_DIR/server_pid"
}

wait_for_health() {
    local timeout=$1
    local deadline=$((SECONDS + timeout))
    echo "[LOOP] Waiting for health at localhost:$PORT (timeout=${timeout}s)..."

    while [ $SECONDS -lt $deadline ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "[LOOP] SERVER HEALTHY!"
            return 0
        fi
        # Check if python process crashed
        if ! ps aux | grep '[p]ython.*sglang' > /dev/null 2>&1; then
            echo "[LOOP] Server process crashed (not running)"
            return 1
        fi
        sleep 5
    done
    echo "[LOOP] Health check timed out after ${timeout}s"
    return 1
}

collect_error_report() {
    local round=$1
    local report="$WORK_DIR/round_${round}_error.txt"
    {
        echo "=== SERVER LOG (last 150 lines) ==="
        tail -150 /tmp/sglang_server.log 2>/dev/null
        echo ""
        echo "=== DMESG (last 20 lines) ==="
        dmesg 2>/dev/null | tail -20
        echo ""
        echo "=== GPU STATE ==="
        rocm-smi 2>/dev/null | head -20
        echo ""
        echo "=== CURRENT CONFIG ==="
        cat "$WORK_DIR/current_config.sh"
    } > "$report"
    echo "$report"
}

build_round_history() {
    local current_round=$1
    local history=""
    for ((r=1; r<current_round; r++)); do
        if [ -f "$WORK_DIR/round_${r}_report.txt" ]; then
            history+="
--- Round $r ---
$(cat "$WORK_DIR/round_${r}_report.txt")
"
        fi
    done
    echo "$history"
}

run_claude_round() {
    local round=$1
    local error_report=$2
    local history=$(build_round_history "$round")

    local error_content
    error_content=$(cat "$error_report")

    cat > "$WORK_DIR/round_${round}_prompt.txt" << PROMPT_EOF
You are an AMD ROCm engineer debugging a SGLang server launch failure on 8x MI355X GPUs.

MODEL: GLM-5-FP8 (zai-org/GLM-5-FP8)
  - 744B MoE (256 experts, 8 active), MLA + NSA/DSA attention, FP8 weights
  - Architecture: GlmMoeDsaForCausalLM (inherits DeepseekV2ForCausalLM in SGLang)

ROUND: $round of $MAX_ROUNDS
SGLang source: $SGLANG_DIR/python/sglang/

PREVIOUS ROUNDS:
$history

CURRENT ERROR REPORT:
$error_content

AVAILABLE BACKENDS TO TRY:
- attention: triton, aiter
- nsa-prefill: tilelang, triton
- nsa-decode: tilelang, triton

YOUR TASK:
1. Read the error log carefully. Identify the root cause.
2. Look at the SGLang source code at $SGLANG_DIR/python/sglang/ to find the failing code path.
3. Apply ONE focused fix: either edit source code OR change the launch config.
4. Write the new launch command to $WORK_DIR/current_config.sh
5. Write a brief report of what you changed and why to $WORK_DIR/round_${round}_report.txt

IMPORTANT RULES:
- ONLY change what is necessary. Do not refactor unrelated code.
- The model path is zai-org/GLM-5-FP8 (resolved via HF_HUB_CACHE)
- Always keep --trust-remote-code and --host 0.0.0.0 --port 30000
- If you edit SGLang source, note the exact file and line changed.
- If a backend crashes, try a different one in the config.
- The server log is at /tmp/sglang_server.log

Use /amd-rocm-porting skill for ROCm-specific guidance.
PROMPT_EOF

    echo "[LOOP] Running Claude Code round $round..."
    ANTHROPIC_BASE_URL=http://localhost:8082 \
    ANTHROPIC_API_KEY=not-used \
    claude --model claude-opus-4-6 \
        --dangerously-skip-permissions \
        -p "$(cat "$WORK_DIR/round_${round}_prompt.txt")" \
        > "$WORK_DIR/round_${round}_claude_output.txt" 2>&1

    local exit_code=$?
    echo "[LOOP] Claude Code round $round exited with code $exit_code"

    # If Claude didn't write a report, create one from its output
    if [ ! -f "$WORK_DIR/round_${round}_report.txt" ]; then
        head -20 "$WORK_DIR/round_${round}_claude_output.txt" > "$WORK_DIR/round_${round}_report.txt"
    fi
}

run_benchmarks() {
    echo "[LOOP] Running benchmarks..."
    # BSZ=1 decode throughput
    HF_HUB_OFFLINE=1 python3 -m sglang.bench_serving \
        --backend sglang --base-url "http://127.0.0.1:$PORT" \
        --dataset-name random \
        --num-prompts 5 --random-input-len 1024 --random-output-len 256 \
        --request-rate inf --max-concurrency 1 \
        --model "$MODEL_PATH" --disable-stream \
        > "$WORK_DIR/benchmark_bsz1.txt" 2>&1
    echo "[LOOP] BSZ=1 benchmark results:"
    cat "$WORK_DIR/benchmark_bsz1.txt" | tail -20

    # Throughput benchmark
    HF_HUB_OFFLINE=1 python3 -m sglang.bench_serving \
        --backend sglang --base-url "http://127.0.0.1:$PORT" \
        --dataset-name random \
        --num-prompts 64 --random-input-len 1024 --random-output-len 256 \
        --request-rate inf --max-concurrency 64 \
        --model "$MODEL_PATH" --disable-stream \
        > "$WORK_DIR/benchmark_throughput.txt" 2>&1
    echo "[LOOP] Throughput benchmark results:"
    cat "$WORK_DIR/benchmark_throughput.txt" | tail -20
}

# ============================================================
# MAIN LOOP
# ============================================================
echo "========================================"
echo " AMD Model Porter - Agent Loop"
echo " Model: $MODEL_PATH"
echo " Max Rounds: $MAX_ROUNDS"
echo " Health Timeout: ${HEALTH_TIMEOUT}s"
echo "========================================"

for ((ROUND=1; ROUND<=MAX_ROUNDS; ROUND++)); do
    echo ""
    echo "========================================"
    echo " ROUND $ROUND / $MAX_ROUNDS"
    echo "========================================"

    launch_server
    if wait_for_health "$HEALTH_TIMEOUT"; then
        echo "[LOOP] Server is HEALTHY on round $ROUND!"

        # Correctness check
        echo "[LOOP] Running correctness check..."
        RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"GLM-5-FP8","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":32,"temperature":0.0}')
        echo "[LOOP] Correctness response: $RESPONSE"

        # Run benchmarks
        run_benchmarks

        echo ""
        echo "========================================"
        echo " SUCCESS! Server healthy on round $ROUND"
        echo "========================================"
        echo "Final config:"
        cat "$WORK_DIR/current_config.sh"
        exit 0
    fi

    # Collect error info
    ERROR_REPORT=$(collect_error_report "$ROUND")
    echo "[LOOP] Error report saved to: $ERROR_REPORT"

    # Kill the crashed server
    kill_server

    # Run Claude Code to analyze and fix
    run_claude_round "$ROUND" "$ERROR_REPORT"

    echo "[LOOP] Round $ROUND complete. Proceeding to next round..."
done

echo ""
echo "========================================"
echo " FAILED: Exhausted all $MAX_ROUNDS rounds"
echo "========================================"
echo "Summary of all rounds:"
for ((r=1; r<=MAX_ROUNDS; r++)); do
    if [ -f "$WORK_DIR/round_${r}_report.txt" ]; then
        echo "--- Round $r ---"
        cat "$WORK_DIR/round_${r}_report.txt"
    fi
done
exit 1
