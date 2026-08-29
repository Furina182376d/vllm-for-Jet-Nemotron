#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可通过环境变量覆盖，例如：
# GPU_ID=0 RUNS=5 MAX_TOKENS=128 ./test.sh both
GPU_ID="${GPU_ID:-1}"
HF_ENV="${HF_ENV:-vllm_old}"
VLLM_ENV="${VLLM_ENV:-vllm}"
MAX_TOKENS="${MAX_TOKENS:-64}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
RUNS="${RUNS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

CONCURRENCIES=(1 4 8 16 32)

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="${SCRIPT_DIR}/benchmark-results/${TIMESTAMP}"
SUMMARY_FILE="${RESULT_DIR}/summary.tsv"

mkdir -p "$RESULT_DIR"

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "backend" \
    "num_prompts" \
    "avg_latency_s" \
    "requests_per_s" \
    "prompt_tokens_per_s" \
    "output_tokens_per_s" \
    > "$SUMMARY_FILE"

run_hf() {
    local num_prompts="$1"
    local log_file="${RESULT_DIR}/hf-n${num_prompts}.log"

    echo
    echo "Running HF: num_prompts=${num_prompts}, batch_size=1"
    echo "Log: ${log_file}"

    env \
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        PYTHONUNBUFFERED=1 \
        conda run --no-capture-output -n "$HF_ENV" \
        python "$SCRIPT_DIR/test-jet.py" \
            --num-prompts "$num_prompts" \
            --batch-size 1 \
            --max-tokens "$MAX_TOKENS" \
            --warmup-runs "$WARMUP_RUNS" \
            --runs "$RUNS" \
        2>&1 | tee "$log_file"

    append_summary "hf" "$num_prompts" "$log_file"
}

run_vllm() {
    local num_prompts="$1"
    local log_file="${RESULT_DIR}/vllm-n${num_prompts}.log"

    echo
    echo "Running vLLM: num_prompts=${num_prompts}"
    echo "Log: ${log_file}"

    env \
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        PYTHONUNBUFFERED=1 \
        conda run --no-capture-output -n "$VLLM_ENV" \
        python "$SCRIPT_DIR/test-vllm.py" \
            --num-prompts "$num_prompts" \
            --max-tokens "$MAX_TOKENS" \
            --warmup-runs "$WARMUP_RUNS" \
            --runs "$RUNS" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        2>&1 | tee "$log_file"

    append_summary "vllm" "$num_prompts" "$log_file"
}

append_summary() {
    local backend="$1"
    local num_prompts="$2"
    local log_file="$3"

    local latency
    local aggregate
    local requests_per_s
    local prompt_tokens_per_s
    local output_tokens_per_s

    latency="$(
        sed -nE \
            's/^Average workload latency: ([0-9.]+) s$/\1/p' \
            "$log_file" |
        tail -n 1
    )"

    aggregate="$(
        grep -F "Aggregate throughput:" "$log_file" |
        tail -n 1
    )"

    requests_per_s="$(
        printf '%s\n' "$aggregate" |
        sed -nE \
            's/.*Aggregate throughput: ([0-9.]+) req\/s.*/\1/p'
    )"

    prompt_tokens_per_s="$(
        printf '%s\n' "$aggregate" |
        sed -nE \
            's/.*req\/s, ([0-9.]+) prompt tok\/s.*/\1/p'
    )"

    output_tokens_per_s="$(
        printf '%s\n' "$aggregate" |
        sed -nE \
            's/.*prompt tok\/s, ([0-9.]+) output tok\/s.*/\1/p'
    )"

    if [[ -z "$latency" ||
        -z "$requests_per_s" ||
        -z "$prompt_tokens_per_s" ||
        -z "$output_tokens_per_s" ]]; then
        echo "Failed to parse benchmark result from ${log_file}" >&2
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$backend" \
        "$num_prompts" \
        "$latency" \
        "$requests_per_s" \
        "$prompt_tokens_per_s" \
        "$output_tokens_per_s" \
        >> "$SUMMARY_FILE"
}

print_speedups() {
    echo
    echo "Output throughput comparison"
    printf '%-12s %-14s %-14s %-10s\n' \
        "num_prompts" "HF tok/s" "vLLM tok/s" "speedup"

    for num_prompts in "${CONCURRENCIES[@]}"; do
        local hf_throughput
        local vllm_throughput
        local speedup

        hf_throughput="$(
            awk -F '\t' \
                -v n="$num_prompts" \
                '$1 == "hf" && $2 == n { print $6 }' \
                "$SUMMARY_FILE"
        )"

        vllm_throughput="$(
            awk -F '\t' \
                -v n="$num_prompts" \
                '$1 == "vllm" && $2 == n { print $6 }' \
                "$SUMMARY_FILE"
        )"

        if [[ -n "$hf_throughput" && -n "$vllm_throughput" ]]; then
            speedup="$(
                awk \
                    -v hf="$hf_throughput" \
                    -v vllm="$vllm_throughput" \
                    'BEGIN { printf "%.2fx", vllm / hf }'
            )"

            printf '%-12s %-14s %-14s %-10s\n' \
                "$num_prompts" \
                "$hf_throughput" \
                "$vllm_throughput" \
                "$speedup"
        fi
    done
}

BACKEND="${1:-both}"

case "$BACKEND" in
    hf)
        for num_prompts in "${CONCURRENCIES[@]}"; do
            run_hf "$num_prompts"
        done
        ;;
    vllm)
        for num_prompts in "${CONCURRENCIES[@]}"; do
            run_vllm "$num_prompts"
        done
        ;;
    both)
        # 先跑 HF，再跑 vLLM，避免两个进程同时占用同一块 GPU。
        for num_prompts in "${CONCURRENCIES[@]}"; do
            run_hf "$num_prompts"
        done

        for num_prompts in "${CONCURRENCIES[@]}"; do
            run_vllm "$num_prompts"
        done

        print_speedups
        ;;
    *)
        echo "Usage: $0 [hf|vllm|both]" >&2
        exit 2
        ;;
esac

echo
echo "Benchmark complete."
echo "Summary: ${SUMMARY_FILE}"
echo "Full logs: ${RESULT_DIR}"