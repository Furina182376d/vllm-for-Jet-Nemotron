import argparse
import os
import time
from pathlib import Path

# Match test-jet.py. The environment can still override this.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


PROMPT = (
    "<|im_start|>user\n"
    "Introduce yourself in one concise sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def has_model_weights(model_path: Path) -> bool:
    weight_patterns = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
    )
    return any(
        path.is_file()
        for pattern in weight_patterns
        for path in model_path.glob(pattern)
    )


def parse_args():
    default_model_path = (
        Path(__file__).resolve().parent / "jetai" / "modeling" / "hf"
    )

    parser = argparse.ArgumentParser(
        description="Benchmark Jet-Nemotron with vLLM."
    )
    parser.add_argument("--model", default=str(default_model_path))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable torch.compile/CUDA Graph capture. Leave disabled if "
            "JetBlock dynamic convolution is not graph-safe."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    if args.debug:
        from jetai.utils.debug import set_debug

        set_debug()

    model_path = args.model
    tokenizer_path = args.tokenizer or model_path

    # Keep the V1 model runner for the current custom model implementation.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    local_model_path = Path(model_path).expanduser()

    if local_model_path.is_file():
        raise SystemExit(
            f"--model must be a model directory, not a weight file: "
            f"{local_model_path}"
        )

    if local_model_path.is_dir() and not has_model_weights(local_model_path):
        raise SystemExit(
            f"No model weights found in {local_model_path}."
        )

    # Register JetNemotronForCausalLM.
    import jetai.vllm_plugin  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.compile,
        # Required by JetBlock's recurrent/convolution state handling.
        async_scheduling=False,
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # Generate an identical token count for every request. Do not stop at
        # EOS or chat-role boundaries during a throughput benchmark.
        ignore_eos=True,
    )

    prompts = [PROMPT] * args.num_prompts

    # A small concurrent warmup is sufficient to load kernels without making
    # startup excessively long.
    warmup_prompts = prompts
    for _ in range(max(args.warmup_runs, 0)):
        llm.generate(
            warmup_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    latencies = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    last_results = None

    for run_index in range(max(args.runs, 1)):
        start = time.perf_counter()
        results = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start

        run_prompt_tokens = sum(
            len(result.prompt_token_ids)
            for result in results
        )
        run_output_tokens = sum(
            len(result.outputs[0].token_ids)
            for result in results
        )

        latencies.append(elapsed)
        total_prompt_tokens += run_prompt_tokens
        total_output_tokens += run_output_tokens
        last_results = results

        print(
            f"Run {run_index + 1}: {elapsed:.3f} s, "
            f"{len(results) / elapsed:.2f} req/s, "
            f"{run_output_tokens / elapsed:.2f} output tok/s"
        )

    total_latency = sum(latencies)
    measured_runs = len(latencies)
    total_requests = args.num_prompts * measured_runs
    prompt_length = len(last_results[0].prompt_token_ids)

    print()
    print(
        f"Backend: vLLM continuous batching\n"
        f"Workload: {args.num_prompts} requests/run, "
        f"{prompt_length} prompt tokens/request, "
        f"{args.max_tokens} output tokens/request\n"
        f"Average workload latency: "
        f"{total_latency / measured_runs:.3f} s\n"
        f"Aggregate throughput: "
        f"{total_requests / total_latency:.2f} req/s, "
        f"{total_prompt_tokens / total_latency:.2f} prompt tok/s, "
        f"{total_output_tokens / total_latency:.2f} output tok/s"
    )

    if last_results:
        print()
        print(last_results[0].outputs[0].text)


if __name__ == "__main__":
    main()