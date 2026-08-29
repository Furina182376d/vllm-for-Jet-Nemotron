import os
import argparse
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def _has_model_weights(model_path: Path) -> bool:
    """Return whether a local Hugging Face directory contains loadable weights."""
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


def _parse_args() -> argparse.Namespace:
    default_model_path = Path(__file__).resolve().parent / "jetai" / "modeling" / "hf"
    parser = argparse.ArgumentParser(description="Run a short Jet-Nemotron vLLM generation.")
    parser.add_argument(
        "--model",
        default=str(default_model_path),
        help="Local model directory or a Hugging Face model ID (default: repository HF directory).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory/ID. Defaults to --model.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="Fraction of GPU memory available to vLLM (default: 0.35).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable vLLM torch.compile/CUDA Graph capture. Disabled by default "
            "because JetBlock dynamic convolution is not graph-safe in vLLM 0.27.1."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the smoke test (default: greedy decoding).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens (default: 32).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Untimed warmup generations before measurement (default: 1).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed generations to average (default: 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Jet model diagnostics (equivalent to JET_DEBUG=1).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.debug:
        from jetai.utils.debug import set_debug

        set_debug()
    model_path = args.model
    tokenizer_path = args.tokenizer or model_path

    # JetNemotron currently uses the V1 runner's input/state path.  The V2
    # runner's Triton/UVA prefill buffer can leave custom-model prompt IDs at
    # their zero-initialized values (token 0), which changes the model input.
    # Keep an explicit environment override for users testing V2 support.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    # vLLM otherwise starts an EngineCore process before reporting this common
    # setup error, which obscures the actual fix in a long traceback.
    local_model_path = Path(model_path).expanduser()
    if local_model_path.is_file():
        raise SystemExit(
            f"--model must be a model directory, not a weight file: {local_model_path}\n"
            f"Use --model {local_model_path.parent} instead."
        )
    if local_model_path.is_dir() and not _has_model_weights(local_model_path):
        raise SystemExit(
            f"No model weights found in {local_model_path}.\n"
            "Download them first, for example:\n"
            f"  hf download jet-ai/Jet-Nemotron-2B --local-dir {local_model_path} "
            '--include "*safetensors*" --include "config.json"\n'
            "Or pass --model with a directory containing *.safetensors/*.bin weights."
        )

    # Importing the plugin registers the custom architecture with vLLM.
    import jetai.vllm_plugin
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tokenizer=tokenizer_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.compile,
        # JetBlock keeps per-layer recurrent/ convolution state between decode
        # steps; synchronous scheduling preserves that state ordering.
        async_scheduling=False,
    )
    
    prompt = (
        "<|im_start|>user\n"
        "Introduce yourself in one concise sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # Stop at chat-role boundaries as well as the native turn marker.
        # Some checkpoints emit the next role as plain text without first
        # producing <|im_end|>.
        stop_token_ids=[151645],
        stop=["\nuser", "\nassistant", "<|im_end|>", "<|im_start|>"],
    )

    for _ in range(max(args.warmup_runs, 0)):
        llm.generate([prompt], sampling_params=sampling_params)

    timings = []
    outputs = []
    for _ in range(max(args.runs, 1)):
        start = time.perf_counter()
        result = llm.generate([prompt], sampling_params=sampling_params)[0]
        elapsed = time.perf_counter() - start
        completion = result.outputs[0]
        timings.append((elapsed, len(result.prompt_token_ids), len(completion.token_ids)))
        outputs.append(completion.text)

    elapsed = sum(item[0] for item in timings)
    prompt_tokens = sum(item[1] for item in timings)
    output_tokens = sum(item[2] for item in timings)
    print(outputs[-1])
    print(
        f"Average generation latency: {elapsed / len(timings):.3f} s "
        f"({len(timings)} run(s))"
    )
    print(
        f"Throughput: {prompt_tokens / elapsed:.2f} prompt tok/s, "
        f"{output_tokens / elapsed:.2f} output tok/s"
    )

if __name__ == "__main__":
    main()
