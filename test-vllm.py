import os
import argparse
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
        default=0.7,
        help="Sampling temperature for the smoke test (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens (use 1 to inspect the first token).",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=20,
        help="Number of vLLM top log-probabilities to return (default: 20).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
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
    )
    
    prompt = "<|im_start|>user\nHello! Please introduce yourself.<|im_end|>\n<|im_start|>assistant\n"
    output = llm.generate(
        [prompt],
        sampling_params=SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
        ),
    )
    completion = output[0].outputs[0]
    print("Prompt Token IDs:", output[0].prompt_token_ids)
    print("Generated Token IDs:", completion.token_ids)
    print("Generated Text:", completion.text)
    if completion.logprobs:
        print("vLLM next-token logprobs:")
        for token_id, logprob in completion.logprobs[0].items():
            print(f"  id={token_id:6d} logprob={logprob.logprob: .6f} token={logprob.decoded_token!r}")
    print("Generated Text:", output[0].outputs[0].text)

if __name__ == "__main__":
    main()
