import argparse
import os
import time

# Both benchmarks use the same physical GPU by default.
# You can override it before launching:
# CUDA_VISIBLE_DEVICES=0 python test-jet.py ...
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/home/tjy/codebases/Jet-Nemotron-HF/jetai/modeling/hf"

PROMPT = (
    "<|im_start|>user\n"
    "Introduce yourself in one concise sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Jet-Nemotron with Hugging Face Transformers."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=16,
        help="Number of independent requests in each measured workload.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "HF static batch size. Use 1 for a conventional serial "
            "model.generate service baseline."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    config._attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    encoded = tokenizer(PROMPT, return_tensors="pt")
    base_input_ids = encoded.input_ids.cuda()
    base_attention_mask = encoded.attention_mask.cuda()
    prompt_length = base_input_ids.shape[1]

    def generate_batch(batch_size):
        input_ids = base_input_ids.repeat(batch_size, 1)
        attention_mask = base_attention_mask.repeat(batch_size, 1)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            use_cache=True,
            # Disable early EOS termination so every request generates exactly
            # --max-tokens tokens, matching vLLM ignore_eos=True.
            eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        return outputs

    # Warm up the kernels at the measured HF batch size.
    warmup_batch_size = min(args.batch_size, args.num_prompts)
    with torch.inference_mode():
        for _ in range(max(args.warmup_runs, 0)):
            generate_batch(warmup_batch_size)
    torch.cuda.synchronize()

    latencies = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    last_outputs = None

    for run_index in range(max(args.runs, 1)):
        run_output_tokens = 0

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            for offset in range(0, args.num_prompts, args.batch_size):
                current_batch_size = min(
                    args.batch_size,
                    args.num_prompts - offset,
                )
                last_outputs = generate_batch(current_batch_size)
                run_output_tokens += (
                    last_outputs.shape[1] - prompt_length
                ) * current_batch_size

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        total_prompt_tokens += prompt_length * args.num_prompts
        total_output_tokens += run_output_tokens

        print(
            f"Run {run_index + 1}: {elapsed:.3f} s, "
            f"{args.num_prompts / elapsed:.2f} req/s, "
            f"{run_output_tokens / elapsed:.2f} output tok/s"
        )

    total_latency = sum(latencies)
    measured_runs = len(latencies)
    total_requests = args.num_prompts * measured_runs

    print()
    print(
        f"Backend: Hugging Face, static batch size={args.batch_size}\n"
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

    if last_outputs is not None:
        print()
        print(
            tokenizer.decode(
                last_outputs[0],
                skip_special_tokens=True,
            )
        )


if __name__ == "__main__":
    main()