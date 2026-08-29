import time

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "/home/tjy/codebases/Jet-Nemotron-HF/jetai/modeling/hf"

# For local testing, you can use the following path.
# NOTE: Be sure to download or soft-link the model weights to `jetai/modeling/hf`
# model_name_or_path = "jetai/modeling/hf/"

# The downloaded config defaults to FlashAttention 2. Override it explicitly
# because this environment intentionally does not install flash-attn.
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
config._attn_implementation = "sdpa"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             config=config,
                                             trust_remote_code=True, 
                                             attn_implementation="sdpa",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = model.eval().cuda()

# Keep this prompt and generation setup identical to ../jet-nemotron/test-vllm.py.
input_str = (
    "<|im_start|>user\n"
    "Introduce yourself in one concise sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = tokenizer(input_str, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
attention_mask = inputs.attention_mask.cuda()

generation_kwargs = {
    "max_new_tokens": 32,
    "do_sample": False,
    "attention_mask": attention_mask,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": [tokenizer.eos_token_id, 151645],
}

# The first CUDA call includes kernel loading and initialization overhead.
with torch.inference_mode():
    model.generate(input_ids, **generation_kwargs)
torch.cuda.synchronize()

latencies = []
output_token_counts = []
outputs = None
num_runs = 5
for _ in range(num_runs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(input_ids, **generation_kwargs)
    torch.cuda.synchronize()
    latencies.append(time.perf_counter() - start)
    output_token_counts.append(outputs.shape[1] - input_ids.shape[1])

prompt_tokens = input_ids.shape[1]
total_latency = sum(latencies)
average_latency = total_latency / len(latencies)
prompt_tokens_per_second = prompt_tokens * num_runs / total_latency
output_tokens_per_second = sum(output_token_counts) / total_latency
print(
    f"Average generation latency: {average_latency:.3f} s ({num_runs} run(s))\n"
    f"Throughput: {prompt_tokens_per_second:.2f} prompt tok/s, "
    f"{output_tokens_per_second:.2f} output tok/s"
)

output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_str)
