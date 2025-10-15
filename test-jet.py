import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name_or_path = "jet-ai/Jet-Nemotron-2B"

# For local testing, you can use the following path.
# NOTE: Be sure to download or soft-link the model weights to `jetai/modeling/hf`
model_name_or_path = "jetai/modeling/hf/"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                             trust_remote_code=True, 
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = model.eval().cuda()

input_str = "Hello, please introduce yourself."

input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_str)