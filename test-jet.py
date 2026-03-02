import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 直接导入插件（会触发注册）
import jetai.vllm_plugin

# 再导入 vLLM
from vllm import LLM

def main():
    llm = LLM(
        model="/home/tongjiayi/Jet-Nemotron/jetai/modeling/hf",
        trust_remote_code=True,
        gpu_memory_utilization=0.65,
    )
    
    prompt = "Hello! Please introduce yourself."
    output = llm.generate([prompt])
    print(output[0].outputs[0].text)

if __name__ == "__main__":
    main()
    
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name_or_path = "jet-ai/Jet-Nemotron-2B"

# # For local testing, you can use the following path.
# # NOTE: Be sure to download or soft-link the model weights to `jetai/modeling/hf`
# model_name_or_path = "jetai/modeling/hf/"

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
#                                              trust_remote_code=True, 
#                                              attn_implementation="sdpa",
#                                              torch_dtype=torch.bfloat16,
#                                              device_map="cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# model = model.eval().cuda()

# input_str = "Hello, please introduce yourself."

# input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()
# output = model.generate(input_ids, max_new_tokens=500, do_sample=False)
# output_str = tokenizer.decode(output[0], skip_special_tokens=True)
# print(output_str)