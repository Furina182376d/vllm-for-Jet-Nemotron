from jetai.vllm_plugin import register
from vllm import LLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
os.environ["VLLM_DISABLE_SPEC_DECODE"] = "1"

def main():
    register()  # 🔥 关键一步：注册模型类型
    
    llm = LLM(
        model="/home/tongjiayi/Jet-Nemotron/jetai/modeling/hf/",
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,  
        load_format="dummy",  
        gpu_memory_utilization=0.3,
        max_num_batched_tokens=1024, 
        speculative_config=None,
    )

    prompt = "Hello! Please introduce yourself."
    output = llm.generate([prompt])
    print(output[0].outputs[0].text)

if __name__ == '__main__':
    main()




# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name_or_path = "jet-ai/Jet-Nemotron-2B"

# For local testing, you can use the following path.
# NOTE: Be sure to download or soft-link the model weights to `jetai/modeling/hf`
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