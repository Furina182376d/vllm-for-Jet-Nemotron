import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 直接导入插件（会触发注册）
import jetai.vllm_plugin

# 再导入 vLLM
from vllm import LLM

def main():
    model_path = "/home/tongjiayi/Jet-Nemotron/jetai/modeling/hf"
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tokenizer=model_path,
        gpu_memory_utilization=0.35,
    )
    
    prompt = "<|im_start|>user\nHello! Please introduce yourself.<|im_end|>\n<|im_start|>assistant\n"
    output = llm.generate([prompt])
    print("Generated Text:", output[0].outputs[0].text)

if __name__ == "__main__":
    main()