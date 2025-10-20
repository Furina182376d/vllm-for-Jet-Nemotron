from safetensors import safe_open

checkpoint_path = "/home/tongjiayi/Jet-Nemotron/jetai/modeling/hf/model.safetensors"

with safe_open(checkpoint_path, framework="pt") as f:
    keys = f.keys()
    print(keys)  # 打印所有权重名字
    if "model.model.lm_head.weight" in keys:
        print("LM head 权重存在 ✅")
    else:
        print("LM head 权重缺失 ❌")