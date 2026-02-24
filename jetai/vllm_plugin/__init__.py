# jetai/vllm_plugin/__init__.py
import atexit
from vllm.model_executor.models import ModelRegistry
from jetai.modeling.hf.modeling_jet_nemotron import JetNemotronForCausalLM

def register_model():
    """注册自定义模型到 vLLM"""
    try:
        ModelRegistry.register_model(
            "JetNemotronForCausalLM",
            JetNemotronForCausalLM,
        )
        print("Successfully registered JetNemotronForCausalLM")
    except Exception as e:
        print(f"Failed to register model: {e}")

# 在导入时自动注册
register_model()

# 确保在程序退出前仍然保持注册
atexit.register(register_model)