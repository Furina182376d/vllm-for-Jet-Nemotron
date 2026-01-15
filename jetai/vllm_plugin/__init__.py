# def register():
#     from vllm import ModelRegistry
#     from jetai.modeling.hf.modeling_jet_nemotron import JetNemotronForCausalLM

#     # 将 model_type 和类名绑定
#     ModelRegistry.register_model("jet_nemotron", JetNemotronForCausalLM)
from .registry import *
register()