# jetai/vllm_plugin/registry.py
from vllm import ModelRegistry
from jetai.modeling.hf.modeling_jet_nemotron import JetNemotronForCausalLM

def register():
    ModelRegistry.register_model(
        "NemotronForCausalLM",
        JetNemotronForCausalLM,
    )