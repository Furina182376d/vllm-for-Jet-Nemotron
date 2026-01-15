# jetai/vllm_plugin/registry.py
from vllm import ModelRegistry

def register():
    from jetai.modeling.hf.modeling_jet_nemotron import (
        JetNemotronForCausalLM,
    )

    ModelRegistry.register_model(
        "jet-nemotron",
        JetNemotronForCausalLM,
    )
