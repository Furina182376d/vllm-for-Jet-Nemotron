def register():
    from vllm.model_executor.models import ModelRegistry
    from jetai.modeling.hf.modeling_jet_nemotron import JetNemotronForCausalLM


    ModelRegistry.register_model(
        "JetNemotronForCausalLM",
        JetNemotronForCausalLM,
    )