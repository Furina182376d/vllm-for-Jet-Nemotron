import sys
sys.path.append("/home/tongjiayi/Jet-Nemotron/jetai/modeling")

from hf.modeling_jet_nemotron import JetNemotronForCausalLM
from hf.configuration_jet_nemotron import JetNemotronConfig
from vllm import LLM, SamplingParams

def main():
    model_name_or_path = "/home/tongjiayi/Jet-Nemotron/jetai/modeling/hf/"

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
        top_p=1.0
    )

    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )

    input_str = "Hello, which high school did you go to?"
    outputs = llm.generate([input_str], sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated text: {output.outputs[0].text}")

if __name__ == "__main__":
    main()
