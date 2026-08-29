"""Run the native Transformers implementation against a local checkpoint.

The current working tree uses ``modeling_jet_nemotron.py`` for vLLM.  The
native HF implementation is therefore loaded from the pre-vLLM commit while
all non-Python files (including the local weights and tokenizer) come from
``--model``.
"""

from __future__ import annotations

import argparse
import importlib
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import torch


DEFAULT_MODEL = Path(__file__).resolve().parent / "jetai" / "modeling" / "hf"
DEFAULT_HF_SOURCE_COMMIT = "b14282f"
PROMPT = "<|im_start|>user\nHello! Please introduce yourself.<|im_end|>\n<|im_start|>assistant\n"
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Jet-Nemotron with the native Transformers implementation."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local checkpoint directory.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer directory; defaults to --model.")
    parser.add_argument("--prompt", default=PROMPT, help="Exact prompt to tokenize.")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--source-commit", default=DEFAULT_HF_SOURCE_COMMIT)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "flash_attention_2"))
    parser.add_argument("--device", default="cuda", help="Device for the loaded model (default: cuda).")
    parser.add_argument("--debug-hidden", action="store_true", help="Print per-layer hidden-state norms.")
    return parser.parse_args()


def make_reference_model_dir(
    model_dir: Path, tokenizer_dir: Path, source_commit: str, temp_root: Path
) -> Path:
    """Extract native HF Python files and overlay local checkpoint assets."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", source_commit, "jetai/modeling/hf"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        try:
            tar.extractall(temp_root, filter="data")
        except TypeError:  # Python < 3.12
            tar.extractall(temp_root)

    reference_dir = temp_root / "jetai" / "modeling" / "hf"
    assets = [
        asset
        for asset in model_dir.iterdir()
        if not ((asset.suffix == ".py" and asset.name != "configuration_jet_nemotron.py")
                or asset.name == "__pycache__")
    ]
    if tokenizer_dir != model_dir:
        assets.extend(asset for asset in tokenizer_dir.iterdir() if asset.name in TOKENIZER_FILES)
    for asset in assets:
        # Keep the historical native HF modules, but use this run's exact
        # config, tokenizer files, and safetensors checkpoint.
        target = reference_dir / asset.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(asset.resolve())
    return reference_dir


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer).expanduser().resolve() if args.tokenizer else model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"--model must be a local directory: {model_dir}")
    if not tokenizer_dir.is_dir():
        raise SystemExit(f"--tokenizer must be a local directory: {tokenizer_dir}")

    with tempfile.TemporaryDirectory(prefix="jet-nemotron-hf-") as temp_name:
        # Keep Transformers' dynamic-module cache writable and isolated from
        # any cache produced by the vLLM run.
        os.environ["HF_MODULES_CACHE"] = str(Path(temp_name) / "hf-modules")
        reference_dir = make_reference_model_dir(
            model_dir, tokenizer_dir, args.source_commit, Path(temp_name)
        )

        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        tokenizer = AutoTokenizer.from_pretrained(reference_dir, trust_remote_code=True)
        config = AutoConfig.from_pretrained(reference_dir, trust_remote_code=True)
        # This checkpoint stores padding in tokenizer_config.json rather than
        # config.json, while the native HF model uses config.pad_token_id.
        config.pad_token_id = tokenizer.pad_token_id
        # The compatibility config normalizes these names for Transformers;
        # the historical native model dispatches on the original names.
        config.layer_types = [
            {"linear_attention": "jet", "full_attention": "attn", "sliding_attention": "swa"}.get(
                layer_type, layer_type
            )
            for layer_type in config.layer_types
        ]
        # Transformers 5 renamed/removed the old ``default`` RoPE registry
        # entry used by the native Jet-Nemotron source.
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:
            def default_rope(config, device=None, seq_len=None):
                head_dim = config.hidden_size // config.num_attention_heads
                positions = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                inv_freq = 1.0 / (config.rope_theta ** (positions / head_dim))
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS["default"] = default_rope
        model_class = get_class_from_dynamic_module(
            "modeling_jet_nemotron.JetNemotronForCausalLM",
            reference_dir,
            trust_remote_code=True,
        )
        # Transformers 5 represents this metadata as a dict; the historical
        # Jet-Nemotron source used the older list form.
        model_class._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        model = model_class.from_pretrained(
            reference_dir,
            trust_remote_code=True,
            config=config,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
        ).to(args.device).eval()
        # Transformers' generation loop probes ``cache.layers`` when this
        # flag is true; JetNemotronCache intentionally stores state differently.
        model_module = importlib.import_module(model_class.__module__)
        cache_class = getattr(model_module, "JetNemotronCache", None)
        if cache_class is not None:
            cache_class.is_compileable = False

        # Capture the first layer's submodule outputs so they can be compared
        # with the corresponding vLLM debug values.  The native HF attention
        # returns a tuple, whereas the MLP returns a tensor.
        attn_capture = {}
        mlp_capture = {}
        block_capture = {}

        def capture_attn(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            attn_capture["value"] = value.detach()

        def capture_mlp(_module, _args, output):
            mlp_capture["value"] = output.detach()

        def capture_block(name):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                block_capture[name] = value.detach()
                if name == "q_proj":
                    block_capture["q_proj_input_norm"] = _args[0].float().norm().item()
            return hook

        layer0_attn = model.model.layers[0].self_attn
        layer0_attn.register_forward_hook(capture_attn)
        model.model.layers[0].mlp.register_forward_hook(capture_mlp)
        for name in ("q_proj", "k_proj", "v_proj", "dynamic_conv1d", "o_proj"):
            module = getattr(layer0_attn, name, None)
            if module is not None:
                module.register_forward_hook(capture_block(name))

        inputs = tokenizer(args.prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {name: value.to(model_device) for name, value in inputs.items()}
        input_ids = inputs["input_ids"]

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, output_hidden_states=args.debug_hidden)
            if args.debug_hidden and outputs.hidden_states is not None:
                for index, state in enumerate(outputs.hidden_states[1:]):
                    print(f"HF_DEBUG layer={index} total_norm={state.float().norm().item():.6f}")
                print(f"HF_DEBUG input_norm={outputs.hidden_states[0].float().norm().item():.6f}")
                print(f"HF_DEBUG block0_input_norm={block_capture.get('q_proj_input_norm', float('nan')):.6f}")
                print(f"HF_DEBUG block0_q_weight_norm={layer0_attn.q_proj.weight.float().norm().item():.6f}")
                print(f"HF_DEBUG block0_v_weight_norm={layer0_attn.v_proj.weight.float().norm().item():.6f}")
                if "value" in attn_capture:
                    print(f"HF_DEBUG layer0_attn_norm={attn_capture['value'].float().norm().item():.6f}")
                if "value" in mlp_capture:
                    print(f"HF_DEBUG layer0_mlp_norm={mlp_capture['value'].float().norm().item():.6f}")
                for name in ("q_proj", "k_proj", "v_proj", "dynamic_conv1d", "o_proj"):
                    if name in block_capture:
                        print(f"HF_DEBUG block0_{name}_norm={block_capture[name].float().norm().item():.6f}")
            next_logits = outputs.logits[0, -1].float()
            values, token_ids = torch.topk(next_logits, k=min(args.top_k, next_logits.numel()))
            generation_config = model.generation_config
            # Older Jet-Nemotron generation code expects this flag, while
            # some Transformers releases do not define it on GenerationConfig.
            if not hasattr(generation_config, "return_legacy_cache"):
                generation_config.return_legacy_cache = False
            generation_config.max_new_tokens = args.max_new_tokens
            generation_config.do_sample = False
            generation_config.use_cache = True
            generated = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        completion_ids = generated[0, input_ids.shape[1] :].tolist()
        print("Prompt Token IDs:", input_ids[0].tolist())
        print("HF greedy token IDs:", completion_ids)
        print("HF greedy text:", tokenizer.decode(completion_ids, skip_special_tokens=False))
        print("HF next-token argmax ID:", token_ids[0].item())
        print("HF next-token top-k:")
        for token_id, value in zip(token_ids.tolist(), values.tolist()):
            token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            print(f"  id={token_id:6d} logit={value: .6f} token={token!r}")


if __name__ == "__main__":
    main()
