# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from itertools import islice
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

try:
    # vLLM versions before v1 exposed these under ``vllm.attention``.
    from vllm.attention.backends.abstract import AttentionType
    from vllm.attention.layer import Attention
except ImportError:
    # vLLM 0.8+ moved the implementation into model_executor/v1.
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
# from vllm.transformers_utils.config import set_default_rope_theta

from .configuration_jet_nemotron import JetNemotronConfig
from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
import inspect
import os

_SUPPORTS_SHAPE_INVARIANTS = (
    "shape_invariants" in inspect.signature(support_torch_compile).parameters
)

class JetNemotronMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Keep the HF parameter layout (separate gate/up matrices).  The
        # checkpoint is not stored in packed gate/up order, and using a
        # packed tensor-parallel layer here can silently alter that mapping
        # even for a single-rank engine.
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class JetNemotronRMSNorm(nn.Module):
    """HF-compatible RMSNorm with the residual-carrying vLLM interface."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return self._norm(hidden_states)
        residual = residual + hidden_states
        return self._norm(residual), residual


class JetNemotronAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        sliding_window: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
            
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = sliding_window
        self.num_key_value_groups = num_heads // num_kv_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Older Transformers configs expose ``rope_theta``/``rope_scaling``
        # directly, while newer ones consolidate them in ``rope_parameters``.
        # Resolve both forms explicitly so vLLM never silently falls back to
        # the default theta (10000).
        rope_parameters = dict(rope_parameters or {})
        base = rope_parameters.get(
            "rope_theta",
            rope_parameters.get(
                "theta",
                10000.0,
            ),
        )
        rope_scaling = rope_parameters.get("rope_scaling", None)

        try:
            # Legacy vLLM accepted ``rotary_dim``, ``base`` and ``rope_scaling``.
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position,
                base=base,
                rope_scaling=rope_scaling,
            )
        except TypeError:
            # vLLM 0.8+ takes the consolidated rope parameter dictionary.
            rope_parameters = dict(rope_parameters or {})
            rope_parameters.setdefault("rope_theta", base)
            rope_parameters.setdefault("rope_type", "default")
            if rope_scaling:
                rope_parameters.update(rope_scaling)
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position,
                rope_parameters=rope_parameters,
            )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
        )
        self._attn_accepts_output_shape = (
            "output_shape" in inspect.signature(self.attn.forward).parameters
        )

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache: torch.Tensor | None = None,
        **kwargs,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # vLLM v1 keeps the KV cache in the attention module's forward
        # context; the legacy fourth positional argument is ``output_shape``
        # in this API and must not receive a model cache object.
        # vLLM 0.27+ obtains the KV cache from forward context and accepts an
        # optional output shape. Older releases used the fourth positional
        # argument for a cache object, so keep a signature-based fallback.
        if self._attn_accepts_output_shape:
            attn_output = self.attn(q, k, v, output_shape=hidden_states.shape)
        else:
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output, None, cache


class DynamicShortConvolutionVLLM(nn.Module):
    """vLLM适配的动态卷积层"""
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        activation: str = "silu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        try:
            from .dynamic_conv import DynamicShortConvolution
            from .dconv_fwdbwd import dynamic_conv_triton_autograd
            from .dconv_fwd_cache import dynamic_conv_triton_cache
            from .dconv_step import causal_conv_step_triton
            
            # 使用原始的DynamicShortConvolution
            self.conv = DynamicShortConvolution(
                hidden_size,
                kernel_size,
            )
            
            # 保存函数引用以便在forward中使用
            self.dynamic_conv_triton_autograd = dynamic_conv_triton_autograd
            self.dynamic_conv_triton_cache = dynamic_conv_triton_cache
            self.causal_conv_step_triton = causal_conv_step_triton
            
        except ImportError:
            raise ImportError(
                "Dynamic convolution is not available. Please install the required dependencies to use this feature."
            )
        
        # 激活函数
        if activation == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # 输出投影层
        self.output_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_proj",
        )

    def forward(
        self,
        *,
        positions: torch.Tensor | None = None,
        hidden_states: torch.Tensor,
        cache: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        vLLM-compatible forward:
        returns (output, attn_metadata, new_cache)
        """
        x, new_cache = self.conv(
            hidden_states,
            cache=cache,
            output_final_state=False,
        )

        x = self.act_fn(x)
        x, _ = self.output_proj(x)

        return x, None, new_cache


class JetNemotronDecoderLayer(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') else "attn"
        
        # 初始化注意力或动态卷积层
        if self.layer_type in ("attn", "full_attention"):
            self.self_attn = JetNemotronAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=config.max_position_embeddings,
                rope_parameters=(
                    getattr(config, "rope_parameters", None)
                    or {"rope_theta": getattr(config, "rope_theta", 10000.0)}
                ),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                sliding_window=None,
                layer_idx=layer_idx,
            )
            self.dynamic_conv = None

        elif self.layer_type in ("swa", "sliding_attention"):
            sliding_window = None
            if hasattr(config, "efficient_attention_config") and config.efficient_attention_config:
                swa_cfg = config.efficient_attention_config.get("swa", {})
                sliding_window = swa_cfg.get("window_size")

            self.self_attn = JetNemotronAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=config.max_position_embeddings,
                rope_parameters=(
                    getattr(config, "rope_parameters", None)
                    or {"rope_theta": getattr(config, "rope_theta", 10000.0)}
                ),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                sliding_window=sliding_window,
                layer_idx=layer_idx,
            )
            self.dynamic_conv = None
        elif self.layer_type in ("jet", "linear_attention"):
            from .jet_block import JetBlock
            self.self_attn = JetBlock(
                config=config,
                # ``linear_attention`` is the Transformers-standard alias;
                # JetBlock still selects its settings from the historical
                # ``efficient_attention_config["jet"]`` entry.
                layer_type="jet",
                layer_idx=layer_idx,
                prefix=f"{prefix}.self_attn",
            )
            self.dynamic_conv = None
        else:
            # 动态卷积层
            self.self_attn = None
            kernel_size = 4
            if hasattr(config, 'efficient_attention_config') and config.efficient_attention_config:
                dconv_config = config.efficient_attention_config.get("dynamic_conv", {})
                kernel_size = dconv_config.get("kernel_size", 4)
            
            self.dynamic_conv = DynamicShortConvolutionVLLM(
                hidden_size=config.hidden_size,
                kernel_size=kernel_size,
                activation=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.dynamic_conv",
            )
        
        # MLP层
        self.mlp = JetNemotronMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # LayerNorm层
        self.input_layernorm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        
        # 用于动态卷积的额外LayerNorm
        self.post_conv_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ) if self.layer_type not in [
            "attn", "full_attention", "swa", "sliding_attention", "jet", "linear_attention"
        ] else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        debug_layer = os.environ.get("JET_DEBUG_TRACE") and hidden_states.shape[0] > 2
        if debug_layer:
            print(
                f"JET_DEBUG layer={self.layer_idx} norm_weight_norm="
                f"{self.input_layernorm.weight.float().norm().item():.6f} "
                f"normed_input_norm={hidden_states.float().norm().item():.6f} "
                f"shape={tuple(hidden_states.shape)} "
                f"row_norms={hidden_states.float().reshape(-1, hidden_states.shape[-1]).norm(dim=-1)[:16].tolist()}",
                flush=True,
            )

        if self.self_attn is not None:
            out = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

            if isinstance(out, tuple):
                hidden_states, _, _ = out
            else:
                hidden_states = out

            if debug_layer:
                print(
                    f"JET_DEBUG layer={self.layer_idx} attn_norm={hidden_states.float().norm().item():.6f}",
                    flush=True,
                )

            # MLP部分
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        else:
            # 动态卷积层
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
            # 应用动态卷积
            hidden_states, _, _ = self.dynamic_conv(
                positions=positions,
                hidden_states=hidden_states,
            )
            
            # MLP部分
            hidden_states, residual = self.post_conv_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)

        if os.environ.get("JET_DEBUG_TRACE") and hidden_states.shape[0] > 2:
            print(
                f"JET_DEBUG layer={self.layer_idx} mlp_norm={hidden_states.float().norm().item():.6f} "
                f"total_norm={(hidden_states + residual).float().norm().item():.6f}",
                flush=True,
            )

        return hidden_states, residual, None


def jet_nemotron_model_invariants(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    """Shape invariants for JetNemotronModel, translated to runtime assertions"""
    torch._assert(input_ids.size()[0] == positions.size()[-1])
    
    if intermediate_tensors is not None:
        torch._assert(
            input_ids.size()[0] == intermediate_tensors["hidden_states"].size()[0]
        )

    if inputs_embeds is not None:
        torch._assert(input_ids.size()[0] == inputs_embeds.size()[0])

    if inputs_embeds is not None and intermediate_tensors is not None:
        torch._assert(
            inputs_embeds.size()[1] == intermediate_tensors["hidden_states"].size()[1]
        )


if _SUPPORTS_SHAPE_INVARIANTS:
    JetNemotronModelBase = support_torch_compile(
        dynamic_arg_dims={
            "input_ids": 0,
            "positions": -1,
            "intermediate_tensors": 0,
            "inputs_embeds": 0,
        },
        shape_invariants=jet_nemotron_model_invariants,
    )
else:
    JetNemotronModelBase = support_torch_compile(
        dynamic_arg_dims={
            "input_ids": 0,
            "positions": -1,
            "intermediate_tensors": 0,
            "inputs_embeds": 0,
        }
    )


@JetNemotronModelBase
class JetNemotronModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def layer_fn(prefix: str):
            layer_idx = extract_layer_index(prefix)
            return JetNemotronDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                layer_idx=layer_idx,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            layer_fn,
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        
        if get_pp_group().is_last_rank:
            self.norm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if os.environ.get("JET_DEBUG_TRACE") and input_ids is not None and input_ids.numel() <= 32:
                embed_info = (
                    "none"
                    if inputs_embeds is None
                    else f"shape={tuple(inputs_embeds.shape)} norm={inputs_embeds.float().norm().item():.6f}"
                )
                print(
                    f"JET_DEBUG input_ids={input_ids.reshape(-1).tolist()} inputs_embeds={embed_info}",
                    flush=True,
                )
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        debug_layers = os.environ.get("JET_DEBUG_TRACE") and hidden_states.shape[0] > 2
        if debug_layers:
            print(
                f"JET_DEBUG embedding_norm={hidden_states.float().norm().item():.6f}",
                flush=True,
            )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
            hidden_states, residual, _ = layer(positions, hidden_states, residual)
            if debug_layers:
                total = hidden_states + residual if residual is not None else hidden_states
                print(
                    f"JET_DEBUG layer={self.start_layer + idx} "
                    f"total_norm={total.float().norm().item():.6f}",
                    flush=True,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        if debug_layers:
            print(f"JET_DEBUG final_norm={hidden_states.float().norm().item():.6f}", flush=True)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            
            if not name.startswith("model.") and f"model.{name}" in params_dict:
                name = f"model.{name}"

            # 处理量化缩放因子
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                if scale_name not in params_dict:
                    continue
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # 动态卷积层的权重加载
            if "dynamic_conv" in name:
                # 动态卷积层可能有不同的参数名
                param_name = name.replace("conv.", "dynamic_conv.conv.")
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                continue

            # ===== Jet block =====
            if "jet_block" in name or "jet." in name:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue

            # Plain HF-layout parameters (including the MLP and JetBlock)
            # should be loaded verbatim before considering packed mappings.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            # ===== qkv / gate_up 堆叠参数 =====
            stacked_loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                merged_name = name.replace(weight_name, param_name)

                if merged_name.endswith(".bias") and merged_name not in params_dict:
                    continue
                if is_pp_missing_parameter(merged_name, self):
                    continue
                if merged_name.endswith("scale"):
                    merged_name = maybe_remap_kv_scale_name(merged_name, params_dict)
                    if merged_name is None:
                        continue
                if merged_name not in params_dict:
                    continue

                param = params_dict[merged_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)

                loaded_params.add(merged_name)
                stacked_loaded = True
                break

            if stacked_loaded:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        print("loaded params:", len(loaded_params))
        print("total params:", len(params_dict))
        return loaded_params


class JetNemotronForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        
        self.model = JetNemotronModel(
            vllm_config=vllm_config, 
            prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."] if hasattr(self.config, 'tie_word_embeddings') and self.config.tie_word_embeddings 
                else None
            ),
        )
        return loader.load_weights(weights)
