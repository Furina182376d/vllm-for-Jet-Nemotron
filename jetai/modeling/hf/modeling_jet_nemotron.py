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
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

from functools import partial
from typing import Callable, Optional, Tuple, Union, Iterable, Set

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.sequence import IntermediateTensors

from .configuration_jet_nemotron import JetNemotronConfig
from .utils import (
    make_layers,
    make_empty_intermediate_tensors_factory,
    PPMissingLayer,
)

logger = logging.get_logger(__name__)


class JetNemotronRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)



class JetNemotronMLP(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x



class JetNemotronAttention(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        tp = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v)
        out, _ = self.o_proj(attn_out)
        return out



class JetNemotronDecoderLayer(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()

        self.input_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = JetNemotronAttention(
            config,
            layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = JetNemotronMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, hidden_states



class JetNemotronModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda pfx: JetNemotronDecoderLayer(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=pfx,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

        if get_pp_group().is_last_rank:
            self.norm = JetNemotronRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        return self.norm(hidden_states)


class JetNemotronForCausalLM(nn.Module):
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
    supports_logprobs = True
    supports_prefix_caching = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # ===== Backbone =====
        self.model = JetNemotronModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        # ===== LM Head =====
        self.lm_head = ParallelLMHead(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )

    # 🔥 vLLM 用它判断：这是 generative / decoder-only 模型
    def get_model(self) -> nn.Module:
        return self.model

    # ===== vLLM Decoder Forward =====
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Args:
            input_ids: [num_tokens]
            positions: [num_tokens]
            intermediate_tensors: KV cache / attention 中间态
        """

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
        )

        # vLLM 允许在 prefill / decode 阶段返回 intermediate_tensors
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        logits = self.lm_head(hidden_states)
        return logits
    
    
from vllm.model_executor.models import ModelRegistry

ModelRegistry.register_model(
    "JetNemotronForCausalLM",
    JetNemotronForCausalLM,
)
