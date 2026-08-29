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

# This file is modified from https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                      fused_recurrent_gated_delta_rule)


from .dynamic_conv import DynamicShortConvolution
from .configuration_jet_nemotron import JetNemotronConfig
from .kv_cache import JetNemotronCache


@dataclass
class JetBlockConfig():
    mode: str = 'chunk'
    expand_v: int = 2.0
    num_heads: int = 6
    head_dim: int = 256
    norm_eps: float = 1e-5
    conv_size: int = 4
    dconv_generator_reduction: int = 8
    dconv_implementation: str = 'triton'


def init_linear_conv1d(weight: torch.Tensor, std: float, bias: Optional[torch.Tensor] = None) -> None:
    weight.data.normal_(mean=0.0, std=std)
    if bias is not None:
        if not getattr(bias, "_no_reinit", False):
            nn.init.zeros_(bias)


class JetBlock(nn.Module):
    def __init__(
        self,
        config: Optional[JetNemotronConfig] = None,
        layer_type: str = 'jet',
        layer_idx: Optional[int] = None,
        hidden_size: Optional[int] = None,
        initializer_range: Optional[float] = None,
        jet_block_config: Optional[JetBlockConfig] = None,
        prefix: str = "",
    ) -> JetBlock:
        super().__init__()

        if jet_block_config is None:
            assert config.efficient_attention_config is not None, "Efficient attention config must be provided in JetConfig."
            assert layer_type in config.efficient_attention_config, \
                f"{layer_type} configuration must be provided in efficient_attention_config."
            jet_block_config = JetBlockConfig(**config.efficient_attention_config[layer_type])

        hidden_size = hidden_size or config.hidden_size
        initializer_range = initializer_range or config.initializer_range

        self.mode = jet_block_config.mode

        self.hidden_size = hidden_size
        self.expand_v = jet_block_config.expand_v

        self.conv_size = jet_block_config.conv_size

        self.head_dim = jet_block_config.head_dim
        self.num_heads = jet_block_config.num_heads

        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = jet_block_config.head_dim
        self.head_v_dim = int(jet_block_config.head_dim * self.expand_v)
        self.layer_idx = layer_idx

        # vLLM's generic model interface does not pass a HF-style cache to
        # custom attention-free layers. Keep the latest state locally so
        # single-token decode steps remain autoregressive. A new prefill
        # (q_len > 1) resets these values below.
        self._cached_conv_state: Optional[torch.Tensor] = None
        self._cached_recurrent_state: Optional[torch.Tensor] = None
        self._cached_position: Optional[int] = None

        self.autotune_interval = 32 * 16 * 1024 # 32 batch size * 16 num head * 1024 sequence length

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.key_dim * self.expand_v}, which is invalid for nn.Linear."
            )
        if not math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}. "
                f"Resulting head_v_dim would be {self.head_dim * self.expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert self.mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{jet_block_config.mode}`."

        # Keep these projections as regular Linear layers.  Jet checkpoints
        # store each matrix independently in Hugging Face layout; using
        # vLLM's tensor-parallel wrappers here can apply an extra shard
        # transform during loading even when TP=1.
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=self.value_dim,
            kernel_size=self.conv_size,
            generator_input_size=self.hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
            static_conv_init=lambda x: init_linear_conv1d(x, std=initializer_range),
            implementation=jet_block_config.dconv_implementation,
        )

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=float(jet_block_config.norm_eps), autotune_interval=self.autotune_interval)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        *,
        positions: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
        cache: Optional[JetNemotronCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    )-> Tuple[torch.Tensor, None, Optional[JetNemotronCache]]:
        unsqueezed = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            unsqueezed = True

        torch._assert(
            hidden_states.dim() == 3,
            f"hidden_states must be [B, T, D], got shape {hidden_states.shape}"
        )
        if attention_mask is not None:
            if len(attention_mask.shape) > 2:
                attention_mask = attention_mask.squeeze(1)
                attention_mask = torch.where(attention_mask[:, -1] > -1, 1, 0)

            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        position_end: Optional[int] = None
        if positions is not None and positions.numel() > 0:
            # vLLM supplies absolute positions for flattened prompt/decode
            # tokens.  Keep a Python scalar so stale state is never reused
            # for a new request that happens to use the same module instance.
            position_end = int(positions.reshape(-1)[-1].item())
        # Match the Hugging Face implementation: short prompts and decode
        # steps use the recurrent kernel, while long prefills use chunking.
        # This distinction is numerically important for the checkpoint.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if cache is not None and len(cache) > self.layer_idx:
            last_state = cache[self.layer_idx]
        elif (
            q_len == 1
            and self._cached_recurrent_state is not None
            and (
                position_end is None
                or self._cached_position is None
                or position_end == self._cached_position + 1
            )
        ):
            last_state = {
                "conv_state": self._cached_conv_state,
                "recurrent_state": self._cached_recurrent_state,
            }
        elif q_len > 1 or self._cached_recurrent_state is not None:
            # A multi-token call, or a non-contiguous single token, starts a
            # new sequence in the generic vLLM interface.
            self._cached_conv_state = None
            self._cached_recurrent_state = None
            self._cached_position = None
        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None and q_len > 1:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])

        conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        debug_block = os.environ.get("JET_DEBUG_TRACE") and self.layer_idx == 0 and q_len > 2
        if debug_block:
            print(
                f"JET_DEBUG block0 q_proj_norm={q.float().norm().item():.6f} "
                f"k_proj_norm={k.float().norm().item():.6f} "
                f"input_norm={hidden_states.float().norm().item():.6f} "
                f"q_weight_norm={self.q_proj.weight.float().norm().item():.6f} "
                f"q_weight_sum={self.q_proj.weight.float().sum().item():.6f} "
                f"q_weight_head={self.q_proj.weight.float().flatten()[:4].tolist()} "
                f"v_weight_norm={self.v_proj.weight.float().norm().item():.6f}",
                flush=True,
            )


        q = F.silu(q)
        k = F.silu(k)

        conv_state = None
        if last_state is not None:
            conv_state = last_state['conv_state']
        else:
            if cache is not None:
                conv_state = hidden_states.new_zeros(
                    batch_size,
                    self.value_dim,
                    self.conv_size
                )
        x = self.v_proj(hidden_states)
        if debug_block:
            print(f"JET_DEBUG block0 v_proj_norm={x.float().norm().item():.6f}", flush=True)
        v, conv_state = self.dynamic_conv1d(
            x=x,
            generator_input=hidden_states,
            mask=conv_mask,
            cache=conv_state,
            output_final_state=True,
        )

        if debug_block:
            print(
                f"JET_DEBUG block0 q_silu_norm={q.float().norm().item():.6f} "
                f"k_silu_norm={k.float().norm().item():.6f} "
                f"v_conv_norm={v.float().norm().item():.6f}",
                flush=True,
            )

        if attention_mask is not None and q_len > 1:
            q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            k = index_first_axis(rearrange(k, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            v = index_first_axis(rearrange(v, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        beta = self.b_proj(hidden_states)
        beta = beta.sigmoid()

        a = self.a_proj(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk' and not torch.compiler.is_compiling():
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                autotune_interval=self.autotune_interval
            )
        else:
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )
        # else:
        #     raise NotImplementedError(f"Not supported mode `{mode}`.")

        if cache is not None:
            cache.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=q_len
            )
        elif recurrent_state is not None:
            self._cached_recurrent_state = recurrent_state.detach()
            self._cached_conv_state = (
                conv_state.detach() if conv_state is not None else None
            )
            self._cached_position = position_end

        g_proj = self.g_proj(hidden_states)
        g = rearrange(g_proj, '... (h d) -> ... h d', d=self.head_v_dim)
        o = self.o_norm(o, g)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if debug_block:
            print(f"JET_DEBUG block0 output_norm={o.float().norm().item():.6f}", flush=True)
        if attention_mask is not None and q_len > 1:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        if unsqueezed:
            o = o.squeeze(0)

        return o, None, cache
