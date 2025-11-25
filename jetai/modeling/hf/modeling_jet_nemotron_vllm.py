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
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention import Attention, AttentionType
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)





from .configuration_jet_nemotron import JetNemotronConfig
from .jet_block import JetBlock
from .kv_cache import JetNemotronCache

try:
    from .dynamic_conv import DynamicShortConvolution
    from .dconv_fwdbwd import dynamic_conv_triton_autograd
    from .dconv_fwd_cache import dynamic_conv_triton_cache
    from .dconv_step import causal_conv_step_triton
except ImportError:
    raise ImportError(
        "Dynamic convolution is not available. Please install the required dependencies to use this feature."
    )

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "jet-ai/Jet-Nemotron-2B"
_CONFIG_FOR_DOC = "JetNemotronConfig"


class JetNemotronMLP(nn.Module):
    def __init__(self, config: JetNemotronConfig, quant_config: Optional[QuantizationConfig] = None,prefix: str = "",):
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

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class JetNemotronAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: JetNemotronConfig, layer_idx: Optional[int] = None, sliding_window: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "",):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
         # === 基础参数 ===
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.sliding_window = sliding_window
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.attn_type = getattr(config, "_attn_implementation", AttentionType.DECODER)
        
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0, "num_heads must be divisible by TP size"
        self.num_heads = self.total_num_heads // tp_size

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

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
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=self.attn_type)

        # # Decide backend based on sliding_window or layer type
        # layer_type = None
        # if layer_idx is not None and hasattr(config, "layer_types"):
        #     layer_type = config.layer_types[layer_idx]

        # if sliding_window is not None or layer_type == "swa":
        #     # You may have a vLLM-compatible SlidingWindowAttention implementation.
        #     # Replace SlidingWindowAttention below with your project's class if different.
        #     self.attn = SlidingWindowAttention(
        #         num_heads=self.num_heads,
        #         head_dim=self.head_dim,
        #         scaling=self.scaling,
        #         num_kv_heads=self.num_kv_heads,
        #         window_size=sliding_window or config.efficient_attention_config["swa"]["window_size"],
        #         cache_config=cache_config,
        #         quant_config=quant_config,
        #         prefix=f"{prefix}.swa" if prefix else f"layers.{layer_idx}.swa",
        #     )
        # elif layer_type in getattr(config, "efficient_attention_config", {}) and layer_type in EFFICIENT_ATTENTION_CLASSES:
        #     backend_cls = EFFICIENT_ATTENTION_CLASSES[layer_type]
        #     self.attn = backend_cls(config, layer_type, layer_idx)
        # else:
        #     self.attn = Attention(
        #         num_heads=self.num_heads,
        #         head_dim=self.head_dim,
        #         scaling=self.scaling,
        #         num_kv_heads=self.num_kv_heads,
        #         cache_config=cache_config,
        #         quant_config=quant_config,
        #         prefix=f"{prefix}.attn",
        #         attn_type=self.attn_type,
        #     )
               

    def _get_target_length(
        self,
        sequence_length: int,
        past_key_values: JetNemotronCache,
    ):
        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        target_length = sequence_length + min(past_seen_tokens, self.sliding_window - 1)
        return target_length

    def _update_causal_mask_for_sliding_window(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        past_key_values: JetNemotronCache,
    ) -> torch.Tensor:

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = self._get_target_length(sequence_length, past_key_values)

        past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + sequence_length, device=input_tensor.device
        )
        
        if attention_mask is not None:
            # left padding
            assert attention_mask.dim() == 4, "Attention mask must be 4D"
            diagonal_attend_mask = attention_mask < -1
            diagonal_attend_mask = diagonal_attend_mask[:, :, :, -target_length:]
        else:
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            diagonal_attend_mask = diagonal_attend_mask[None, None, :, :]
                
        if past_key_values is None or target_length > self.sliding_window:
            # training mode or prefill mode when dealing with long prefix)
            sliding_attend_mask = torch.arange(past_seen_tokens + sequence_length, device=device)[-target_length:] <= (
                cache_position.reshape(-1, 1) - self.sliding_window
            ) # bs, sequence_length, target_length
            sliding_attend_mask = sliding_attend_mask[None, None, :, :]            
            
            diagonal_attend_mask = diagonal_attend_mask | sliding_attend_mask

        # training
        causal_mask = torch.full(
            (1, 1, sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        causal_mask = causal_mask * diagonal_attend_mask
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class JetNemotronRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JetNemotronRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


EFFICIENT_ATTENTION_CLASSES = {
    "jet": JetBlock,
}


class JetNemotronDecoderLayer(nn.Module):
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
        self.layer_idx = layer_idx

        layer_type = config.layer_types[layer_idx]
        if layer_type == "attn":
            self.self_attn = JetNemotronAttention(
                config,
                layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        elif layer_type == "swa":
            assert config.efficient_attention_config is not None, "Efficient attention config must be provided in JetNemotronConfig."
            assert "swa" in config.efficient_attention_config, (
                "Sliding Window Attention is enabled but no `swa` configuration found in `efficient_attention_config`."
            )
            self.self_attn = JetNemotronAttention(
                config,
                layer_idx,
                sliding_window=config.efficient_attention_config["swa"]["window_size"],
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            assert layer_type in EFFICIENT_ATTENTION_CLASSES, (
                f"Layer type {config.layer_types[layer_idx]} not supported. Supported types are: "
                f"{['attn', 'swa'] + list(EFFICIENT_ATTENTION_CLASSES.keys())}"
            )
            self.self_attn = EFFICIENT_ATTENTION_CLASSES[layer_type](
                config,
                layer_type,
                layer_idx,
                prefix=f"{prefix}.self_attn",
            )

        self.mlp = JetNemotronMLP(config)
        self.input_layernorm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        if output_attentions:
            hidden_states, self_attn_weights = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                output_attentions=True,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs, residual



class JetNemotronRotaryEmbedding(nn.Module):
    def __init__(self, config: JetNemotronConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


JET_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`JetNemotronConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Jet-Nemotron Model outputting raw hidden-states without any specific head on top.",
    JET_START_DOCSTRING,
)
class JetNemotronPreTrainedModel(PreTrainedModel):
    config_class = JetNemotronConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JetNemotronDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


JET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Jet Nemotron Model outputting raw hidden-states without any specific head on top.",
    JET_START_DOCSTRING,
)
class JetNemotronModel(nn.Module):
    """
    vLLM-style Jet Nemotron model with Jet-specific causal mask helpers preserved.

    Notes:
      - Forward accepts optional `attention_mask` and `cache_position` to preserve Jet's mask semantics.
      - Layers are expected to accept (positions, hidden_states, residual, attention_mask=None, cache_position=None)
        OR a reduced signature. If your layers do not accept attention_mask/cache_position, remove these from the call.
    """

    def __init__(
        self,
        *,
        vllm_config,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = JetNemotronDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Check sliding window consistency (same as Qwen2 check)
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            if config.max_window_layers < config.num_hidden_layers:
                raise ValueError(
                    "Sliding window for some but not all layers is not supported. "
                    f"max_window_layers = {config.max_window_layers}, num_hidden_layers = {config.num_hidden_layers}."
                )

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        # Preserve attention implementation flag if exists
        self._attn_implementation = getattr(config, "_attn_implementation", None)

        # Embedding: only on first rank (or tied last rank)
        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False) and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Rotary embedding instance (if needed by layers)
        # Note: some designs compute position embeddings at model-level; we keep rotary object for compatibility.
        self.rotary_emb = JetNemotronRotaryEmbedding(config=config)

        # Layers: pipeline-friendly construction
        decoder_layer_type = decoder_layer_type or JetNemotronDecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: decoder_layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=layer_prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states", "residual"], config.hidden_size)
        )

        # Final norm only present on last pp rank
        if get_pp_group().is_last_rank:
            self.norm = JetNemotronRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        else:
            self.norm = PPMissingLayer()

        # store cache/sliding config locally to allow mask helpers to use it if needed
        self.cache_config = cache_config

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # ------------------------------
    # Forward: vLLM style but preserves Jet mask behavior
    # ------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # Optional: preserve Jet transformers inputs for mask generation
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Args:
            input_ids: [batch, seq_len] (used on first pp rank)
            positions: [batch, seq_len] absolute positions (passed to layers / rotary)
            intermediate_tensors: for non-first ranks, contains hidden_states/residual
            inputs_embeds: optional precomputed embeddings (bypass embedding lookup)
            attention_mask / cache_position: optional, used to build causal masks similarly to HF Jet implementation
        Returns:
            - IntermediateTensors for mid-pipeline ranks
            - final hidden states on last rank
        """

        # --- Prepare hidden_states depending on pipeline rank ---
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None

            # prepare causal mask if attention_mask is provided or Jet's attn implementation requires it
            causal_mask = None
            # If user did not pass cache_position, compute from sequence length (mimic HF behavior)
            if cache_position is None:
                # compute starting index from zero-based past length (no past in vLLM first call)
                seq_len = hidden_states.shape[1]
                cache_position = torch.arange(0, seq_len, device=hidden_states.device)
            # Only prepare causal mask if attention_mask is given or attn implementation expects it
            # This mirrors Jet transformers: some attention impls (flash_attention_2) accept None mask.
            if attention_mask is not None or getattr(self.config, "_attn_implementation", None) == "sdpa":
                causal_mask = self._update_causal_mask(
                    attention_mask=attention_mask,
                    input_tensor=hidden_states,
                    cache_position=cache_position,
                    past_key_values=None,  # vLLM handles cache differently; we pass None here
                    output_attentions=output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False),
                )
        else:
            assert intermediate_tensors is not None, "IntermediateTensors required on non-first pp ranks"
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            causal_mask = None  # mid ranks typically don't recompute mask; pass None unless needed

        # create position embeddings to be shared across the decoder layers (if your layer expects them)
        # If your Jet layers compute rotary inside, you can still compute position_embeddings here for convenience
        # but we do not force its use; layers should accept positions and/or position embeddings.
        # position_embeddings = self.rotary_emb(hidden_states, positions)  # optional - uncomment if used

        # --- Layer execution ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer in self.layers[self.start_layer:self.end_layer]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # If layer signature supports (positions, hidden_states, residual, attention_mask, cache_position)
            # we pass causal_mask and cache_position. If not, you can adapt layers or remove those params.
            try:
                layer_call = functools.partial(
                    layer,
                    # positions first (vLLM-style)
                    positions,
                )
                # Call layer expecting it returns (hidden_states, residual) OR a tuple with attn
                hidden_states, residual = layer_call(
                    hidden_states,
                    residual,
                    attention_mask=causal_mask,
                    cache_position=cache_position,
                )
            except TypeError:
                # Fallback: try calling with reduced signature (positions, hidden_states, residual)
                hidden_states, residual = layer(positions, hidden_states, residual)

            if output_attentions:
                # If layer returns attention, we would need to adapt; many vLLM layers don't return it.
                # For compatibility: you can modify layer to optionally return (hidden_states, residual, attn)
                # Here we try to extract attn if present.
                # (This part is best-effort; if your layer doesn't return attn, all_self_attns remains None/empty)
                pass

        # Last rank: finalize
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        # If RMSNorm accepts residual for fused ops, pass it; otherwise adapt to single-arg call
        try:
            hidden_states, _ = self.norm(hidden_states, residual)
        except TypeError:
            hidden_states = self.norm(hidden_states)

        # attach hidden_states/attentions if the API consumer expects them
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # Adjust mapping for your MLP naming (gate/up/down) if needed
            ("gate_up_proj", "gate_proj", "gate"),
            ("gate_up_proj", "up_proj", "up"),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if (self.quant_config is not None and (hasattr(self.quant_config, "get_cache_scale") and (scale_name := self.quant_config.get_cache_scale(name)))):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                remapped_name = name.replace(weight_name, param_name)
                if remapped_name.endswith(".bias") and remapped_name not in params_dict:
                    continue
                if is_pp_missing_parameter(remapped_name, self):
                    continue
                param = params_dict[remapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(remapped_name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is None:
                    continue
                if is_pp_missing_parameter(remapped_name, self):
                    continue
                param = params_dict[remapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(remapped_name)

        return loaded_params

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: JetNemotronCache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_empty = attention_mask.sum(dim=-1).long() == 0
                last_is_1 = (attention_mask[:, -1].long() == 1) | is_empty
                is_padding_right = last_is_1.sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Jet-Nemotron. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if self.config._attn_implementation == "flex_attention":
            raise NotImplementedError(
                "Flex attention is not supported yet. Please use `flash_attention_2`, `eager`, or `sdpa` instead."
            )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: JetNemotronConfig,
        past_key_values: JetNemotronCache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`JetNemotronConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class JetNemotronForCausalLM(JetNemotronPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: JetNemotronConfig):
        super().__init__(config)
        self.model = JetNemotronModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(JET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[JetNemotronCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("jet-ai/Jet-Nemotron-2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("jet-ai/Jet-Nemotron-2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        assert not generation_config.return_legacy_cache, "Legacy cache is not supported for generation."
        if generation_config.use_cache is False:
            return
        model_kwargs["past_key_values"] = JetNemotronCache()
    
    def _beam_search(self, *args, **kwargs):
        raise NotImplementedError("Beam search is not supported for Jet-Nemotron models.")

    def _contrastive_search(self, *args, **kwargs):
        raise NotImplementedError("Contrastive search is not supported for Jet-Nemotron models.")
    
    def _group_beam_search(self, *args, **kwargs):
        raise NotImplementedError("Group beam search is not supported for Jet-Nemotron models.")
    
    def _constrained_beam_search(self, *args, **kwargs):
        raise NotImplementedError("Constrained beam search is not supported for Jet-Nemotron models.")