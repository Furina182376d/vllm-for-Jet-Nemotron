# Jet-Nemotron 从 HF 到 vLLM 的适配改动

`a4c20d11e9d899d306a9b21427b0e4f6cadb019c`（`original`）是模型最初的 Hugging Face Transformers 实现。之后 `main` 分支从 `stable1` 到最新的 `tests` 提交，目标不是改变 Jet-Nemotron 的网络结构，而是把它改造成 vLLM 可以实例化、切分、调度、加载和执行的模型。

HF 的调用方式是“给模型完整输入，模型返回 `CausalLMOutput`”；vLLM 的调用方式是“由引擎管理 batch、位置、KV cache 和 logits，模型只处理当前 stage 的 token”。因此改动主要集中在运行时接口和状态管理上。

## 总体对比

| 方面 | `original` HF 实现 | `main` vLLM 实现 |
| --- | --- | --- |
| 模型构造 | `JetNemotronConfig` | `VllmConfig`、`CacheConfig`、`QuantizationConfig`、`prefix` |
| forward 接口 | `input_ids`、attention mask、HF `past_key_values` | `input_ids`、绝对 `positions`、扁平 token、`IntermediateTensors` |
| Attention | HF attention / 普通线性层 | vLLM `Attention`、`QKVParallelLinear`、`RowParallelLinear` |
| Embedding / LM head | 普通 embedding，forward 内产生 logits | `VocabParallelEmbedding`、`ParallelLMHead`、`LogitsProcessor` |
| 并行 | 一次执行完整模型 | Tensor Parallel、Pipeline Parallel、按 stage 执行 layer |
| Cache | HF `JetNemotronCache` 传入模型 | vLLM attention context；JetBlock 保留自己的递归/卷积状态 |
| 输出 | `CausalLMOutputWithPast` | forward 返回 hidden states，`compute_logits()` 单独计算 logits |
| 权重加载 | HF `from_pretrained` | 自定义 `load_weights()` 处理分片、量化和 PP |
| 注册方式 | HF `AutoModel` 体系 | vLLM `ModelRegistry` 插件注册 |

## 1. 将模型生命周期改为 vLLM 生命周期

### 具体做了什么

将顶层 `JetNemotronForCausalLM` 从 `PreTrainedModel` 改为 `nn.Module`，构造函数从：

```python
JetNemotronForCausalLM(config)
```

改为接收：

```python
JetNemotronForCausalLM(vllm_config=vllm_config, prefix=prefix)
```

模型内部保存 `vllm_config.model_config.hf_config`、`cache_config` 和 `quant_config`，并将 `prefix` 传递给每个子模块。

### 为什么要这样做

vLLM 的 model loader 和并行运行时会在构造阶段注入这些配置。`prefix` 用来生成稳定的参数路径，使多个 stage 或嵌套模块的权重名称与 vLLM loader 对齐。HF 的 `config` 单独构造方式无法提供这些运行时对象。

### 结果

模型可以被 vLLM 的模型执行器创建，而不是只能由 Transformers 的 `from_pretrained()` 创建。

对应代码：[modeling_jet_nemotron.py:794](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:794)

## 2. 将 Attention 替换为 vLLM Attention

### 具体做了什么

`JetNemotronAttention` 做了以下替换：

- HF 的 Q/K/V 投影改为 vLLM 的 `QKVParallelLinear`。
- 输出投影改为 `RowParallelLinear`。
- HF attention 计算改为 vLLM `Attention` 内核。
- 使用 `get_tensor_model_parallel_world_size()` 计算当前 rank 的本地 query head 和 KV head 数量。
- 使用 vLLM `get_rope()` 生成 rotary embedding。
- 将 `cache_config`、`quant_config`、`attn_type` 和滑动窗口参数传给 vLLM attention。
- 对不同 vLLM 版本的 `Attention.forward()` 做签名检测：新版本传 `output_shape`，旧版本使用旧调用形式。

forward 也从 HF 风格的多参数调用改为：

```python
self.attn(
    positions=positions,
    hidden_states=hidden_states,
)
```

并统一返回 `(output, None, cache)`。

### 为什么要这样做

vLLM 的 Attention 内核负责 paged KV cache、CUDA kernel、batch 调度和 attention metadata。继续使用 HF attention 会绕过这些机制，无法使用 vLLM 的增量解码和显存管理。Tensor-parallel 还要求每个 rank 只计算自己的 head 分片。

### 结果

普通全注意力和滑动窗口注意力都可以走 vLLM 的 attention runtime，KV cache 不再由模型手工拼接，而由 vLLM 的 forward context 管理。

对应代码：[modeling_jet_nemotron.py:117](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:117)

## 3. 保留 HF 权重布局，避免错误的参数打包

### 具体做了什么

Attention 使用 vLLM 的并行投影，但 MLP 的 `gate_proj`、`up_proj` 和 JetBlock 内部投影仍然使用普通 `nn.Linear`，没有强行改成 vLLM 的 packed 层。`JetNemotronMLP` 仍按：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

执行。

### 为什么要这样做

该 checkpoint 的 gate/up 权重以独立 HF 参数保存。直接使用 packed tensor-parallel 层可能在权重加载时再次做 shard/拼接变换，即使 TP=1 也可能改变参数对应关系。main 中因此把“需要 vLLM kernel 的层”和“必须保持 checkpoint 布局的层”分开处理。

### 结果

权重名称和矩阵布局保持与 HF checkpoint 一致，同时 Attention 仍能使用 vLLM 的并行实现。

## 4. 改造 DecoderLayer 和 residual 传递

### 具体做了什么

`JetNemotronRMSNorm.forward()` 增加 `residual` 参数。当传入 residual 时，它返回归一化结果和更新后的 residual：

```text
residual = residual + hidden_states
return norm(residual), residual
```

`JetNemotronDecoderLayer.forward()` 也从只返回 hidden state 改为返回：

```text
(hidden_states, residual, None)
```

并根据 `layer_type` 选择 full attention、sliding attention、JetBlock 或 dynamic convolution。

### 为什么要这样做

vLLM 的 decoder 执行路径允许 residual 在层之间携带，避免每个子层都立即生成新的完整 hidden tensor。这既匹配 vLLM 的模型接口，也便于 Pipeline Parallel 在 stage 之间传输固定字段。

### 结果

模型层可以直接嵌入 vLLM 的 decoder 执行循环，并为非末级 pipeline stage 生成标准的 hidden/residual 中间结果。

对应代码：[modeling_jet_nemotron.py:351](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:351)

## 5. 加入 Pipeline Parallel 支持

### 具体做了什么

`JetNemotronModel` 使用 `get_pp_group()` 做 rank 判断：

- 首 rank 创建 `VocabParallelEmbedding`，其他 rank 使用 `PPMissingLayer`。
- 通过 `make_layers()` 创建当前 stage 负责的 layer 区间，并从参数前缀解析真实 `layer_idx`。
- 末 rank 创建最终 RMSNorm，其他 rank 使用 `PPMissingLayer`。
- 非末 rank 返回 `IntermediateTensors({"hidden_states", "residual"})`。
- 末 rank 才执行最终 norm 并返回可用于 logits 的 hidden states。
- 为 tied embeddings 的配置，在末 rank 也保留必要的 embedding 权重。

### 为什么要这样做

Pipeline Parallel 的每个进程只拥有模型的一段。若每个 rank 都创建完整 embedding、norm 和所有 layer，会浪费显存并破坏 vLLM 的 stage 调度。`IntermediateTensors` 是 vLLM 用于 stage 间传递中间状态的标准容器。

### 结果

同一个模型类可以在单卡、多 TP rank 或多 PP stage 下运行，而不需要为每种并行方式复制一套模型代码。

对应代码：[modeling_jet_nemotron.py:549](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:549)

## 6. 将 forward 和 logits 计算拆成 vLLM 接口

### 具体做了什么

模型 forward 改为接收 vLLM 传入的：

- 扁平化的 `input_ids`
- 绝对位置 `positions`
- 可选 `inputs_embeds`
- 可选的 `intermediate_tensors`

forward 只负责 embedding、执行当前 stage 的 layer 和返回 hidden/intermediate state。新增 `compute_logits(hidden_states)`，通过 `LogitsProcessor` 和 LM head 产生 logits。

同时新增：

- `embed_input_ids()`：供 vLLM 在需要时单独获取 embedding。
- `make_empty_intermediate_tensors`：创建 PP 通信所需的空结构。
- `set_aux_hidden_state_layers()`：为 Eagle3 等 speculative decoding 指定额外 hidden layer。
- `get_eagle3_aux_hidden_state_layers()`：返回默认的辅助层位置。

### 为什么要这样做

vLLM 会根据 sampling、惩罚项、词表并行和 speculative decoding 决定何时以及如何处理 logits。模型 forward 内部直接返回 HF `CausalLMOutput` 会绕过这些步骤，也无法让引擎只在需要时计算 logits。

### 结果

模型输出符合 vLLM 的“hidden states -> logits processor -> sampler”调用链，并支持 LoRA、Pipeline Parallel 和 Eagle3 能力探测。

对应代码：[modeling_jet_nemotron.py:846](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:846)

## 7. 实现 vLLM 权重加载和参数映射

### 具体做了什么

在 `JetNemotronModel.load_weights()` 中遍历 `(name, loaded_weight)`，并按以下顺序处理：

1. 跳过不需要加载的 `rotary_emb.inv_freq`。
2. 为缺少 `model.` 前缀的 checkpoint 名称补前缀。
3. 查询 quantization config 对应的 cache scale，并调用参数的 `weight_loader`。
4. 将 dynamic convolution 的 checkpoint 名称转换为 vLLM 模块名称。
5. 直接加载 JetBlock 和普通 HF-layout 参数。
6. 将 `q_proj/k_proj/v_proj` 映射到 `qkv_proj`，并传入 `q/k/v` shard id。
7. 将 `gate_proj/up_proj` 映射到 packed 名称 `gate_up_proj`，传入对应的 shard id。
8. 使用 `maybe_remap_kv_scale_name()` 处理不同 vLLM 版本的量化 scale 名称。
9. 使用 `is_pp_missing_parameter()` 跳过当前 pipeline stage 不拥有的参数。

顶层 `JetNemotronForCausalLM.load_weights()` 再通过 `AutoWeightsLoader` 加载整个模型；当 `tie_word_embeddings=True` 时跳过独立的 `lm_head` 权重，直接复用 embedding 权重。

### 为什么要这样做

HF 的 `from_pretrained()` 按完整模型参数树加载，而 vLLM 的 loader 需要将 checkpoint 参数分发到 TP/PP 分片，并通过各层的 `weight_loader` 完成切片或量化处理。QKV 和 gate/up 的 checkpoint 名称与 vLLM packed 参数名也不一致，必须显式映射。

### 结果

HF checkpoint 可以直接交给 vLLM loader，在 TP、PP 和量化场景下正确落到对应参数上。

对应代码：[modeling_jet_nemotron.py:678](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:678)

## 8. 适配 JetBlock 和动态卷积的状态管理

### 具体做了什么

JetBlock 的接口从 HF 风格：

```text
hidden_states, past_key_value, attention_mask, use_cache
```

改为 vLLM 风格的 `positions`、`hidden_states`、`cache`，返回 `(output, None, cache)`。同时：

- 支持 `[T, D]` 和 `[B, T, D]` 两种输入形状。
- 根据 `positions` 判断单 token 是否紧接上一次 decode，避免复用错误请求的状态。
- 在没有显式 cache 但连续单 token decode 时，保存 `_cached_conv_state`、`_cached_recurrent_state` 和 `_cached_position`。
- 新的 prefill 或不连续位置会清空本地状态。
- 编译状态下避免调用不兼容的 chunk kernel，改用 fused recurrent kernel。
- dynamic convolution 包装为 `DynamicShortConvolutionVLLM`，统一返回 vLLM 所需的三元组。

### 为什么要这样做

JetBlock 不是普通 KV attention，而是依赖卷积状态和 gated-delta recurrent state。vLLM 的通用 attention cache 不会自动保存这些状态；如果不额外处理，单 token decode 会从错误的初始状态开始，输出会偏离 HF 实现。

### 结果

JetBlock 和 dynamic convolution 可以参与 vLLM 的 prefill/decode 流程，同时保持跨 token 的递归状态。

对应代码：[jet_block.py](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/jet_block.py:66)

## 9. 配置、版本兼容和编译支持

### 具体做了什么

- `configuration_jet_nemotron.py` 增加 layer type 别名，将历史的 `attn`、`swa`、`jet` 映射为 `full_attention`、`sliding_attention`、`linear_attention`。
- 对不同 Transformers 版本的 RoPE 校验接口做 fallback。
- Attention 导入同时兼容旧版 `vllm.attention.*` 和新版 `vllm.model_executor/v1.*` 路径。
- 兼容新版/旧版 `get_rope()` 参数形式。
- 使用 `support_torch_compile` 和 shape invariants 注册模型，允许 vLLM 编译模型执行图。
- 新增 `interfaces.py`、`interfaces_base.py`，提供 vLLM 的 LoRA、PP、Eagle3 等能力协议。

### 为什么要这样做

开发过程中使用的 vLLM/Transformers 版本接口存在变化。如果只绑定单一版本，模型会在导入 attention、RoPE 或 capability 检测时失败。`torch.compile` 还要求 vLLM 知道哪些维度可以动态变化，否则编译图无法复用。

### 结果

模型能够在多个 vLLM API 版本间工作，并可以被 vLLM 的编译和能力探测逻辑识别。

## 10. 注册插件和验证工具

### 具体做了什么

- `jetai/vllm_plugin/__init__.py` 导入模型并自动调用 `ModelRegistry.register_model()`。
- 注册名统一为 `JetNemotronForCausalLM`，并在退出时通过 `atexit` 再次确保注册。
- 新增/更新 `test-vllm.py`、`test-hf-reference.py`、`test.sh` 和输出报告，用于比较 HF 与 vLLM 的结果、验证生成和测速。
- 增加 `jetai.utils.debug`，支持通过 `JET_DEBUG` 等环境变量输出层级、权重、cache 和 hidden-state 诊断信息。

### 为什么要这样做

vLLM 不会自动从 HF `model_type` 找到仓库内的自定义模型，必须先注册模型类。HF 对照测试和 debug 输出用于定位权重映射、残差传递、cache 状态等适配问题。

### 结果

启动 vLLM 时可以通过插件发现 Jet-Nemotron，并有一套对照和回归手段检查适配后的数值行为。

## 结论

这次改造的核心不是重新设计 Jet-Nemotron，而是完成以下运行时转换：

```text
HF config/model/cache/output
        ->
vLLM config/parallel layers/runtime cache/hidden-state + logits pipeline
```

HF 版本负责标准 Transformers 生态中的加载和生成；`main` 版本负责 vLLM 所需的模型注册、Tensor Parallel、Pipeline Parallel、运行时 attention/cache、分片权重加载、量化适配和 logits 处理。
