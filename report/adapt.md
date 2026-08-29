# Jet-Nemotron 从 HF 到 vLLM 的适配改动

`a4c20d11e9d899d306a9b21427b0e4f6cadb019c`（`original`）是 Jet-Nemotron 最初的 Hugging Face Transformers 实现。之后 `main` 分支从 `stable1` 到最新的 `tests` 提交，主要把模型的调用契约改成 vLLM 的运行时契约，模型核心数学结构基本保持不变。

## 总体对比

| 方面 | `original` HF 实现 | `main` vLLM 实现 |
| --- | --- | --- |
| 模型构造 | `JetNemotronConfig` | `VllmConfig`、`CacheConfig`、`QuantizationConfig`、`prefix` |
| forward 接口 | `input_ids`、attention mask、HF `past_key_values` | `input_ids`、绝对 `positions`、扁平 token、`IntermediateTensors` |
| Attention | HF attention / `nn.Linear` | vLLM `Attention`、`QKVParallelLinear`、`RowParallelLinear` |
| Embedding / LM head | 普通 `nn.Embedding`、HF logits | `VocabParallelEmbedding`、`ParallelLMHead`、`LogitsProcessor` |
| 并行 | 单模型完整执行 | Tensor Parallel、Pipeline Parallel、分段 layer |
| Cache | HF `JetNemotronCache` | vLLM attention context；JetBlock 额外支持本地递归/卷积状态 |
| 输出 | `CausalLMOutputWithPast`，forward 内计算 logits | forward 返回 hidden states，单独通过 `compute_logits()` 计算 logits |
| 权重加载 | HF `from_pretrained` | 自定义 `load_weights()`，处理分片、QKV/gate-up 映射、量化 scale 和 PP 缺失层 |
| 注册方式 | HF `AutoModel` 体系 | `ModelRegistry.register_model("JetNemotronForCausalLM", ...)` |

## 主要修改

### 1. Attention 接入 vLLM

`JetNemotronAttention` 改为使用 vLLM 的注意力内核和并行线性层，按 tensor-parallel world size 计算本地 head 数量，并通过 `get_rope()` 使用 vLLM 的 RoPE。KV cache 不再由 HF 模型显式传给 attention，而是交给 vLLM 的 attention runtime 管理。

对应代码：[modeling_jet_nemotron.py:117](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:117)

### 2. 改造 Pipeline Parallel

`JetNemotronModel` 根据 `get_pp_group()` 判断当前 rank：

- 首 rank 负责 embedding。
- 中间 rank 只执行本地 layer。
- 尾 rank 负责最终 norm 和 logits 所需的 hidden states。
- stage 之间通过 `IntermediateTensors({"hidden_states", "residual"})` 传递数据。
- 使用 `make_layers()`、`PPMissingLayer` 支持分段加载。

对应代码：[modeling_jet_nemotron.py:549](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:549)

### 3. 适配 vLLM 的 residual 接口

RMSNorm 和 DecoderLayer 不再只返回一个 hidden tensor，而是维护 `hidden_states + residual`。这匹配了 vLLM decoder layer 的 residual-carrying 执行方式，并减少重复加法和中间张量。

### 4. 重写 CausalLM 外壳

`JetNemotronForCausalLM` 从 HF 的 `PreTrainedModel` 改为 vLLM 的 `nn.Module` 模型接口，新增：

- `compute_logits()`
- `embed_input_ids()`
- `make_empty_intermediate_tensors`
- LoRA / PP / Eagle3 capability interfaces
- `set_aux_hidden_state_layers()`

对应代码：[modeling_jet_nemotron.py:794](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:794)

### 5. 自定义权重加载

vLLM 不直接走 HF 的参数加载流程，因此增加了映射逻辑：

- `q_proj/k_proj/v_proj` 合并到 `qkv_proj`。
- `gate_proj/up_proj` 映射到 `gate_up_proj`。
- 处理 TP shard id。
- 处理量化 cache scale。
- 跳过 Pipeline Parallel 当前 stage 不存在的参数。
- 保留 JetBlock 和 dynamic convolution 的独立 HF 权重布局。

对应代码：[modeling_jet_nemotron.py:678](/home/tjy/codebases/jet-nemotron/jetai/modeling/hf/modeling_jet_nemotron.py:678)

### 6. JetBlock / 动态卷积适配

原本 JetBlock 使用 HF 风格 cache；`main` 中改为接受 vLLM 风格的 `positions`、`cache` 并返回 `(output, None, cache)`。同时增加单 token decode 时的本地状态缓存，以便 vLLM 的通用模型接口没有显式传递 HF cache 时仍能保持自回归状态。

### 7. 注册和兼容层

新增 `interfaces.py`、`interfaces_base.py`、vLLM plugin 注册逻辑，以及不同 vLLM 版本的导入兼容代码。配置中还增加了 `attn/full_attention`、`swa/sliding_attention`、`jet/linear_attention` 等 layer type 别名。

## 结论

这次改造的核心目标不是重新设计 Jet-Nemotron，而是把它包装成 vLLM 能够识别、分片、并行执行、加载权重、管理 cache 和计算 logits 的模型。HF 版本负责标准 Transformers 生态中的加载和生成；`main` 版本则实现 vLLM 所需的模型注册、Tensor Parallel、Pipeline Parallel、运行时 attention/cache、分片权重加载和 logits 处理。
