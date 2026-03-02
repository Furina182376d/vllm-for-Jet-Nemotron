# Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

### <div align="center"> NeurIPS 2025 <div>

<div align="center">
  <a href="https://hanlab.mit.edu/projects/jet-nemotron/"><img src="https://img.shields.io/static/v1?label=Website&message=Jet-Nemotron&color=darkred&logo=github-pages"></a> &ensp;
  <a href="https://www.arxiv.org/abs/2508.15884"><img src="https://img.shields.io/static/v1?label=arXiv&message=Jet-Nemotron&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/jet-ai/"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Jet-AI&color=yellow&logo=huggingface"></a> &ensp;
  <a href="https://youtu.be/qAQ5yMThhRY"><img src="https://img.shields.io/static/v1?label=Demo&message=Jet-Nemotron&color=yellow"></a> &ensp;
</div>

<p align="center" border-radius="10px">
  <img src="assets/jet-nemotron.png" width="90%" alt="teaser_page1"/>
</p>

## 🔥🔥 News
- (🔥 New) \[2025/9/29\] We released the Jet-Nemotron models and inference code.
- (🔥 New) \[2025/9/18\] Jet-Nemotron is accepted by NeurIPS 2025! 🎉🎉🎉 See you at San Diego!
- \[2025/8/22\] We released the Jet-Nemotron technical report on arXiv.

## 💡 Introduction

Jet-Nemotron is a new family of hybrid-architecture language models that surpass state-of-the-art open-source full-attention language models such as Qwen3, Qwen2.5, Gemma3, and Llama3.2, while achieving significant efficiency gains—up to 53.6× speedup in generation throughput on H100 GPUs (256K context length, maximum batch size). It is built upon two core innovations: 
- **Post Neural Architecture Search**, an efficient post-training architecture exploration and adaptation pipeline applicable to arbitrary pre-trained transformer models; 
- **JetBlock**, a novel linear attention block that significantly outperforms previous designs such as Mamba2.

### Highlight 1: PostNAS – Post-Training Architecture Exploration and Adaptation
Unlike prior methods that train from scratch to explore new model architectures, PostNAS builds on a pre-trained transformer model while enabling flexible exploration of attention block designs, greatly reducing the cost and risk of developing new language model architectures. 

- <ins>PostNAS first identifies the optimal placement of full-attention layers, then searches for improved attention block designs.</ins>
<figure>
  <img src="assets/postnas-roadmap.png" alt="teaser_page2"/>
</figure>

- <ins>In the pre-trained transformer model, not all attention layers contribute equally. PostNAS reveals important attention layers within pre-trained transformer models. </ins>
<figure>
  <img src="assets/search-results.png" alt="teaser_page3"/>
</figure>

- <ins>KV cache size is the most critical factor influencing long-context and long-generation throughput. PostNAS hardware-aware search discovers architectures that deliver similar generation throughput, while having more parameters and achieving better accuracy. </ins>
<figure>
  <img src="assets/hardware-aware.png" alt="teaser_page4"/>
</figure>

### Highlight 2: JetBlock - A New Linear Attention Module with SOTA Accuracy
With PostNAS, we introduce the JetBlock — a novel linear attention module that integrates dynamic convolution with hardware-aware architecture search to enhance linear attention, delivering substantial accuracy gains over previous designs while maintaining similar training and inference throughput. Below, we present an apples-to-apples comparison between the Mamba2 Block and the JetBlock, using identical training data and training recipes.

<p align="center" border-radius="10px">
  <img src="assets/jetblock.png" width="90%" alt="teaser_page5"/>
</p>

### Performance
Jet-Nemotron-2B and Jet-Nemotron-4B match or surpass the accuracy of leading efficient language models (e.g., Qwen3) across a comprehensive benchmark suite while running significantly faster — 21× and 47× faster than Qwen3-1.7B-Base, respectively.
<figure>
  <img src="assets/main-results.png" alt="teaser_page6"/>
</figure>

### Contents
+ [Setup Environments](#1-setup-environments)
+ [Models](#2-models)
+ [Generate with Jet-Nemotron](#3-generate-with-jet-nemotron)
+ [Evaluation on Benchmarks](#4-evaluation-on-benchmarks)
+ [Measure Throughput](#5-measure-throughput)
+ [Build Your Own JetBlock](#6-build-your-own-jetblock)
+ [Contact](#contact)
+ [License](#license)
+ [Bibtex](#-bibtex)


## 1 Setup Environments
```bash
git clone https://github.com/NVlabs/Jet-Nemotron
cd Sana
pip3 install -e .
```

**NOTE**: To install `flash-attn` properly, you may need to install [specific release version](https://github.com/Dao-AILab/flash-attention/releases) or [build from source](https://github.com/Dao-AILab/flash-attention#installation-and-features).

(Optional) To support **[throughput measurement](https://github.com/jet-ai-projects/Jet-Nemotron/tree/main#5-measure-throughput)** or **[chunk-prefilling](https://github.com/jet-ai-projects/Jet-Nemotron/blob/a42b38cafc202709d2eb3e3d75edca694a8ba5b5/jetai/evaluation/meta_eval.py#L47) when eval_batch_size > 1**, please install a modified version of `transformers==4.52.0`:
```bash
pip3 install -U transformers@git+https://github.com/jet-ai-projects/transformers.git@jetai
```

## 2 Models
+ Jet-Nemotron-2B: [jet-ai/Jet-Nemotron-2B](https://huggingface.co/jet-ai/Jet-Nemotron-2B/)
+ Jet-Nemotron-4B: [jet-ai/Jet-Nemotron-4B](https://huggingface.co/jet-ai/Jet-Nemotron-4B/)

Load the model with
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("jet-ai/Jet-Nemotron-2B", 
                                             trust_remote_code=True, 
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")
```
**NOTE**: The kernels in Jet-Nemotron currently do not support running on CPUs. You may get unexpected results on CPUs.

To use or contribute to the model definition files in this repo (`jetai/modeling/hf`), you can first download or soft-link the model weights and model config to `jetai/modeling/hf/`:
```bash
hf download jet-ai/Jet-Nemotron-2B --local-dir jetai/modeling/hf --include "*safetensors*" --include "config.json"
```
Then you can load the model with
```python
model = AutoModelForCausalLM.from_pretrained("jetai/modeling/hf", 
                                             trust_remote_code=True, 
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")
```

## 3 Generate with Jet-Nemotron
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "jet-ai/Jet-Nemotron-2B"

# For local testing, you can use the following path.
# NOTE: Be sure to download or soft-link the model weights to `jetai/modeling/hf`
# model_name_or_path = "jetai/modeling/hf/"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                             trust_remote_code=True, 
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = model.eval().cuda()

input_str = "Hello, I'm Jet-Nemotron from NVIDIA."

input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_str)
```
or 
```bash
python3 jetai/inference/generate.py --model_name_or_path ${PATH_TO_YOUR_MODEL}
```

## 4 Evaluation on Benchmarks
Run evaluation for MMLU, MMLU-pro, BBH, Commonsense, Math, Code, Retrieval, and LongBench Tasks.
```bash
bash scripts/eval/2B/mmlu.sh
bash scripts/eval/2B/mmlu_pro.sh
bash scripts/eval/2B/bbh.sh
bash scripts/eval/2B/commonsense.sh
bash scripts/eval/2B/math.sh
bash scripts/eval/2B/code.sh
bash scripts/eval/2B/retrieval.sh
bash scripts/eval/2B/longbench.sh
```
You can use the first command line argument to specify `model_name_or_path`:
```bash
bash scripts/eval/2B/mmlu.sh ${PATH_TO_YOUR_MODEL}
```

NOTE: The evaluation code will use the `.parquet` version of `social_i_qa`, `mathqa`, and `longbench` data from our repo because their official repos does not supports loading with `datasets >= 4.0.0`.

## 5 Measure Throughput
```bash
python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B
python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-4B --batch_size 64 --prefill_chunk_size 1024
```

<details>
  <summary>Measure Throughput for All Context Lengths</summary>

  ```bash
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 4096 --batch_size 1024 --prefill_chunk_size 256
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 8192 --batch_size 512 --prefill_chunk_size 512
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 16384 --batch_size 512 --prefill_chunk_size 512
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 32768 --batch_size 256 --prefill_chunk_size 1024
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 65536 --batch_size 128 --prefill_chunk_size 2048
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 131072 --batch_size 128 --prefill_chunk_size 2048
  python3 jetai/inference/measure_throuput.py --model_name_or_path jetai/Jet-Nemotron-2B --prompt_len 262144 --batch_size 64 --prefill_chunk_size 2048
  ```

</details>

## 6 Build Your Own JetBlock
The following code is a minimal example to build your own JetBlock.
```python
import torch
from jetai.modeling.hf.jet_block import (
    JetBlock, 
    JetBlockConfig
)

jet_block_config = JetBlockConfig(
    expand_v=2.0,
    num_heads=6,
    head_dim=256,
    conv_size=4,
)

jet_block = JetBlock(
    hidden_size=1536,
    initializer_range=0.02,
    jet_block_config=jet_block_config,
).cuda().to(torch.bfloat16)

hidden_states = torch.randn(16, 4096, 1536).cuda().to(torch.bfloat16)

hidden_states, _ = jet_block(
    hidden_states=hidden_states,
)

print(hidden_states)
```

## License
+ [Code](./LICENSE/code)
+ [Jet-Nemotron Models](./LICENSE/jet_nemotron_models)

## Contact
+ [Han Cai](http://hancai.ai/)
+ [Yuxian Gu](https://t1101675.github.io/)
+ [Song Han](https://hanlab.mit.edu/songhan)

## 📖 BibTeX
```
@article{gu2025jet,
  title={Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search},
  author={Gu, Yuxian and Hu, Qinghao and Yang, Shang and Xi, Haocheng and Chen, Junyu and Han, Song and Cai, Han},
  journal={arXiv preprint arXiv:2508.15884},
  year={2025}
}
```
# Jet-Nemotron



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.tsinghua.edu.cn/tongjy23/jet-nemotron.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.tsinghua.edu.cn/tongjy23/jet-nemotron/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.










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