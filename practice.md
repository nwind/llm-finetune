# 微调实践

本章将通过实例的方式介绍微调，包括简单的问答模型及业界著名模型的实践。

## 基于 TRL 和 LoRA 微调问答模型

除了前面提到的 LLaMA-Factory 工具，我们也可以基于 TRL 库实现 LoRA 微调。虽然目前功能不如 LLaMA-Factory 丰富，但已经包含了绝大部分常用功能。它的优点是 Hugging Face 官方支持，因此未来发展更稳定，贡献者数量也相对更多。

基于 TRL 实现微调的代码如下所示：

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM
from datasets import load_dataset

model_path = "/path_to/Qwen2.5-0.5B"  # 模型路径

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_cache=False  # 关闭 KV 缓存，训练时不需要
)

# 配置 LoRA
peft_config = LoraConfig(
    r=8,  # 小矩阵的秩
    lora_alpha=16,  # 缩放比
    inference_mode=False,  # 关闭推理模式
    target_modules="all-linear",  # 训练哪些模块，推荐全部线性层
    modules_to_save=["lm_head", "embed_token"],  # 默认不训练这两个模块，但加上可以提升效果
    task_type="CAUSAL_LM",  # 任务类型，目前大模型都是因果语言模型
)

output_dir = "lora_output"  # 输出 LoRA 文件目录

# 配置微调参数，这里只列出部分参数，这些参数在微调训练一章有介绍
sft_config = SFTConfig(
    output_dir=output_dir,  # 输出 LoRA 文件目录
    max_seq_length=4096,  # 最大文本长度
    per_device_train_batch_size=4,  # 单个显卡上的批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数
    optim="adamw_torch",  # 优化器
    num_train_epochs=3,  # 训练轮数
    save_steps=10,  # 保存步数
    logging_steps=10,  # 日志步数
    learning_rate=3e-3,  # 学习率
    bf16=True,  # 使用 BF16 精度让训练更加稳定
    warmup_ratio=0.1,  # 预热比率
    lr_scheduler_type="cosine",  # 学习率调度器类型
    gradient_checkpointing=True,  # 梯度检查点，降低显存占用
    # packing=True,  # 将多个样本打包成一个样本，提升训练效率，但显存占用会变大
    # use_liger=True  # 使用 liger 内核优化，需要 pip install liger-kernel
)

# 加载训练数据
dataset = load_dataset("json", data_files="/path/to/zhihu_chatml.jsonl", split='train')

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config
)

trainer.train()  # 开始训练
trainer.save_model(output_dir)  # 保存模型
```

训练数据支持对话和补全两种格式，通常使用对话格式，因为它能支持多轮对话。数据格式类似如下文本的 JSONL 文件：

```txt
{"messages": [{"role": "user", "content": "法国首都在哪里？"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "谁写了《罗密欧与朱丽叶》？"}, {"role": "assistant", "content": "..."}]}
```

训练数据我们使用 COIG-CQIA [^CQIA]，它是一个开源高质量微调指令，来自零一万物、中科院深圳先进技术研究院和 M-A-P 等机构的研究者们，主要语言是中文。它由一个个 JSONL 文件组成，你可以只用其中的部分文件，我们以其中的知乎问答数据为例进行微调训练。

[^CQIA]: <https://modelscope.cn/datasets/m-a-p/COIG-CQIA>

原始文件的每行是类似如下格式：

```json
{
  "instruction": "大学最冷门的专业有哪些？\n最冷门的专业",
  "input": "",
  "output": "1. 上海戏剧学院 木偶戏表演\n\n",
  "task_type": {"major": ["问答"], "minor": ["知乎问答"]},
  "domain": ["通用"],
  "metadata": "qid:67789453, aid:1048529303, tag:[]",
  "answer_from": "human",
  "human_verified": true,
  "copyright": "暂无版权及作者信息"
}
```

由于 TRL 中的数据格式要求与此不一致，我们需要写代码进行转换（需要使用 `pip install jsonlines` 安装 jsonlines 库）。

```python
import sys
import jsonlines

output = []
with jsonlines.open(sys.argv[1]) as reader:
    for data in reader:
        output.append({
            "messages": [
                {"role": "user", "content": data["instruction"]},
                {"role": "assistant", "content": data["output"]}
            ]
        })

with jsonlines.open(sys.argv[2], 'w') as writer:
    writer.write_all(output)
```

然后就能使用下面的命令进行转换：

```bash
python convert_to_chatml.py zhihu.jsonl zhihu_chatml.jsonl
```

接着修改之前代码中的路径后训练即可。如果要多卡训练，可以安装 `accelerate` 库，先使用 `accelerate config` 来配置训练参数，然后使用 `accelerate launch` 来启动训练，比如：

```bash
# 配置多卡或多机环境
accelerate config

# 启动训练，其它参数和前面一样
accelerate launch sft.py
```

启动训练后会在训练 `logging_steps` 步数后打印日志，类似如下：

```txt
{'loss': 2.820, 'grad_norm': 0.71903, 'learning_rate': 0.00299, 'epoch': 0.47}
{'loss': 2.555, 'grad_norm': 0.59527, 'learning_rate': 0.00284, 'epoch': 0.93}
{'loss': 2.305, 'grad_norm': 0.86397, 'learning_rate': 0.00245, 'epoch': 1.37}
{'loss': 2.144, 'grad_norm': 1.01672, 'learning_rate': 0.00190, 'epoch': 1.84}
{'loss': 1.859, 'grad_norm': 1.15081, 'learning_rate': 0.00128, 'epoch': 2.28}
{'loss': 1.521, 'grad_norm': 0.90571, 'learning_rate': 0.00069, 'epoch': 2.74}
```

正常情况下 `loss` 应该逐步降低。如果发现没有明显降低甚至升高，那可能是学习率太低了，或者这个任务对模型太难了，需要调高学习率。最终 `loss` 降到多少比较合适取决于任务类型，简单的任务甚至能降到 `0.1` 以下。

训练之后的模型如何推理？可以参考本书的[使用 vLLM 进行推理]章节。在这个例子中，我们使用知乎问答训练，虽然数据量不大，但模型已经可以回答问题了，比如问「天空为什么是蓝色的？」，模型也能回答，下面是截取的部分回答：

```txt
世界上有很多种原因可以解释天空为什么是蓝色的。一种简单的解释是，蓝色是光的原始波长。光是一种电磁波，它可以用光纤等传播。猫眼对蓝色光比较敏感，在光线很强的星云或恒星的周围更容易看到蓝色影像...
```

可以看到这个 0.5B 的模型幻觉现象比较严重，想要训练好还需要大量数据和调参。

## 通用问答模型 LLaMA 3

LLaMA 3 在预训练基础上进行了 SFT 微调和 DPO 偏好优化，虽然它主要面向通用问答场景，但其中有许多可借鉴的方法。本节将介绍这些训练细节。

整个训练过程经过六轮迭代，每一轮都经过如下几个过程：

**训练「奖励模型」（Reward model）**，这个奖励模型是使用如下步骤训练的：

- 人工收集问题。
- 部署多个模型，每个模型都会对这个问题回答两次，得到两个答案。
- 人工标注这些回答的好坏，同时还支持人工对答案进行编辑，最终得到排序：人工编辑后的答案 > 好的答案 > 差的答案。
- 使用前面的标准来训练奖励模型。

**SFT 微调**，其中的训练数据由三部分组成：

- 人工收集的问题，这些问题没有现成答案，所以先让模型生成多个答案，然后用「奖励模型」选择其中最好的答案，这一步也叫「拒绝采样」。
- 特定领域的合成数据，代码、数学、工具调用等，后面展开介绍。
- 人工编写的问题和答案。

这里展开介绍一下合成数据，LLaMA 3 的合成数据主要有以下几种：

- **代码数据**，这部分数据有 270 万，基于以下三种方式合成：
  - **自然语言生成代码的数据合成**，使用以下过程生成 100 万个编程对话数据：
    - **生成问题描述**，随机抽取代码片段，让模型生成针对这个代码片段的问题描述。
    - **生成代码**，让模型生成代码。
    - **正确性分析**，分析代码是否语法正确，让模型生成单元测试用例来执行。
    - **自我修正**，如果前面出错，就让模型尝试自己修复。
    - **微调和迭代**，用前面得到的数据对模型进行微调，然后下一轮使用这个微调后的模型再继续执行。
  - **编程语言之间的转换**，比如将 Python 代码转成 PHP，解决 PHP 代码较少的问题。
  - **反向翻译**，要求模型基于代码生成解释，和第一种方式正好相反，这部分生成了 120 数据。
- **多语言数据**，由四部分组成：
  - **人工标注**，这部分占比 2.4%。
  - **其它 NLP 任务数据**，这部分占比 44.2%，比如 exams-qa 等公开数据。
  - **拒绝采样数据**，这部分数据占比 18.8%，使用前面提到的「拒绝采样」方式挑选最好的回答，这里的不同之处是要保证语言一致性，比如中文问题应该用中文回答，哪怕对应的英文回答评分可能更高。
  - **翻译数据**，使用机器翻译的文本。
- **工具调用**，使用 Few-Shot 生成问题，然后生成对应的工具调用。
- **数学推理**、**长文本**，对大部分读者可能用处不大，这里就不展开了。

合成数据在训练数据中占比 48%，提升了大模型在这些特殊领域上的能力。

后面我们会看到许多模型使用 GPT-4 生成微调数据，这种做法成本低但按 OpenAI 使用协议是不允许的，所以 LLaMA 3 使用迭代的方式来逐步提升模型能力，先用已有模型输出答案，对这个答案进行人工优化，然后训练新的模型，最后拿这个新模型输出答案并进行人工优化，整个过程进行了六轮迭代，人力成本很高，但它是一种很通用的做法。如果某个垂直领域下 GPT-4 效果很差或无法使用，我们就能使用类似 LLaMA 3 的迭代思路来生成微调数据。

**DPO 偏好对齐**，在 SFT 微调之后，LLaMA 3 进行了 DPO 偏好优化，训练数据使用之前训练奖励模型时的排序，学习率是 1e-5，$beta$ 超参数是 0.1，除了 DPO 也试验过 PPO 训练，但发现 DPO 计算量小，效果也不错，所以就只用 DPO 了。

## 苹果智能模型 AFM

苹果智能模型 Apple Intelligence [^gunterAppleIntelligenceFoundation2024] 是 iPhone 16 上最大的改进，它由一个在手机上运行的 3B 参数量模型 AFM-on-device 和一个在服务器上部署的模型 AFM-server 组成，苹果罕见地公开了模型的预训练和微调细节，我们能从中学习许多经验。

[^gunterAppleIntelligenceFoundation2024]: <http://arxiv.org/abs/2407.21075>

模型整个训练过程如下图所示：

![苹果智能模型训练过程](images/practice/apple_intelligence_overview.png)

整个阶段包括：数据准备、数据预处理、预训练、后训练、优化部署，最后在手机上针对特定任务进行微调。

苹果智能模型的架构和 LLaMA 很相似，使用了 RMSNorm、GQA、RoPE 等方案，其中 AFM-on-device 的词表大小是 49k，AFM-server 的词表大小是 100k。

预训练文本的来源是：

- **网页数据**，使用 Applebot 爬取，然后经过以下处理：
  - 使用 Safari 的阅读模型和 Boilerpipe [^kohlschutterBoilerplateDetectionUsing2010] 算法提取页面正文。
  - 使用启发式和基于模型的分类器进行安全性和脏话过滤。
  - 使用局部敏感的 n-gram 哈希进行全局模糊去重。
  - 使用基于模型 [^kongLargeLanguageModelguided2024] [^liDataCompLMSearchNext2024] 的分类器进行质量过滤。
  - 使用 4-13 gram 来过滤 811 个常见评测指标数据集。（主要是为了避免影响后续评测的可信度，这点可以看出苹果不在乎跑分，而是关注模型实际效果）
- **版权数据**，从出版商获取，这些高质量的长文本数据主要用于扩充上下文训练。
- **代码**，从 Github 上根据开源协议过滤，包含 14 种常见语言，使用和网页数据类似的方法进行过滤。
- **数学**，包括来自网络的数学问答数据集，并基于特殊字符来自动识别网页中的数学内容，然后进行去重和清理。
- **公开数据集**，过滤其中的个人信息。

[^kohlschutterBoilerplateDetectionUsing2010]: <https://dl.acm.org/doi/10.1145/1718487.1718542>

[^kongLargeLanguageModelguided2024]: <http://arxiv.org/abs/2406.04638>

[^liDataCompLMSearchNext2024]: <http://arxiv.org/abs/2406.11794>

预训练分为三个阶段：

- **核心阶段**，混合所有文本，训练 token 数量是 6.3T，上下文长度是 4096。
- **持续阶段**，数据降低网页比例，增加代码和数据比例，训练 token 数量是 1T，上下文长度是 8192。
- **长文本扩展阶段**，混入长文本数据和合成的长文本数据，训练 token 数量是 100B，上下文长度是 327678。

其中 AFM-server 使用 8192 个 TPUv4 芯片训练，而 AFM-on-device 是从 AFM-server 修剪后再使用相同训练数据蒸馏出来的，蒸馏使用 2048 个 TPUv5p 训练。

接下来是后训练，包括了指令微调 SFT 和人类对齐 RLHF。

SFT 数据包含以下两部分：

- **人工标注数据**，从各种来源收集，建立了数据质量保障措施，包括人工评级、基于模型的自动过滤方法。
- **合成数据**，包括以下几部分：
  - **数学问题**，主要是依赖问题重写来基于现有问题生成新问题。
  - **工具使用**，首先合成单一工具调用的例子，然后通过人工标准来改进调用多工具能力。
  - **编码**，从 71 个不同编程主题的例子开始，让模型生成新问题，以及对应的代码和单元测试代码，通过执行单元测试来判断是否正确，得到 1.2 万个训练数据。

在指令微调方面，测试了多种混合比例的效果，对每种任务设置一个权重，通过不断调整权重来测试最终模型训练效果。

指令微调之后是基于强化学习的 RLHF 对齐，偏好对齐数据主要来自人工标注，评分标准包括：指令遵循、简洁性、真实性和无害性。

接下来是针对具体功能的微调，主要用于手机端，使用 LoRA 训练，每个任务训练一个单独的 LoRA，LoRA 微调影响所有线性层，对于 3B 模型，每个任务 LoRA 在 FP16 下只需十几 MB，适合动态加载。

比如摘要任务，用于手机上为邮件、消息通知提供摘要能力，这是一个单独的 LoRA，微调数据使用两步生成：

- 从公共数据集、供应商及公司内部提交的示例中收集邮件、消息通知文本。
- 使用 AFM-server 基于前面的文本生成摘要，然后通过使用一个过滤器来过滤不适合的摘要，比如长度过长。

对于摘要功能的评估使用人工评估，每个评分人员必须经过一系列资格培训，包括学士学位中写作相关内容，并在内部质量评测上表现良好。

摘要的评分标准使用以下几个维度，每个维度使用「好」、「一般」、「差」三个分数打分：

- **构成**，评估摘要的整体可读性，考虑语法、标点、拼写和简洁性。
- **全面性**，评估摘要在捕捉要点或为用户指出任何行动、结论方面的全面性。
- **贴切性**，评估摘要与原始内容的贴切程度。不完全贴切的摘要可能包含夸张、推断、不准确或虚构的细节。
- **遵循指示**，评估摘要是否满足特定的风格和格式要求。要求针对每个功能量身定制，反映特定的产品和设计期望。
- **有害性**，评估摘要是否包含根据苹果公司的安全分类属于有害或不安全的内容。

只要有一个维度是「差」，则整个摘要被标记为「差」，只有所有维度都是「好」，这个摘要才被标记为「好」。

最后在手机端的部署使用 4 位量化，甚至某些层还使用 2 位量化，不过 LoRA 依旧使用 FP16 来避免性能损失。

## 深度思考模型 DeepSeek-R1

OpenAI 的 o1 模型开启了新的模型性能提升方法，相当于内置了 CoT 功能，让模型通过大量思考来解决复杂问题，在数学和推理能力上有了显著提升。

不过 o1 模型并未披露训练细节，但最新的开源模型 DeepSeek-R1 在许多评测指标上都达到了 o1 的水平，因此我们可以通过它来了解如何训练一个类似效果的模型。

DeepSeek-R1 不仅在数学和推理方面表现优异，还很擅长解答模糊的问题，比如笔者使用很简单的提示词：「我要开发一款 AI 绘图软件，请帮忙写一份商业计划书」，模型给出的答案将近 2 千字，而且内容详实，效果超越了所有其它问答模型，下面是第一段的内容：

```markdown
## 1. 执行摘要

- 项目名称：明确产品名称（如“ArtGenius AI”）。
- 愿景：成为全球领先的 AI 创意工具，赋能个人与企业的视觉表达。
- 核心价值：
  - 技术优势：基于自研/改进的扩散模型（Diffusion Model）或多模态模型，支持高精度图像生成与风格控制。
  - 差异化功能：例如“行业模板库”“实时协作设计”“3D 绘图融合”等独特功能。
  - 用户体验：极简交互设计，降低非专业用户使用门槛。
- 融资需求：初期计划融资 XXX 万元，用于技术研发、团队扩张与市场推广。
```

因此 DeepSeek-R1 在许多场景下都很有价值，值得研究。

DeepSeek-R1 训练首先训练了一个完全用强化学习的模型 DeepSeek-R1-Zero，它和常见的 RLHF（Reinforcement Learning from Human Feedback）不同，它只有 RL（强化学习），没有 HF（人类反馈），这个训练不需要准备人类偏好数据，因为训练任务是有确定答案的代码生成和数学问题，可以很容易判断是否正确，因此这里的奖励模型只有准确性奖励，所以使用规则方式来判断，规则如下：

1. **准确性奖励**，如果是数学问题，直接判断答案是否正确；如果是代码问题，则使用单元测试来判断。
2. **格式奖励**，判断输出过程有没有正确输出思考过程 `<think>` 和答案 `<answer>`。

在 DeepSeek 的开源窗口里并没有这部分代码的具体实现，不过 Hugging Face 的 Open R1 项目提供了类似的实现，其中的数学校验和格式校验代码如下：

```python
from math_verify import parse, verify

def accuracy_reward(completions, solution, **kwargs):
    """检查生成的答案是否和标准答案一致"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        try:
            answer = parse(content)
            reward = float(verify(answer, parse(sol)))
        except Exception:  # 如果解析失败，则返回 0
            reward = 0.0
        rewards.append(reward)
    # 如果相同就是 1，否则是 0
    return rewards

def format_reward(completions, **kwargs):
    """检查格式是否正确"""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

然后使用如下提示词模板来构造训练数据：

```txt
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}. Assistant:
```

其中的 `{prompt}` 是用户的问题，让模型生成两部分内容，一个是思考阶段，用 `<think>` 标签包裹，另一个是答案，用 `<answer>` 标签包裹。

然后使用 GRPO 方法训练，可以通过调用 TRL 库来实现，将前面的函数传递进去，比如类似下面的代码：

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# 加载问题集，这里只需要问题，不需要答案，类似下面的内容
# {"prompt": "问题 1"}
# {"prompt": "问题 2"}
dataset = load_dataset("json", data_files="prompts.jsonl")

# 训练参数
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    learning_rate=1e-5,
    logging_steps=10,
    gradient_accumulation_steps=16,
    max_completion_length=128,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    # 前面实现的奖励函数
    reward_funcs=[accuracy_reward, format_reward],
    args=training_args,
    train_dataset=dataset,
    peft_config=LoraConfig(task_type="CAUSAL_LM"),
)

trainer.train()
```

目前 LLaMA-Factory 及 torchtune 等训练库都不支持 GRPO 训练，因此只能使用上面的代码来实现。

不可思议的是，仅使用类似上面的简单代码，DeepSeek-R1-Zero 在数学和代码方面的评测就已经接近最强的 OpenAI o1 模型，而且包含了推理过程。这是一项重要发现，因为整个过程无需人工标注，成本很低，为什么之前没人发现？其实之前学术界做过类似尝试，但并不成功，可能是因为使用小模型效果不好，而 DeepSeek-R1-Zero 使用的 DeepSeek-V3-Base 参数量有 671B，这个基础模型在 AIME 2024 评测集上的初始成功率就有 15.6%，这或许是成功的关键。

但 DeepSeek-R1-Zero 有个问题是生成的推理过程可读性差，还会出现多种语言混合输出的情况，作者认为光靠强化学习难以解决这个问题，于是重新训练了新的模型 DeepSeek-R1，训练分为 4 个阶段：

一：冷启动阶段 SFT 微调

使用了多种方法生成初期微调数据，包括：

- 使用 few-shot 长 CoT 示例作为提示词来让 DeepSeek-R1-Zero 生成思考过程。
- 使用 zero-shot 直接让 DeepSeek-R1-Zero 生成思考过程。
- 收集之前 DeepSeek-R1-Zero 训练数据中比较好的回答。
- 人工二次处理。

通过这些方式收集了几千条数据，然后使用 SFT 对 DeepSeek-V3-Base 基础模型进行微调，得到第一个模型。

这个阶段得到了两个好处：

1. **可读性增强**，解决了 DeepSeek-R1-Zero 输出内容可读性差的问题。
2. **能力提升**，训练后的性能优于 DeepSeek-R1-Zero。

二：推理相关的强化学习

接下来针对数学和代码场景进行强化学习训练，这里和前面的 DeepSeek-R1-Zero 是一样的，不过奖励模型增加了对语言一致性的奖励，用于惩罚模型输出中英文混合的内容。

三：拒绝采样和 SFT 微调

这个阶段的主要目的是让模型具备回答通用问题的能力，训练数据分为两部分：

- **推理数据**，使用拒绝采样方法来生成，这部分的推理数据不像前面数学和代码那样容易判断，因此是使用 DeepSeek-V3 模型进行评估，一共有 60 万微调数据。
- **非推理的数据**，主要是来自 DeepSeek-V3 的微调数据，一共有 20 万条。

然后拿这 80 万条数据进行 SFT 微调，训练 2 epoch。

四：面向所有场景的强化学习

最后又进行了一次面向全场景的强化学习训练，这里的奖励方法分为两部分：

- 对于能判断正确的问题，比如数学和代码，使用前面的规则方式判断。
- 对于没有唯一答案的其它问题，使用奖励模型来判断。

以上就是 DeepSeek-R1 的训练过程，它的主要核心是多阶段训练，通过微调和强化学习来一步步提升模型能力，最终得到一个效果不错的模型。

## Qwen2.5 中的后训练方法

经过两年多的发展，Qwen 逐渐成为国内开放权重大模型的主流选择，包括前面提到的 DeepSeek-R1 的小模型蒸馏版本也是使用 Qwen 作为基础模型。本节将介绍 Qwen2.5 中的后训练方法。

Qwen2.5 的后训练分为 3 个阶段：SFT、DPO 和 GRPO。

**一：SFT 阶段**，这个阶段的训练数据有 100 万条，使用 32K 上下文长度训练 2 轮，学习率从 7e-6 减小到 7e-7，裁剪超过 1.0 的梯度，训练数据包括如下几方面：

- **长序列生成**，使用指令回译技术从预训练语料生成长文本查询指令，施加输出长度约束，并采用 Qwen2 进行低质量配对数据过滤。
- **数学能力**，引入 Qwen2.5-Math 思维链数据集，集成公共数据集、基础教育题库及合成问题等多源查询。通过拒绝采样与奖励模型协同机制，结合标注答案引导，构建高质量分步推理流程。
- **代码能力**，融合 Qwen2.5Coder 指令调优数据，采用多语言智能体协同框架，在近 40 种编程语言中生成多样化优质指令对。通过代码问答网站合成新样本及 GitHub 算法代码采集实现数据集扩展，并运用多语言沙盒环境进行静态代码检查与自动化单元测试验证，确保代码质量与正确性。
- **指令遵循**，建立代码验证框架，要求大模型同步生成指令与验证代码，配合单元测试进行交叉验证。基于执行反馈的拒绝采样机制严格筛选监督微调数据，确保模型对指令的精准遵循。
- **结构化数据理解**，构建涵盖传统任务（表格问答、事实验证、纠错解析）与复杂任务（结构化/半结构化数据处理）的综合性数据集。通过响应中嵌入推理链条，显著提升模型从结构化数据中提取信息的能力，实现多任务性能优化。
- **逻辑推理**，新增 7 万条跨领域查询指令，涵盖选择题、判断题及开放式问题。训练模型系统运用演绎推理、归纳概括、类比推理、因果推断及统计推理等方法。通过迭代优化机制筛除错误答案与瑕疵推理数据，持续强化逻辑推理精准度。
- **跨语言迁移**，采用翻译模型将高资源语言指令转化为多低资源语言变体，生成候选响应。通过语义对齐评估确保多语言响应与源语言版本在逻辑结构与风格特征上的一致性，维持跨语言表达的完整性。
- **鲁棒性系统指令**，构建数百条通用系统提示语以增强训练后多样性，确保系统提示与对话情境协调性。多提示评估表明模型保持优异性能且波动率降低，鲁棒性显著提升。
- **响应筛选**，建立多重自动标注体系，包含专用评判模型与多智能体协作评分系统。响应数据需通过严格质量评估，仅全系统一致认可的无瑕样本得以保留，确保输出质量符合最高标准。

**二：DPO 阶段**，这个阶段主要关注数学、编程、指令执行和逻辑推理等客观领域提示词，使用 SFT 生成不同答案，使用人工和自动化来检查结果，正确的作为正面实例，错误的作为负面实例，共构建了 15 万对偏好数据，然后使用 DPO 算法训练，学习率是 7e-7，最后使用 Online Merging Optimizers [^luOnlineMergingOptimizers2024] 方法来更新参数，减少对齐导致的损失。

[^luOnlineMergingOptimizers2024]: <http://arxiv.org/abs/2405.17931>

**三：GRPO 阶段**，这个阶段需要先训练奖励模型，提示词来自两个数据集：开源数据集和一个更为复杂的专有查询集，然后使用不同阶段的 Qwen 模型生成答案。

偏好排序使用人工或自动化标注，注意这里只提到了「自动化标注」，这类开放问题是没法用简单规则标准的，只能是大模型。如果只是使用 Qwen 2 模型进行标注，论文里肯定会说清楚，因此笔者猜测这里可能是用 GPT-4o 等其它模型来标注，没法明说，只好这样模糊地描述了。

标注的标准包括以下几部分：

- **真实性**：回答必须基于事实的准确性，忠实反映提供的背景和指示。模型应避免生成虚假或没有数据支持的信息。
- **有用性**：模型的输出应真正有帮助，有效地解决用户的问题，同时提供积极、吸引人、具有教育意义且相关的内容。应准确遵循给定的指示，并为用户提供价值。
- **简洁性**：回答应简洁明了，避免不必要的冗长。目标是清晰有效地传达信息，而不让用户因过多细节而感到困扰。
- **相关性**：回答的所有部分应与用户的问题、对话历史和助手的上下文直接相关。模型应根据用户的需求和期望量身定制其输出，确保完全符合用户的要求。
- **无害性**：模型必须优先考虑用户安全，避免任何可能导致非法、不道德或有害行为的内容。它应始终促进道德行为和负责任的沟通。
- **去偏见**：模型应生成没有偏见的回答，包括但不限于性别、种族、国籍和政治。它应公平对待所有话题，遵循广泛接受的道德和伦理标准。

有了偏好数据后线训练奖励模型，然后使用 GRPO 算法进行强化学习训练，每个提示词生成 8 个结果。

## 真正开源模型 Tulu 3 的实现细节

前面提到的开源模型及大多数其它开源模型其实都不是真正开源，只能叫开放权重下载，因为它们都不提供相关的训练代码及数据，第三方开发者无法自己训练一个同样的模型。

虽然许多论文中会详细介绍实现细节，但文字说明难免产生歧义，不如代码清晰，目前真正代码和数据都开源的模型大部分是用于教学的小模型，只有 Tulu 3 [^lambertTulu3Pushing2024] 是具备一定竞争力的，它的部分评测指标达到了 GPT-4o Mini 的水平，因此有较大参考价值，本节将详细介绍它的训练细节。

[^lambertTulu3Pushing2024]: <http://arxiv.org/abs/2411.15124>

Tulu 3 的关注后训练而不是预训练，这也是本书的重点，因为预训练模型目前已经很成熟，只需使用开放权重的模型即可。

Tulu 3 的后训练主要包括三阶段：SFT、DPO、RLVR，其中 RLVR 是 Tulu 3 中自创的强化学习方式，下面分别介绍。

### SFT 微调训练阶段

SFT 阶段的训练数据主要来自公开数据集和合成数据，只有 24 条手动创建的数据，这 24 条数据主要是用于回答身份，比如问你是什么模型，所有数据都能在 Hugging Face 上找到[^tulu3_data]。

[^tulu3_data]: <https://huggingface.co/collections/allenai/tulu-3-datasets-673b8df14442393f7213f372>

挑选公开数据集主要考虑这几方面：

- **多样性和高质量**，选择了 WildChat 和 Open Assistant，因为它来自真实用户交互，还有 No Robots，因为它的数据全是人类标注的。
- **特殊能力**，主要是数学及代码相关的，来自 OpenMathInstruct 和 CodeAlpaca。
- **来源及版权**，只使用来源和版权清晰的数据，同时还区分了是否可商用。

合成数据的提示词是使用 GPT-4o 结合 Persona Hub（在 [提升数据多样性] 一节中有介绍）来构造，包括数学、代码和指令跟随任务，以其中的代码为例，构造的提示词如下：

```txt
{persona}
Assume you are the persona described above and you are asking a python programming question in stack overflow.
```

其中的 persona 是角色描述，比如「A machine learning researcher focused on neural networks」，通过这种方式让大模型构造不同人可能会问的问题。

为了避免干扰后面的评估，这些数据集还通过 n-gram 等方式过滤掉和评估数据里类似的数据。

所有微调数据有 94 万条，使用方法是首先使用特殊任务的数据（比如数学和代码）来微调，然后不断融合其它类型的数据，并在训练过程中持续评估效果，如果效果下降就减少某种类型的数据量。

Tulu 3 还做了许多消融实验，得到以下经验：

- **数据多样性是有用的**，如果去掉 WildChat 会导致大部分能力小幅下降。
- **基础模型很重要**，对于数学任务，如果使用的基础模型是 Qwen 2.5 Math 7B，微调后端的分数能赶上 LLaMA 3.1 70B 的分数，因此对于特殊领域的任务，继续预训练可以提升能力。
- **不同模板之间有微小差别**，如果将最后的换行符换成终止符，效果可以提升 0.2。
- **随机种子对结果有微小影响**，这也是为什么训练脚本的配置是 `--seed 123`，但这个属于玄学了。

具体训练方法是自己基于 PyTorch 实现的，使用求和而不是取平均值，避免累计梯度计算的问题，这个问题在 [常见超参数说明] 一节中有介绍。

训练超参数尝试了多种学习率，发现 5e-6 效果最好，使用的轮次为 2，因为发现轮次 3 以上后效果持续下降。

### DPO 偏好训练阶段

偏好训练的数据使用下面三阶段生成：

1. **提示词选取**，从 SFT 阶段的训练数据中挑选，还有部分未参与 SFT 训练的数据。
2. **生成答案**，随机挑选 4 个大模型生成答案，包括 GPT-4o、LLaMA 3.1、Qwen 2.5 等，再使用前面 SFT 之后的 Tulu 3 模型生成一个答案。
3. **偏好标注**，使用 GPT-4o 对答案从乐于助人、遵循指令、诚实和真实性四个方面打分 1-5 分，提示词较长这里不复述，可以查看原论文。

偏好数据一共 35 万，使用 DPO 算法训练，除了原版算法，还测试了两个改进算法：SimPO 和 LN-DPO，发现 LN-DPO 效果最好，所以使用它，这个算法和原版的主要区别是结果除以输出文本长度，下面是两个公式对比：

$$
\text{DPO} = -\log \sigma \left(\beta \log \frac{\pi_\theta (y_w | x)}{\pi_\text{ref} (y_w | x)} - \beta \log \frac{\pi_\theta (y_l | x)}{\pi_\text{ref} (y_l | x)}\right)
$$

$$
\text{LN-DPO} = -\log \sigma \left(\frac{\beta}{|y_w|} \log \frac{\pi_\theta (y_w | x)}{\pi_\text{ref} (y_w | x)} - \frac{\beta}{|y_l|} \log \frac{\pi_\theta (y_l | x)}{\pi_\text{ref} (y_l | x)}\right)
$$

除了 DPO 还尝试了 PPO 算法，PPO 算法的效果和 DPO 差不多，但 PPO 计算成本高，用 2 个节点运行 PPO 耗时 28 小时，而 1 个节点运行 DPO 只需 4 小时。

作者认为 PPO 还可以通过调参优化性能，但由于资源和时间限制没做。

### RLVR 强化学习训练阶段

最后一个阶段是使用强化学习训练容易验证结果的能力，这些任务包括：

- 数学问题，包括 GSM8K 及 MATH 的训练数据。
- 指令遵循问题，来自 SFT 数据，验证返回结果是否符合格式。

这些问题可以通过是否正确来判断，如果正确就返回 10，错误返回 0，训练算法使用 DPO，作者做了许多实验，得到以下经验：

- RLVR 可以提升目标领域的分数。
- 价值模型使用奖励模型初始化的效果好于使用 SFT 模型初始化，这个也是通常的做法。
- 使用奖励模型计算奖励的效果还不如使用正确性判断，因此训练时去掉了奖励模型。
- 使用 RLVR 训练 SFT 模型和 DPO 模型都能得到相似结果，但使用 DPO 模型在测试集上的表现更好。

整体来说 Tulu 3 的强化学习训练比较务实，只用可验证的问题进行训练，这个阶段和 DeepSeek-R1-Zero 的做法是类似的，只是用的基础模型和算法不同。

## 自然语言生成 SQL CodeS 模型

自然语言查询数据可以让不会 SQL 的用户也能通过查询数据库，这是一个很实用的功能，因此是大模型研究的热门领域之一。

本节将介绍 CodeS [^liCodeSBuildingOpensource2024]，它通过微调开源模型来实现自然语言生成 SQL，其中许多做法值得参考，而且微调了 1B 到 15B 大小的模型，可以用来比较不同模型的效果。

[^liCodeSBuildingOpensource2024]: <http://arxiv.org/abs/2402.16347>

CodeS 使用 StarCoder 作为基础模型，考虑到在这个模型的训练数据中 SQL 占比很小，作者对它进行了继续预训练，数据由三部分组成：

- 11G SQL 数据（来自 StarCoder）。
- 6G 自然语言到代码的数据（来自 CoNaLa、CodeAlpaca-20k 等），作者还从开源代码中使用正则提取出 SELECT 语句，抽取了 45 万个 SQL 语句，然后使用 GPT-3.5 来生成对应的自然语言，期望通过这些数据让模型更好了解自然语言和 SQL 的关系。
- 4.5G 自然语言对话的数据（来自 Alpaca、Unnatural-instructions 和 UltraChat）。

接着使用这些文本数据依次对 StarCoder 的 1B、3B、7B 和 15B 模型进行了继续预训练，其中 SQL 数据训练了两轮，另外两个数据训练一轮。

增量训练使用全参数微调，使用 DeepSpeed Zero-3 来训练，1B、3B、7B 和 15B 模型的训练时间是 1.5、3、8 和 16 天（论文中没有说明，但测试用的是 8 卡 NVIDIA A800，训练可能也是用这个）。

微调训练主要使用 Spider，使用 GPT-3.5 做了数据增强，主要是几步：

- 收集一些真实用户问题，并手工编写这些问题对应的 SQL。
- 根据表结构、示例数据和现有问题，让 GPT-3.5 生成新的问题。
- 让 GPT-3.5 根据这个问题生成 SQL。

同时还使用模板的方式来合成问题，比如下面问题以及对应的 SQL：

```markdown
## 问题模板

Return the lowest {COLUMN} of {TABLE}

## 对应的 SQL 模板

SELECT {COLUMN} FROM {TABLE} GROUP BY {COLUMN} ORDER BY COUNT (\*) ASC LIMIT 1
```

使用这个模板就能基于现有表结构批量替换来生成问题和 SQL，作者还使用 GPT-3.5 扩充了类似的模板对。

由于数据库中通常包含大量表和字段，容易超出提示词上下文长度限制，因此使用了模式过滤器来选择最相关的 N 个表和列，同时还对数据库里的值进行检索，方法是使用 Lucene 进行索引，然后根据问题找出数据库里最相关的值。

最终在提示词由如下几部分组成：

- 用户问题。
- 使用 RESDSQL [^liRESDSQLDecouplingSchema2023] 的方法训练了个分类器，通过它根据用户问题找出最相关的 N 个表和列，并将这些列的类型、字段名、注释、值信息提取出来，还有表的主键和外键。
- 根据用户问题找出数据库里最相关的值。

[^liRESDSQLDecouplingSchema2023]: <http://arxiv.org/abs/2302.05965>

下面是一个示例：

```txt
Database prompt:
table movie , columns = [ movie.mid ( int | primary key | comment : movie id | values : 101 , 102 ) , movie.title ( text | values : Gone with the Wind , Star Wars )]
table reviewer , columns = [ reviewer.rid ( int | primary key | comment : reviewer id | values : 201 , 202 ) , reviewer.name ( text | values : Sarah Martinez , Daniel Lewis )]
table rating , columns = [ rating.rid ( int | comment : reviewer id | values : 201 , 202 ) , rating.mid ( int | comment : movie id | values : 101 ,106 ) , rating.stars ( int | comment : rating stars | values : 2 , 4 )]

foreign keys :
rating.rid = reviewer.rid
rating.mid = movie.mid

matched values :
reviewer.name ( Sarah Martinez )

Question:
What are the names of all directors whose movies have been reviewed by Sarah Martinez?
```

其中 `Database prompt` 部分是筛选出来的表及字段信息，每个字段取其中两个值作为示例，`foreign keys` 是外键信息，`matched values` 是根据问题找出相关的值，`Question` 是用户问题。

最终效果如下，首先是不微调直接测试继续预训练模型 Few-Shot 的效果，其中 EX（execution accuracy）是指执行结果相同的正确率，但结果相同可能只是碰巧，TS（test-suite accuracy）通过测试多个数据库来避免这种情况，但 BIRD 数据集不支持 TS，所以还是使用 EX 作为测试结果。

| 模型              | Spider TS 准确率 | BIRD EX 准确率 |
| ----------------- | ---------------- | -------------- |
| StarCoderBase-1B  | 48.6             | 22.69          |
| StarCoderBase-3B  | 60.8             | 36.31          |
| StarCoderBase-7B  | 64.6             | 40.61          |
| StarCoderBase-15B | 70.0             | 41.20          |
| Llama2-13B        | 47.6             | 25.36          |
| CodeS-1B          | 59.1             | 31.03          |
| CodeS-3B          | 69.7             | 41.85          |
| CodeS-7B          | 71.8             | 44.26          |
| CodeS-15B         | 73.4             | 45.44          |

从这个测试结果可以得到以下几个结论：

- StarCoder 相对 Llama 2 在生成 SQL 方面更有优势，1B 模型就能达到 13B 的效果，这也是 CodeS 选择基于它的原因。
- 经过继续预训练后的 CodeS 模型分数有提升，尤其是 1B 和 3B 的小模型提升明显。

接着是微调后的效果：

| 模型             | Spider TS 准确率 | BIRD EX 准确率 |
| ---------------- | ---------------- | -------------- |
| SQL-PaLM         | 82.8             | 78.2           |
| DAIL-SQL (GPT-4) | 83.6             | 76.2           |
| SFT Llama2-7B    | 77.8             | 73.0           |
| SFT Llama2-13B   | 81.6             | 76.6           |
| SFT CodeS-1B     | 77.9             | 72.2           |
| SFT CodeS-3B     | 83.4             | 78.1           |
| SFT CodeS-7B     | 85.4             | 80.3           |
| SFT CodeS-15B    | 84.9             | 79.4           |

其中 SQL-PaLM [^sunSQLPaLMImprovedLarge2023] 是 Google 基于 PaLM 微调的模型，它的参数量有 54B，DAIL-SQL (GPT-4) 是基于 GPT-4 使用多种工程方法优化后的方案，SFT Llama2-7B 是使用 Llama 2 微调后的模型。

[^sunSQLPaLMImprovedLarge2023]: <http://arxiv.org/abs/2306.00739>

通过这个微调结果可以得到以下结论：

- 微调后的 1B 模型就能达到 GPT-4 简单使用 Few-Shot 的效果。
- 微调后的 7B 模型超过了 GPT-4 提示词优化的效果。
- 3B 模型的效果不太差，部署性价比最高，反而是 15B 模型还不如 7B 模型。

总结一下，这个模型微调实践有许多可以借鉴的经验：

- 做继续预训练对效果有帮助，尤其是参数量小的模型。
- 要善于利用 GPT-3.5 来扩充数据，包括问题及模板。
- 模型参数量大效果不一定好，比如 PaLM 参数量有 54B，效果还不如作者微调的 3B 模型，所以最好测试多种参数量的效果。

## 自然语言生成 SQL 的难点细节

前面介绍了 CodeS 的微调实践，但它主要是针对 Spider 数据集的实践，这个数据集相对简单，在真实业务场景下还会面临许多其它问题。本节将结合笔者的经验进一步展开讨论，通过这个例子来让读者了解大模型应用落地的实际困难，对其它垂直领域也有借鉴意义。

难点一：表名和字段名通常是英文。

最著名的自然语言生成 SQL 训练数据是 Spider [^yuSpiderLargeScaleHumanLabeled2019]，目前在这个训练数据下最高准确率可以达到 91.2%，但这个测试集是英文，英文的优势是用户问的名词和 SQL 字段几乎一样，比如用户问题是「List the name of clubs」，对应的 SQL 就是「SELECT name FROM club」。

[^yuSpiderLargeScaleHumanLabeled2019]: <http://arxiv.org/abs/1809.08887>

但我们要做的是中文查询 SQL，在数据库中极少有人用中文字段名，前面的例子就变成了「列出俱乐部名称」，这时大模型不仅要正确切分句子，还要根据中文找英文。

这导致在中文场景下准确率会明显下降，比如 Spider 对应的中文测试集 CSpider [^minPilotStudyChinese2019] 最高准确率只有 62%。

[^minPilotStudyChinese2019]: <http://arxiv.org/abs/1909.13293>

要解决这个问题必须给大模型提供中英文对照信息，比如我们可以在上下文中用注释告诉大模型字段中文是什么，类似下面的例子：

```sql
CREATE table "club" (
  "name" VARCHAR(255) COMMENT "名称"
)
```

训练数据里多些这样的例子就能让大模型学会中英文对照。

难点二：类型字段通常使用数字而不是中文。

数据库中还会出现枚举类型的字段，比如支付方式，在数据库里可能存的是 0 和 1，对应微信和支付宝，但用户问的时候肯定是中文，比如「查询微信支付的比例」，这时就要转成对应的数值。

同样可以通过注释的方式告诉大模型，比如下面的例子：

```sql
CREATE TABLE "order" (
  "pay_type" int COMMENT "支付方式：0 代表微信、1 代表支付宝",
)
```

要做到这点一方面需要在训练数据里增加类似例子，另一方面在对接实际表结构时，需要让用户能补充字段值对应的中文信息。

难点三：需要动态关联值。

前面提到枚举类型字段是固定的，可以通过注释可以告诉大模型，但有时值可能是动态的，比如「查询我上个月的销量」，这里的「我」是谁？或者「查询部门上个月的销量」，这里的部门也需要关联到对应的人。

这个问题无法直接通过训练解决，因为训练时数据都是固定的，要解决这个问题需要从两方面解决：

- 训练模型在遇到这种情况是输出特殊值，比如 `<current_user>`。
- 在查询引擎上做特殊处理，对这些特殊值进行自动转换，比如查询 `user_id = '<current_user>'` 要转成查当前登录用户的 id。

难点四：需要猜测字段。

前面提到的例子用户问题中明确提到了字段，但实际场景下用户很可能会省略字段，比如「查询苹果手机的总销量」，而不是「查询**商品名称**中包含苹果手机的总销量」，忽略了商品名称这个字段名，这时需要根据值推测查询哪个字段，并且判断是用等于查询还是模糊查询，由于在表结构定义里并没有值信息，导致难以正确判断。

要解决这个问题可以在提示词里加几个表数据示例，训练大模型推测字段的能力，但这类数据不好准备。

难点五：上下文窗口问题。

类似 Spider 的测试集通常表结构都比较简单，只有个位数的表，每个表的字段数量也不超过十个，但在真实场景下许多系统超过 100 张表，比如微软某个内部财务数据中心有 632 张表，200 个视图，总共超过 7400 列 [^floratouNL2SQLSolvedProblem2024]，如果都放在上下文窗口里很容易就超过大模型上下文窗口限制。

[^floratouNL2SQLSolvedProblem2024]: <https://www.cidrdb.org/cidr2024/papers/p74-floratou.pdf>

这里我们做个简单分析，比如类似下面的表结构：

```sql
-- 博客表
CREATE TABLE "blog" (
  "id" int,
  "title" text COMMENT "标题",
  "body" text COMMENT "内容",
  "created_at" datetime COMMENT "创建时间",
  "author_id" int COMMENT "作者id"
);
```

用 LLaMA 3 算一下需要 51 个 token，实际表字段至少是这个是两倍，也就是 102 个 token，如果有 50 个表就需要 5k token，再加上示例很容易就超过 8K 的窗口大小限制。而我们还需要给输出留出至少 1K token。

这个问题是不是随着大模型升级能解决了呢？最近两年大模型上下文窗口得到了显著提升，但内容变多后准确率也下降了，比如 LLaMA 3.1 虽然支持 128K 窗口，但有人测试过它在 4K 内下的准确率是 96.5，到了 64K 就降低为 88.4，到了 128K 就降低为 66.6 [^hsiehRULERWhatsReal2024]。

那是否可以通过向量数据库来减少窗口大小呢？虽然看起来可以，但用于训练句子嵌入模型的文本对和表结构描述差异较大，向量搜索准确率可能不高。

难点六：大模型时间观差。

大模型训练时通常没有包括时间，这也是大模型幻觉的根源之一，训练的时候并不知道某个文本到底是什么时间的，因为知识重在变化，将这些知识混在一起训练会导致大模型无法区分哪些是过时的信息。

你可以试试问大模型上个月的这周一是几年几月几号，这种问题需要精确计算，大模型并不擅长。

而在数据查询中我们经常需要查询「上个月的销量是多少」，这种任务要如何训练呢？可以尝试的方法是在训练提示词里写出今天的日期，比如「今天是 2024 年 6 月 28 日」，然后期望大模型能自动找出规律，生成类似 `DATE_FORMAT(date, '%Y-%m') = '2024-07'` 的过滤条件。

但这样做有三个问题：

- 训练效果不好保证，需要构造大量时间示例来避免大模型只是记住特例而不是找到规律。
- 有些问题需要真正懂时间计算，难以训练，比如「上周」并不能像「上个月」那样能通过简单数学计算得出。
- 时间函数 SQL 有方言问题，前面的例子是 MySQL 下的写法，如果是 ClickHouse 要用 `formatDateTime` 函数，如果是 DB2 要用 `VARCHAR_FORMAT` 函数、Oracle 和 Postgres 要用 `to_char` 函数、SQLServer 要用 `DATEPART` 和 `CONCAT` 函数组合才能实现，如果要支持多种数据库，难道要所有方言都训一遍？这将导致训练数据成倍膨胀，也导致大模型更容易幻觉。

如何解决？推荐的办法是直接在引擎内部解决，大模型只需要生成 `date = '上个月'`，当引擎发现左侧条件是个日期类型字段，就会转成对应方言的日期函数调用，避免了大模型难以计算时间和方言问题，但缺点是必须完整枚举所有可能的时间特殊值。

除了前面提到的相对时间，引擎还可以实现自动识别常见日期格式，比如数据库里存的是日期时间字段，在查询 `date = '2024'` 时，查询引擎会自动转成 `YEAR(date) = '2024'`，同样也简化了模型训练。

难点七：数据库方言问题。

前面提到了方言问题，除了时间函数之外还有其它方言问题，比如 MySQL 的字段要用反引号包裹，以及 LIMIT 写法常见的就有 10 种，还有大量你可能不知道的语法限制，比如 Oracle 不支持 `GROUP BY 1` 这种写法，你必须将这个 1 展开为 SELECT 里对应的字段。

这个问题要怎么解决？有两个方案：

- 训练时上下文增加数据库信息，并训练多种方言的写法，但这样做一方面会大量增加训练数据，另一方面容易导致模型幻觉，使用错误的方言。
- 利用查询引擎解决，比如统一使用 LIMIT 语法，针对不同方言生成对应的写法，比如使用 Trino 查询引擎。

难点八：关联关系可能有多种情况。

虽然 SQL 数据库都叫关系型数据库，但在表结构中并没有关系配置，而是靠查询的时候动态关联关系，比如要多对多查询就要增加一个中间表，然后 JOIN 两个表。

由于表结构中没有这样的信息，导致只能猜测关系，比如有作者 `author` 和地址 `address` 两个表，它们之间如果是一对一关系，有时我们会在作者表中加个 `address_id` 字段，也可以在 `address` 表中加个 `author_id` 字段，看起来很容易判断？然而在真实场景下通常不会那么规范，比如作者表中的地址字段可能叫 `addr_id`，地址表中的作者字段可能又叫 `owner_id`，难以进行简单匹配。

这种信息缺失将降低大模型准确率，因此我们需要补上这个信息，增加类似 ORM 框架里的关系定义，在注释中提供这些信息：

```java
CREATE TABLE "author" (
  "id" int,
  "address_id" text COMMENT "one-to-one: address.id",
);
```

然后训练大模型如何根据这些关系来生成代码。

难点九：单个 SQL 不能解决所有问题。

目前所有自然语言查询 SQL 的训练都是文本和 SQL 一对一匹配，但这并不能解决所有问题，比如「查询班级和对应的学生，只返回 10 个班级」，这种情况需要两次或者 N+1 次查询。

另外有些数据库不支持窗口函数，这时要实现相同功能就要用代码来实现组内配置，因此有些问题难以通过单个 SQL 解决。

这个问题怎么解决？单靠大模型训练很难解决。

难点十：专业知识。

比如查询「蛋白水平高于正常的患者」，这里的「正常」就是一种垂直领域的专业知识，大模型必须有这方面的专业知识才能正确回答。

难点十一：安全性问题。

如果生成 SQL 后直接调用还会带来安全性问题，包括：

- SQL 安全，用户可能直接问一句「删掉所有数据」。
- 访问权限，数据访问需要权限控制，比如只能查询自己部门的数据。
- 数据隐私，前面提到为了提升效果我们需要将几条真实数据放在提示词中，如果使用大模型 API 还可能导致数据泄露。

解决前两个问题需要对生成的 SQL 做解析和分析，解析具体查询了哪个表及字段，这样才能做行列级别权限控制，避免越权。

最后一个问题可以尝试动态脱敏，或私有化部署大模型。

难点十二：自然语言的歧义。

除了前面提到的大模型幻觉问题，还有个更难的问题是自然语言本身的歧义特性，比如「查询用户名为空」的数据，这个空到底是指 IS NULL、还是空字符串、还是用户名是真的叫「空」？类似的歧义问题很常见，甚至有分析称通过人工标注发现 KaggleDBQA 中 41% 是有歧义的 [^floratouNL2SQLSolvedProblem2024]。

怎么解决？无解，只要是基于自然语言的技术都不可能解决，因为语言就是无法准确表示，这个道理两千多年前的思想家就意识到了：

> 道可道，非常道；名可名，非常名。-《道德经》
>
> 佛说某某，即非某某，是名某某。-《金刚经》

这是大模型应用落地面临的最大问题，只能通过产品机制缓解，比如输出界面让用户确认是否符合预期。

## 其它微调实践中的数据构造方法

从前面分析的经验可以看出，构造训练数据是微调实践中最重要的工作，因此本节将介绍许多实际应用中的数据构造方法。

### 淘宝用户检索问题改写 BEQUE

BEQUE 是淘宝在搜索领域的大模型应用，主要用于改写长尾用户问题。该系统已经在淘宝上线，改写了 0.34% 的用户问题，整体 GMV 提升了 0.4%。

模型的主要功能是将不常见的用户问题改成更常见的商品名称，从而提升检索效果。例如，用户输入「自建盲盒」，搜索引擎无法找到相关产品，但如果改写成「DIY 盲盒」，就能找到。

开发者通过以下三种方式构建数据：

- **上一代系统里的改写数据**：之前开发过类似功能的系统，因此使用了之前较好的改写示例作为初始训练数据。
- **从日志中挖掘**：抽取了 2000 万条日志，包括三方面内容：
  - 人工标记某个搜索的改写是否合适，作为质量判断任务。
  - 根据用户点击来判断，如果用户点击了某个商品，就认为用户输入的问题与该商品相关性强，作为标题预测任务。
  - 基于大模型用 CoT 来解释查询重写的思考过程。
- **人工重写**：共生成了 15 万条数据。

### 教育大模型 Taoli

数据构造基于 500 余册国际中文教育教材与教辅书、汉语水平考试试题以及汉语学习者词典等，构建了 8.8 万条教育领域的问答数据。这些数据包括：

- 语法改错数据集 YACLC。
- 释义数据，基于字典生成，例如使用字典中的例句，然后反问「“因”在此上下文中的具体含义是什么？」。
- 简化复杂句子数据集 MCTS。
- 汉语国际教育动态语料库。

### 医疗问答大模型 Meerkat

Meerkat 的训练数据由三部分组成：

- 在 MedQA 数据集基础上，使用 GPT-4 补充增加 CoT 推理过程，这个数据集称为 MedQA-Cot。
- 基于 18 本医学教科书生成 78K 数据，这个数据集称为 MedBooks-Cot-18。
- 其他公开数据集，包括 MedMCQA、LiveQA、MedicationQA、AlpaCare 等。

### 金融模型 CFGPT

CFGPT 是专门针对中文金融领域的大模型，能够用于金融场景的问答。它收集了大量预训练和指令数据。

预训练数据集包括 5.91 亿份文档和 1930 亿个 token，分为六个子数据集：

- CFData-CP（6.24%）：包括 3900 份公司招股说明书，共计 130 亿个 token。
- CFData-CA（12.28%）：包括 600 万份公司公告，共计 170 亿个 token。
- CFData-RR（2.51%）：包括 39.2 万份研究报告，共计 30 亿个 token。
- CFData-FN（18.70%）：包括 8200 万份财经新闻，共计 260 亿个 token。
- CFData-SM（60.15%）：包括 4.95 亿份社交媒体内容，共计 840 亿个 token。
- CFData-Wiki（0.09%）：包括 25.5 万份维基百科内容，共计 1.37 亿个 token。

其中 CFData-CP、CFData-CA 和 CFData-RR 主要是 PDF 格式，使用 PDFMiner 转换成文本格式，并使用正则去掉文档中的所有图形、表格和乱码，过滤掉小于 1000 字符的文档。其余三个数据源为网页数据，使用 HTML 解析器提取。由于这部分数据质量较差，因此进行了敏感词过滤，敏感词来自 PanGu-$alpha$，而维基百科还将繁体中文转成简体，并对这些文档使用 LSH 算法进行去重。

SFT 指令有 150 万条，通过 6 种不同方式构建：

- CFData-SA（5.69%）：12 万个，是情感分析任务，使用了两种方法构建：
  - 使用 GPT-4 标注社交媒体中的文本内容为正面、负面或中立。
  - 使用 CFData-RR 研究报告中的内容和投资评级，将这些评级转换成正面、负面和中立。
- CFData-RS（50.60%）：36.9 万个，基于 CFData-RR 研究报告中的正文和摘要，让模型根据正文生成摘要。
- CFData-ED（22.69%）：49 万个，是金融事件分类任务，来自 CN-Fin 数据集。
- CFData-TD（12.37%）：36.9 万个，基于 CFData-RR 研究报告中的主题，也是一种分类任务。
- CFData-QA（0.39%）：1.2 万个，金融领域的问答，这部分训练数据是基于英文问答数据 FinQA 和 ConvFinQA 翻译成中文。
- CFData-SP（8.27%）：21.2 万个，股票价格预测任务，使用社交媒体和新闻来预测次日的股票价格是上涨、下跌还是持平。

CFGPT 的做法是将文本数据转成不同指令任务，以摘要为例，对应的提示词如下：

```txt
Please summarize the following financial report. The report is “[report]”.
```

通过类似这样的方式构建了大量训练指令，类似于前面介绍过的[从文本中构造问答]。
