Unsloth是一个专注于加速大语言模型微调过程的开源项目。它通过一系列底层优化，显著提升了微调速度并大幅降低了内存消耗，同时能保持模型性能。无论是研究者还是开发者，都能借助Unsloth更高效地定制自己的大语言模型。本文将介绍Unsloth的使用，相关学习资源如下：

* 开源仓库：[Unsloth](https://github.com)
* 官方文档：[Unsloth Docs](https://github.com)

目录

* [1 Unsloth框架介绍](https://github.com)
  + [1.1 Unsloth概览](https://github.com)
  + [1.2 微调技术概览](https://github.com)
  + [1.3 Unsloth安装](https://github.com)
* [2 Unsloth微调教程](https://github.com)
  + [2.1 模型与训练方法选择](https://github.com)
  + [2.2 LoRA和数据集](https://github.com)
    - [2.2.1 LoRA介绍](https://github.com)
    - [2.2.2 避免过拟合和欠拟合](https://github.com)
    - [2.2.3 训练数据集介绍](https://github.com)
  + [2.3 Qwen3使用示例](https://github.com)
  + [2.4 Unsloth训练Qwen3教程](https://github.com)
    - [2.4.1 预训练模型初始化](https://github.com)
    - [2.4.2 数据集加载](https://github.com)
    - [2.4.3 模型训练](https://github.com)
    - [2.4.4 模型推理](https://github.com)
    - [2.4.5 模型保存](https://github.com)
* [3 参考](https://github.com)

# 1 Unsloth框架介绍

## 1.1 Unsloth概览

Unsloth是一款专为大语言模型微调与强化学习设计的开源框架，致力于以更高的效率和更低的资源成本推动人工智能技术的普及。用户可在本地环境、Google Colab、Kaggle等平台上，借助其运算加速与显存优化能力，轻松完成Qwen、DeepSeek等主流大模型的训练、评估、保存及推理优化。

传统大语言模型微调往往面临硬件要求高、迭代速度慢和资源受限等挑战，而Unsloth通过高效的底层实现和友好的接口设计，显著降低了微调的技术门槛，使更多人能够高效、低成本地训练属于自己的定制模型。

![https://www.codemajin.net/fine-tuning-llm-with-unsloth/](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img1.jpg)

**核心优势**

| 特点 | 说明 | 适用场景/用户 |
| --- | --- | --- |
| 🚀极致速度 | 相比Hugging Face，Unsloth训练模型更快 | 需快速实验与迭代的研发场景 |
| 💾省内存 | 减少GPU显存占用 | 注重成本控制的用户 |
| ✅无损精度 | 无需依赖近似计算 | 对精度要求极高的任务 |
| 🔗广泛兼容 | 支持主流Transformer类模型（涵盖多模态、语音、文本及扩散模型）；支持全量微调、预训练及4/8/16位精度训练；兼容Linux、Windows及主流云平台 | 使用多种架构的团队 |
| 🧩易于使用 | 提供简洁API，兼容Hugging Face生态，可导出GGUF、Ollama等格式 | 初学者、资源有限的小型团队 |
| ⚡高效推理 | 支持INT4量化（QLoRA），推理阶段同步提速 | 需兼顾微调与推理效率的应用场景 |
| 💡低成本 | 单张GPU（如4090或8GB显存卡）即可微调10B+参数模型 | 个人开发者 |
| 🔧高效计算 | 基于Triton（OpenAI开源的高性能GPU编程语言）实现高效计算 | 对技术底层效率有要求的开发团队 |

目前，Unsloth支持借助Accelerate、DeepSpeed等库实现多GPU训练，但实际配置过程较复杂，需手动完成设置，相关训练教程可参考：[Multi-GPU-Unsloth](https://github.com):[闪电加速器](https://shuiyuetian.com)，Unsloth团队正积极优化多GPU训练功能。

**使用建议**

Unsloth与Meta、Google、Microsoft、Mistral、Qwen等主流模型团队深度合作，持续修复关键漏洞，提升框架的准确性与稳定性。该框架支持用户灵活调整聊天模板和数据集格式，并提供涵盖视觉模型、TTS、BERT、强化学习等多样化示例Notebook，助力用户快速上手，详情可参考：[Unsloth Notebooks](https://github.com)。快速上手建议：

* 从QLoRA起步：4-bit量化是资源有限用户的理想选择；
* 调整关键参数：如LoRA秩（`r`）和`alpha`，建议从小值（如16）开始尝试，以平衡模型能力与过拟合风险；
* 监控训练过程：密切关注损失曲线，借助Unsloth的快速迭代优势积极调参；
* 利用社区资源：通过Discord聊天社区等渠道获取帮助、交流经验。

## 1.2 微调技术概览

**什么是微调？**

微调（Fine-tuning）是一种基于预训练大语言模型、利用特定领域数据进一步训练的技术，其核心目标提升模型在特定场景下的性能表现。该技术主要包括两个层面：一是对预训练模型进行持续的无监督预训练；二是指令微调（SFT），即引导模型学习如何根据指令调用已有知识，完成特定格式的任务或匹配特定风格。通过微调，通用大模型能够逐步转化为专业化的领域专家。与检索增强生成（RAG）不同，微调将知识直接内化至模型参数中，实现更深层次的能力融合。本文聚焦于大语言模型的指令微调。

那么，为什么要进行微调？

1. 知识增强：向模型注入领域新知识，扩展其认知边界
2. 行为定制：调整模型的输出风格、语气及响应方式
3. 性能优化：提升模型在特定任务上的准确性、相关性和可靠性

利用Unsloth实现完整指令微调训练的教程见： [How To Fine-tune & Run LLMs](https://github.com)。

**微调常见问题**

* 微调能否增加新知识？
  可以。只要训练数据中包含新信息，模型就能有效学习并掌握新的知识或模式。
* RAG是否一定优于微调？
  并非如此。经过良好优化的微调模型在特定任务上可以媲美甚至超越RAG系统。借助如Unsloth等高效训练工具，微调的技术门槛也显著降低。
* 微调成本是否很高？
  并非必然。采用LoRA/QLoRA等参数高效微调方法，结合免费或低成本的算力资源，完全能够实现低成本甚至零成本的微调。
* 微调如何与其他技术结合？
  微调与RAG具有互补优势：微调赋予模型领域基础能力，RAG则提供实时外部知识，兼顾专业性与时效性。此外，强化学习（RL）也可在微调后通过奖励机制进一步优化模型表现。

![https://medium.com/decodingml/8b-parameters-1-gpu-no-problems-the-ultimate-llm-fine-tuning-pipeline-f68ef6c359c2](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img2.jpg)

## 1.3 Unsloth安装

**Unsloth安装命令**

Unsloth可直接在Linux、Windows、Google Colab等系统上运行，直接安装命令如下：

> pip install unsloth

**系统要求**

* 操作系统：支持Linux与Windows
* 显卡：
  + 兼容2018年及之后发布的NVIDIA显卡
  + 需至少支持CUDA 7.0，例如V100、T4、Titan V、RTX 20/30/40系列、A100、H100、L40等
  + GTX 1070/1080可运行，但性能较慢
  + 支持AMD与Intel的CPU，Apple Silicon版本目前仍在开发中
* 软件兼容：安装Unsloth时将自动更新已有环境中的torch、transformers等库至最新版本，无需手动处理版本冲突
* 依赖项：需安装xformers、torch、BitsandBytes及triton

**微调显存要求**

在使用Unsloth对大语言模型进行微调时，出现内存不足错误通常是由于批处理大小设置过高。将批处理大小调整为1、2或3可有效降低显存占用。

下表列出了不同参数规模与微调方法下的显存需求，其中QLoRA使用4位精度，LoRA使用16位精度。所列数据为理论最低值，部分模型实际可能需要更多显存。详见：[Unsloth-requirements](https://github.com)。

| 模型参数 | QLoRA（4位）显存 | LoRA（16位）显存 |
| --- | --- | --- |
| 3B | 3.5GB | 8GB |
| 7B | 5GB | 19GB |
| 8B | 6GB | 22GB |
| 9B | 6.5GB | 24GB |
| 11B | 7.5GB | 29GB |
| 14B | 8.5GB | 33GB |
| 27B | 22GB | 64GB |
| 32B | 26GB | 76GB |
| 40B | 30GB | 96GB |
| 70B | 41GB | 164GB |
| 81B | 48GB | 192GB |
| 90B | 53GB | 212GB |
| 405B | 237GB | 950GB |

# 2 Unsloth微调教程

## 2.1 模型与训练方法选择

**优先选择指令模型**

大语言模型主要分为基座模型（Base）和指令模型（Instruct）两类，两者均基于文本预测任务进行训练。基座模型通常仅经过预训练和少量通用指令微调；指令模型则在基座模型基础上，进一步通过大规模指令微调和人类反馈强化学习优化其理解和生成能力。常提到的对话模型（Chat Model）本质上属于指令模型。

选择基座模型还是指令模型，通常取决于数据规模、质量与类型：

* 1000行以上数据：数据量较大时，微调基座模型效果更佳。
* 300–1000行高质量数据：中等规模高质量数据下，微调基座模型或指令模型均可。
* 300行以下数据：数据量较小时，建议选择指令模型。微调后既能适配特定任务，又可保留其内置的指令遵循能力，无需额外提示即可响应一般指令（除非需大幅改变模型行为）。

推荐优先从指令模型入手，原因包括：

* 支持直接使用ChatML、ShareGPT等对话模板进行微调，所需数据量更少；
* 基座模型需依赖Alpaca、Vicuna等特定模板，对数据量要求相对更高。

![https://medium.com/data-science-in-your-pocket/unsloth-the-fastest-way-to-fine-tune-llms-041bb6a785ac](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img3.jpg)

**Unsloth模型格式**

在Hugging Face中Unsloth仓库不同后缀代表模型的量化格式或优化版本，选择时可参考以下说明：

* 名称以`unsloth-bnb-4bit` 结尾：为Unsloth动态4位量化模型。其显存占用略高于标准位量化模型，但精度显著更高。
* 名称仅以`bnb-4bit` 结尾（不含`unsloth`）：为标准位量化模型。
* 无后缀：为原始16位或8位格式。这类模型是官方发布的原始版本，但Unsloth会在部分版本中加入对话模板、分词器等重要修复。

在此基础上，在准备微调时，首要决策之一就是选择合适的模型：

1. 选择与用例匹配的模型
   例如：若进行基于图像的训练，可选择Llama 3.2 Vision等视觉模型；针对代码数据集，则适合选用Qwen Coder 2.5等专用模型。
2. 留意授权与要求
   不同模型可能有特定的授权条款和系统要求，务必仔细查看。
3. 评估存储、计算能力和数据集
   可参考Unsloth的显存指南，确定目标模型所需的显存配置。数据集的类型会影响模型的选择，同时也会决定训练所需的时间。
4. 选定模型及参数
   建议选用最新模型，以获得最佳性能和功能。可以通过浏览Unsloth的模型目录，及时了解最新且相关的选项。

可以将模型名称修改为任意名称，只需使其与Hugging Face上Unsloth仓库的模型名称相匹配即可，例如 “unsloth/llama-3.1-8b-unsloth-bnb-4bit”。对于初学者，建议从诸如`unsloth/llama-3.1-8b-unsloth-bnb-4bit`之类的小型指令模型入手，再逐步探索更多可能性。

所有Unsloth支持的模型见：[Unsloth Models](https://github.com)。

**训练方法的选择：LoRA与QLoRA**

在实施微调时，降低计算与内存需求的主流技术主要有以下两种：

* LoRA（低秩适配）：仅微调少量16位的适配器权重矩阵，保持原始模型参数基本不变，从而显著减少训练过程中需要更新的参数量。
* QLoRA（量化LoRA）：在LoRA基础上引入模型权重的4位量化，可在有限硬件资源下高效微调超大规模模型，通过4位精度显著降低内存占用与计算开销。

建议从QLoRA入手，它是当前高效且易于使用的微调方法之一。借助如Unsloth所采用的动态4位量化技术，其精度损失相较于标准的16位LoRA微调已几乎可忽略不计。

![https://towardsdatascience.com/fine-tune-llama-3-1-ultra-efficiently-with-unsloth-7196c7165bab/](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img4.jpg)

已微调的模型可再次多次微调，但最佳做法是合并所有数据集一次性完成。若基于已微调模型续训，可能改变其此前获得的质量与知识。需注意，实验验证至关重要。微调无唯一最佳方法，仅有适配不同场景的最佳实践，需尝试多种方法与配置，才能找到最契合自身数据集及需求的方案。

## 2.2 LoRA和数据集

### 2.2.1 LoRA介绍

LoRA提供了众多超参数（如学习率、训练轮次等），其组合可能达数百万种。合理选择参数对微调至关重要，直接影响模型的准确性、稳定性与输出质量。Unsloth基于数百篇研究论文与实验经验，总结了这些参数的最佳实践，并解析了它们对模型行为的影响。虽然建议直接使用其默认配置，但理解这些概念将有助于更全面地掌控整个微调过程。

超参数调整的目标是在提升模型准确率的同时避免过拟合或欠拟合。对于大型语言模型（如Llama 70B），其权重包含数百亿参数，通常不会全部参与更新，而是采用LoRA等参数高效微调方法。LoRA在每一层旁引入两个小型矩阵A和B，仅优化这两个矩阵，实际训练参数量通常仅占总量的1%左右。通过冻结原始权重、仅更新新增的适配器参数，LoRA显著降低了计算与存储开销，同时在多数任务中保持模型性能，已成为当前大模型微调的主流方法之一。关于LoRA的详细介绍见：[LoRA Hyperparameters Guide](https://github.com)。

以下简要介绍相关参数：

**学习率**

定义模型训练中每一步的权重更新幅度。

* 较高学习率：收敛快，但过高易造成训练震荡，可能错过最优解。
* 较低学习率：训练更稳定、精度高，但收敛慢、耗时长；虽常被认为易欠拟合，实际也可能引发过拟合或阻碍有效学习。
* 常用范围：2e-4（0.0002）至 5e-6（0.000005）
  + LoRA/QLoRA微调：建议初始值2e-4
  + 强化学习：推荐5e-6
  + 全量微调：通常适用更低学习率

**训练次数（Epochs）**

指模型完整遍历训练数据集的次数。

* 轮次过多：可能提升训练集上的表现，但也容易导致过拟合，降低模型泛化能力。
* 轮次过少：训练时间短且不易过拟合，但若模型未能充分学习数据规律，可能造成欠拟合。
* 建议：多数指令微调任务建议训练1–3轮。超过3轮后收益递减，过拟合风险显著增加。

**超参数设置**

其他常用参数如下：

| Hyperparameter | 功能说明 | 推荐值 |
| --- | --- | --- |
| Rank(r) | 控制可训练参数数量，秩越高能力越强，内存占用越大 | 8,16,32,64,128（常用16或32） |
| LoRA Alpha(lora\_alpha) | 用于控制低秩矩阵的缩放系数 | 通常设为r或2r |
| LoRA Dropout(lora\_dropout) | 训练时随机丢弃部分激活值，防止过拟合 | 0（默认0.1） |
| Target Modules(target\_modules) | 指定添加LoRA的模型模块 | q\_proj,k\_proj,v\_proj,o\_proj,gate\_proj,up\_proj,down\_proj（推荐全部） |
| Weight Decay | 抑制权重过大，提升泛化能力 | 0.01至0.1 |
| Warmup Steps | 训练初期逐步提高学习率，稳定训练 | 总步数的5%–10% |
| Scheduler Type | 训练过程中调整学习率的方式 | linear或cosine |
| Seed | 固定随机数种子，保证结果可复现 | 任意整数（如42） |

关于LoRA超参数详细介绍可见：[LoRA、QLoRA、QA-LoRA 原理笔记](https://github.com)。

**作用模块**

在QLoRA与LoRA的对比中，QLoRA采用4-bit精度，可降低超过75%的显存占用，而LoRA（16-bit）在精度和速度上略优。根据论文及实验经验，建议将LoRA同时作用于注意力层与MLP层（如`target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`），以有效提升模型精度。

下图对比了不同目标模块配置下LoRA与QLoRA的Rouge分数（分数越高越好），前三组分别为：

* QLoRA-All：将LoRA应用于所有FFN/MLP层和注意力层，是本实验中表现最佳的配置。
* QLoRA-FFN：仅在FFN层（包括gate\_proj, up\_proj, down\_proj）上应用LoRA。
* QLoRA-Attention：仅在注意力层（包括q\_proj, k\_proj, v\_proj, o\_proj）上应用LoRA。

![https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img5.jpg)

**梯度累积与批次大小的等效关系**

较大的有效批次通常能稳定训练，而较小的批次可能因梯度方差增大而影响收敛。有效批次大小由以下两个参数共同决定：

> 有效批次大小 = batch\_size × gradient\_accumulation\_steps

以下为Unsloth推荐配置，适用于多数微调场景：

| 参数 | 定义 | 影响 | 推荐值 |
| --- | --- | --- | --- |
| batch\_size | 单次前向或反向传播中各GPU处理的样本数 | 主要影响内存占用 | 2 |
| gradient\_accumulation\_steps | 权重更新前累积梯度的步数 | 模拟更大批次以节省显存；步数增加会延长每轮训练时间 | 8 |
| 有效批次大小 | 实际用于梯度更新的样本总数 | 影响训练稳定性与性能 | 16（2×8） |

![https://huggingface.co/docs/trl/main/distributing_training](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img6.jpg)

### 2.2.2 避免过拟合和欠拟合

**过拟合**

深度学习模型容易过度记忆训练数据和噪声，导致泛化能力下降。当训练损失低于0.2时，常提示过拟合，模型在未知任务上表现变差。

一种简单的缓解方法是LoRA Alpha缩放：将每个LoRA矩阵的alpha值乘以0.5。其原理类似于权重平均，将基础模型与LoRA权重相加后除以2，等同于alpha减半。该方法通过平均化机制抑制过拟合，提升模型在未知任务上的泛化性能。

其他常用解决方案包括：

* 调整学习率：过高易引发过拟合，训练周期短时尤需注意；周期较长可适当提高。建议尝试不同取值以寻优。
* 控制训练轮数：通常1–3轮即可，避免过度训练。
* 增大权重衰减（weight\_decay）：初始建议设为0.01或0.1。
* 启用LoRA Dropout：可设为0.1以提高泛化能力。
* 增大批次大小或梯度累积步数：有助于提升训练稳定性。
* 扩展数据集：结合高质量开源数据与自有数据，扩大样本规模。
* 早停机制：验证损失连续多轮上升时自动停止训练。
* 权重平均：将原始模型与微调后的模型权重相加取平均，平滑输出表现。

**欠拟合（过于泛化）**

指模型未能充分学习训练数据中的特征，通常因模型复杂度过低或训练不足导致。改进方法包括：

* 调整学习率：初期可适当提高以加速收敛，长期训练则需降低，需实验确定最优值。
* 增加训练轮次：延长训练时间，同时监控验证集损失以防过拟合。
* 提高LoRA秩与alpha值：秩建议不低于alpha，模型越小或数据越复杂，秩应越大，通常设为4至64。
* 使用领域相关数据集：确保训练数据质量高且与目标任务相关。
* 将批大小设为1：增强每次参数更新的强度，提高模型对数据的敏感度。

### 2.2.3 训练数据集介绍

构建大语言模型训练数据集的关键环节之一是设计恰当的对话模板，以利于模型高效处理。关于数据集的详细介绍见：[Unsloth Datasets Guide](https://github.com)。

**数据格式要求**

为进行分词处理，数据集需采用可被分词器读取的格式。请注意，每种数据类型对应不同的格式样式。

| 格式类型 | 说明 | 训练类型 |
| --- | --- | --- |
| 原始语料 | 来自网站、书籍或文章等的原始文本 | 持续预训练（CPT） |
| 指令文本 | 包含指令及对应输出的示例 | 监督微调（SFT） |
| 对话记录 | 用户与AI助手之间的多轮对话 | 监督微调（SFT） |
| 强化学习数据 | 用户与AI助手的对话，助手回复带有人工/模型/脚本的排序评分 | 强化学习（RL） |

**格式化数据**

在明确数据筛选标准并完成收集后，需将数据转换为机器可读的格式，以适应不同阶段的模型训练需求。以下从四种核心训练场景出发，分别介绍对应的主流数据格式及示例：

1. 预训练数据格式

在模型的继续预训练阶段，通常无需对文本结构做特殊设计，直接采用原始文本即可。这种无结构化的输入方式有助于模型从连续文本中自然学习语言规律与常识知识。

```
"text": "北京烤鸭是中国著名的京菜代表，其制作需经过烫皮、挂色、风干、烤制等多道工序，成品鸭皮酥脆..."
```

2. 指令微调格式

为让模型适应特定任务（如问答、总结、创作），可采用Alpaca风格的指令格式。该格式包含指令（任务目标），输入（任务素材），输出（预期结果）三部分，结构清晰，便于标注。

```
{
  "Instruction": "为以下城市写一句旅游宣传语",
  "Input": "西安（关键词：兵马俑、古城墙、大唐不夜城）",
  "Output": "穿越秦唐，梦回长安——西安等你来探秘"
}
```

3. 多轮对话格式

针对多轮对话场景（如客服、聊天助手），需保留上下文逻辑，常用ShareGPT格式。通过from字段标注角色（human为用户，gpt为模型），value记录发言内容，清晰呈现对话流程。

```
{
  "conversations": [
    {"from": "human", "value": "推荐一道适合初学者的家常菜"},
    {"from": "gpt", "value": "番茄炒蛋简单易学，需要我介绍具体步骤吗？"},
    {"from": "human", "value": "好的，请说明关键步骤和注意事项"},
    {"from": "gpt", "value": "步骤：1. 番茄切块，鸡蛋打散；2. 热油炒蛋后盛出；3. 炒番茄至出汁，加糖调味；4. 混入鸡蛋翻炒。注意火候，避免蛋炒老。"}
  ]
}
```

4. ChatML格式

ChatML格式由OpenAI提出，是当前工业界广泛使用的对话格式，也被Hugging Face等平台默认支持。它通过role字段定义角色（如user，assistant，system），用content记录内容，结构清晰且兼容性强。

```
{
  "messages": [
    {"role": "system", "content": "你是一位中文烹饪助手，回答需简明实用"},
    {"role": "user", "content": "蒸鱼应该用大火还是小火？"},
    {"role": "assistant", "content": "建议大火蒸制，时间约8–10分钟，这样鱼肉更鲜嫩。"}
  ]
}
```

**合成数据生成**

为获得理想的微调效果，建议数据集不少于100条；若追求更优性能，推荐使用1000条以上的数据。通常情况下，数据量越大，效果越好。若原始数据不足，可引入合成数据或补充Hugging Face上的相关数据集以增强多样性。请注意，微调效果高度依赖数据质量，务必做好数据清洗和预处理。

生成合成数据时，可使用本地大语言模型（如Llama 3.3 70B）或OpenAI的GPT-4.5。通常更推荐使用参数规模更大的模型以保证生成质量。通过vLLM、Ollama或 llama.cpp等推理引擎可直接生成数据，但需手动收集生成结果，并优化提示词以扩展内容。合成数据的主要用途包括：

1. 创造全新数据：既可完全从头生成，也可基于现有样本进行改写或扩展；
2. 增强数据多样性：避免模型过拟合，提升泛化能力；
3. 完善现有数据：例如将文本自动转换为指定格式（如将对话转为问答形式）。

## 2.3 Qwen3使用示例

本文将以Qwen3为例进行模型训练演示。Qwen3由阿里通义千问推出，在推理、指令遵循及多语言支持等核心能力上实现行业领先，是大语言模型训练的优选架构。

Unsloth已于2025年7月完成升级，支持最新的Qwen-2507模型。在使用Unsloth运行或微调量化版Qwen模型时，几乎无损精度。同时，Unsloth为Qwen3原生支持128K上下文长度，可一次性处理数万字的长文档或对话；该扩展基于YaRN技术，将模型原有的40K处理上限提升至128K。优化后，模型训练速度提升2倍，显存占用降低70%。

![https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune/qwen3-2507](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img6_2.jpg)

**模型版本**

为帮助开发者根据模型运行、长上下文支持、微调与部署等场景需求，选择合适规模的Qwen3模型，Unsloth基于其技术能力，围绕以下三个维度提供了多种参数规格的版本：

1. Dynamic 2.0 GGUF（适用于模型运行）
   涵盖0.6B至235B-A22B等多种参数规模，支持用户直接运行Qwen3模型，适用于常规推理与基础任务场景。
2. 128K Context GGUF（适用于长上下文处理）
   提供4B到235B-A22B等多个版本，重点优化了128K上下文长度的处理能力，适用于长文档分析、超长对话及对语义连贯性要求较高的复杂任务。
3. Dynamic 4-bit Safetensor（适用于微调与部署）
   覆盖0.6B至32B参数规模，采用4位量化的Safetensor格式，在保持模型性能的同时显著降低存储与计算资源开销，便于进行任务特定微调或生产环境部署。

**推理参数**

为达到每秒6个token以上的推理速度，Unsloth建议总内存（即显存、内存或两者总和）不低于所使用模型的大小。即使总内存低于模型大小，仍可运行模型，但推理速度会降低。根据Qwen官方建议，模型推理的推荐设置如下：

| 参数 | 非思考模式（Non-ThinkingMode） | 思考模式（ThinkingMode） | 解释 |
| --- | --- | --- | --- |
| 温度（Temperature） | 0.7 | 0.6 | 值越低输出越确定 |
| 最小概率（Min\_P） | 0.0（可选，0.01效果更佳） | 0.0 | 仅考虑累积概率达到该值的候选词 |
| 累积概率（Top\_P） | 0.8 | 0.95 | 从累积概率前百分之几的候选词中选取 |
| 候选词数量（TopK） | 20 | 20 | 每次只从概率最高的K个词中选择 |

**Qwen3对话模板**

Qwen3系列模型采用ChatML对话模板，默认启用思考模式。请注意，若使用贪婪解码，可能导致模型性能下降或生成内容无限重复。基础对话格式如下：

```
<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n
```

如需关闭思考模式，需插入一对空的与标签，格式如下：

```
<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n\n\n\n\n
```

**示例推理代码**

下面代码展示通过Unsloth的FastModel类加载Qwen3-0.6B模型并启用4位量化：

```
from modelscope import snapshot_download
from unsloth import FastModel

# 定义要使用的模型名称，这里使用的是Qwen3-0.6B模型
model_name = "Qwen/Qwen3-0.6B"
# 利用modelscope加速下载模型
model_dir = snapshot_download(model_name)

model, tokenizer = FastModel.from_pretrained(
    model_name = model_dir,  # 指定模型所在的目录路径
    max_seq_length = 2048,   # 设置最大序列长度为2048，可以根据需要调整以支持长文本
    load_in_4bit = True,     # 启用4位量化以减少内存占用
    load_in_8bit = False,    # 禁用8位量化（新特性：8位量化精度稍高，但内存占用是4位的2倍）
    full_finetuning = False, # 禁用全参数微调（新特性：现在支持全参数微调）
)

# 准备模型输入
prompt = "推荐一部搞笑的科幻电影。"
# 构造对话消息列表，包含用户角色和内容
messages = [
    {"role": "user", "content": prompt}
]
# 应用聊天模板处理消息，转换为模型所需的输入格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 不直接进行token化
    add_generation_prompt=True,  # 添加生成提示
    enable_thinking=True  # 启用思考模式，默认为True
)
# 将文本转换为模型输入张量，并移动到模型所在设备
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 进行文本生成，输出为token
generated_ids = model.generate(
    **model_inputs,  # 解包模型输入
    max_new_tokens=2048  # 最大生成的新token数量
)
# 提取生成的部分（排除输入部分）
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# 解析思考内容
try:
    # 查找特殊标记151668（表示思考内容结束）的位置
    index = len(output_ids) - output_ids[::-1].index(151668)
    # 这个结束符就是
    # tokenizer.decode(output_ids[index-1])
except ValueError:
    # 如果未找到特殊标记，索引设为0
    index = 0

# 解码思考内容（特殊标记之前的部分）
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# 解码回复内容（特殊标记之后的部分）
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# 打印思考内容和最终回复内容
print("思考内容:", thinking_content)
print("回复内容:", content)
```

## 2.4 Unsloth训练Qwen3教程

Qwen3能够同时进行数学推理和常识问答。但如果只用“天空是什么颜色？蓝色”这类常识样本训练，模型在微调后可能出现能力退化，甚至无法正确解答“1+2×3=？”这类简单题目。
为保持模型的推理能力，建议在训练素材中混合使用推理类和非推理类样本。例如，可组合75%的思维链样本。如“1+2×3：先算乘法2×3=6，再加1得7”，以及25%的常识类样本，如直接提供答案的问题。这样模型既能正确回答常识问题，也能维持数学推理能力，实现两类任务的平衡。

![https://medium.com/data-and-beyond/a-practical-guide-to-fine-tune-mistral-7b-with-unsloth-for-phishing-email-detection-2faa5b531e27](https://gitlab.com/luohenyueji/article_picture_warehouse/-/raw/main/Python-Study-Notes/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A05-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6Unsloth%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97/img/img7.jpg)

下面将依次介绍如何使用Unsloth加载Qwen3模型，并详细讲解数据预处理、模型训练、模型运行及模型保存的完整流程。

### 2.4.1 预训练模型初始化

以下代码演示了如何利用Unsloth库加载Qwen3-0.6B模型，通过4位精度量化大幅减少内存使用，并借助LoRA方法实现高效的参数微调。

```
# 从modelscope库导入snapshot_download函数，用于下载模型快照
from modelscope import snapshot_download
# 从unsloth库导入FastLanguageModel类，用于高效加载语言模型
from unsloth import FastLanguageModel

# 定义要使用的模型名称，这里使用的是Qwen3-0.6B模型
model_name = "Qwen/Qwen3-0.6B"
# 利用modelscope的snapshot_download函数加速下载模型，并返回模型保存的目录路径
model_dir = snapshot_download(model_name)

# 使用FastLanguageModel的from_pretrained方法加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_dir,       # 模型所在的目录路径
    max_seq_length = 2048,        # 上下文长度 - 可以设置更长，但会占用更多内存
    load_in_4bit = True,          # 以4位精度加载，使用更少内存
    load_in_8bit = False,         # 以8位精度加载会更准确，但占用2倍内存
    full_finetuning = False,      # 是否使用全量微调，当前设置为否
)

# 为模型配置LoRA方法
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # LoRA注意力维度，可选择任何大于0的值，建议8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", # 注意力模型和FFN模块
                      "gate_proj", "up_proj", "down_proj",],  # 指定要微调的模块
    lora_alpha = 32,  # LoRA缩放参数，建议设置为与rank相同或rank的2倍
    lora_dropout = 0, # LoRA 层的 dropout 率（这里设为 0 以优化性能）
    bias = "none",    # 是否训练偏置（这里设为 "none" 表示不训练）
    # [新特性] "unsloth"模式可减少30%的VRAM使用，支持2倍大的批量大小
    use_gradient_checkpointing = "unsloth",  # True或"unsloth"用于超长上下文
    random_state = 0,  # 随机种子，确保结果可复现
    use_rslora = False,   # 是否使用RSLoRA
    loftq_config = None,  # 是否使用 LoftQ
)
```

### 2.4.2 数据集加载

Qwen3包含推理和非推理两种模式，本示例使用以下两个训练数据集：

1. 推理数据：Open Math Reasoning（开放数学推理）数据集
   从中采样了10%的可验证推理轨迹，这些样本使用了DeepSeek R1，且准确率超过95%。
   从这些数据里，Unsloth仅筛出DeepSeek-R1回答、正确率≥95%且每一步都可验证的标准答案，再从中随机抽取10%使用。

   * 用处：专注于数学推理能力的微调数据集，包含各种数学问题及其详细解答过程。
   * 来源：[unsloth/OpenMathReasoning-mini](https://github.com)
   * 样本数量：19,252
   * 格式：包含数学问题、期望答案、问题类型、解答过程等字段
   * 特点：增强模型的数学推理和思维链（Chain-of-Thought）能力
2. 通用对话数据：Maxime Labonne的FineTome-100k数据集
   其格式为ShareGPT风格，已转换为Hugging Face标准的多轮对话格式。

   * 用处：高质量的指令遵循数据集，专为大语言模型微调设计
   * 来源：[mlabonne/FineTome-100k](https://github.com)
   * 样本数量：100,000
   * 格式：包含对话内容、来源和质量分数
   * 特点：数据质量高，覆盖广泛的指令类型和领域

数据处理代码如下：

```
from datasets import load_dataset
# 数据集下载链接
# reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
# non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

# 如果无法直接访问Hugging Face，可以使用以下两个命令从镜像网站下载数据集到本地（速度也很慢）
# git clone https://hf-mirror.com/datasets/unsloth/OpenMathReasoning-mini
# git clone https://hf-mirror.com/datasets/mlabonne/FineTome-100k
# 从本地加载数据集
reasoning_dataset = load_dataset("./OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("./FineTome-100k", split = "train")

# 查看数据集结构
# 特征包含预期答案、题目类型、题目来源、生成模型，72B TIR模式下的通过率、题目本身、解答过程、推理模式
print(reasoning_dataset)
# 特征包含对话、来源、分数
print(non_reasoning_dataset)

# 将reasoning_dataset转换为对话格式
def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    
    # 初始化一个空列表，用于存储转换后的对话
    conversations = []
    
    # 同时遍历问题和解决方案列表，将它们配对成对话
    for problem, solution in zip(problems, solutions):
        # 为每对问题和解决方案创建一个对话结构
        # 每个对话包含两个角色的消息：用户（提问）和助手（回答）
        conversations.append([
            {"role" : "user",      "content" : problem},      # 用户角色的消息内容是问题
            {"role" : "assistant", "content" : solution},     # 助手角色的消息内容是解决方案
        ])
    
    return { "conversations": conversations, }

# 使用tokenizer将推理数据集转换为模型可理解的对话模板格式
# 参数tokenize=False表示只进行格式转换，不进行分词处理
reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_dataset.map(generate_conversation, batched = True)["conversations"],
    tokenize = False,
)
print(reasoning_conversations[0])

# 接下来，处理处理非推理型数据集，并同样将其转换为对话格式。
# 使用standardize_sharegpt函数，对该数据集的格式进行规范化处理。
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(non_reasoning_dataset)

non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)
print(non_reasoning_conversations[0])

# 查看数据集尺寸
print(len(reasoning_conversations))
print(len(non_reasoning_conversations))

# 非推理类数据集规模大的多。希望模型保留一定推理能力，训练数据选取75%推理类数据搭配25%对话类数据
chat_percentage = 0.25

import pandas as pd
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
    random_state = 0,  
)

# 打印各类数据量及实际比例用于验证
print(len(reasoning_conversations))  
print(len(non_reasoning_subset))   
print(len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations)))

# 合并推理类数据和抽样后的非推理类数据
data = pd.concat([
    pd.Series(reasoning_conversations),  
    pd.Series(non_reasoning_subset)      
])
data.name = "text"  # 为合并后的数据系列命名

from datasets import Dataset
# 将pandas DataFrame转换为Hugging Face数据集格式
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
# 对数据集进行随机打乱，确保数据分布均匀
combined_dataset = combined_dataset.shuffle(seed = 0)
```

### 2.4.3 模型训练

为加快训练速度，训练仅迭代30步。若需完整训练，可将num\_train\_epochs设为1，并将max\_steps设为None以取消步数限制。

```
# trl库是Hugging Face开发的，用于通过强化学习来微调与对齐大型语言模型的工具
# SFTTrainer用于监督微调训练，SFTConfig用于配置训练参数
from trl import SFTTrainer, SFTConfig
import torch
# 初始化SFTTrainer训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,  # 训练数据集
    eval_dataset=None,  # 评估数据集
    args=SFTConfig(
        dataset_text_field="text",  # 数据集中用于训练的文本字段名称
        per_device_train_batch_size=2,  # 每个设备的训练批次大小
        # 梯度累积步数，通过累积梯度来模拟更大的批次大小
        # 实际等效批次大小 = per_device_train_batch_size * gradient_accumulation_steps
        gradient_accumulation_steps=4,
        warmup_steps=5,  # 学习率预热步数，逐步增加到设定的学习率
        # num_train_epochs = 1,  # 训练轮数，注释掉表示不使用轮数限制
        max_steps=30,  # 最大训练步数，达到后停止训练
        learning_rate=2e-4,  # 学习率，长时间训练建议降低到2e-5
        logging_steps=1,  # 每多少步记录一次日志
        optim="adamw_8bit",  # 使用8位AdamW优化器，节省内存
        weight_decay=0.01,  # 权重衰减系数，用于防止过拟合
        lr_scheduler_type="linear",  # 学习率调度器类型，此处为线性衰减
        seed=0,
        report_to="none",  # 日志报告工具
    ),
)

# 获取编号为0的GPU设备属性信息，包括名称、总内存等
gpu_stats = torch.cuda.get_device_properties(0)
# 计算当前程序已保留的最大GPU内存，也就是PyTorch的CUDA分配器最高向操作系统申请了多少内存
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# 计算GPU的总内存容量
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# 打印GPU名称和总内存信息
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# 打印当前已保留的GPU内存信息
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
# resume_from_checkpoint是否从之前保存的检查点恢复训练
trainer_stats = trainer.train(resume_from_checkpoint=False)

# 计算GPU的最大预留内存
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# 计算LoRA训练额外占用的GPU内存
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# 计算峰值内存占GPU总内存的百分比
used_percentage = round(used_memory / max_memory * 100, 3)
# 计算LoRA训练占用内存占GPU总内存的百分比，保留3位小数
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
# 打印训练总耗时
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# 打印GPU峰值预留内存
print(f"Peak reserved memory = {used_memory} GB.")
# 打印训练过程中额外占用的GPU峰值内存
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# 打印GPU峰值内存占总内存的百分比
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# 打印LoRA额外占用内存占GPU总内存的百分比
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
```

### 2.4.4 模型推理

推理阶段，据Qwen-3团队的建议：

* 若用于普通对话任务，推荐参数设置为：temperature=0.7，top\_p=0.8，top\_k=20。
* 若用于推理任务，推荐参数设置为：temperature=0.6，top\_p=0.95，top\_k=20。

以下代码对比这两种生成模式。前者直接给结果；后者先思考解题步骤再给结果，更适合需解释过程的数学问题场景。

```
# 定义对话消息列表，包含用户的问题
# 这里问题是求解方程 (x + 3)^2 = 0
messages = [{"role": "user", "content": "Solve (x + 3)^2 = 0."}]

# 使用tokenizer的聊天模板处理消息
# 将消息转换为模型可以理解的格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

# 导入TextStreamer，用于实时流式输出模型生成的内容
from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,  # 最大生成的新token数量，控制回答长度
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    streamer=TextStreamer(
        tokenizer, skip_prompt=True
    ),  # 流式输出器，跳过提示部分只显示回答
)

# 再次定义相同的用户问题，用于演示思考模式
messages = [{"role": "user", "content": "Solve (x + 3)^2 = 0."}]

# 使用聊天模板处理消息，这次启用思考模式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # 启用思考模式，模型会先思考再给出答案
)

# 在思考模式下生成回答
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=1024,  # 思考模式需要更多token来容纳思考过程
    temperature=0.6,  # 稍低的温度，使思考过程更集中
    top_p=0.95,  # 更高的核采样参数，允许更多样化的思考
    top_k=20,  # 同样从概率最高的20个token中选择
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

### 2.4.5 模型保存

Unsloth支持两条互补的持久化路线：

1. 保留LoRA适配器：体积最小，便于继续微调或增量更新；
2. 合并并量化导出：得到独立权重文件，方便直接部署或上传到 Hub。

**保存LoRA适配器**

训练完成后，只需把模型与分词器以`save_pretrained`写入同一目录即可：

```
model.save_pretrained("lora_model")      # 仅保存 LoRA 参数
tokenizer.save_pretrained("lora_model")
```

后续加载时，用`from_pretrained`接口指定本地路径，Unsloth会自动把基础模型与LoRA权重重新组装：

```
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",   # 加载lora参数，同时加载训练时的基础模型
    max_seq_length=2048,
    load_in_4bit=True,
)
```

**合并后导出，用于部署**

可将LoRA合并到基础模型中，并支持将合并后的模型一次性导出为float16、int4或GGUF（GPT-Generated Unified Format）格式，便于在GPU或CPU端侧进行高效推理。

GGUF是一种高效的模型存储与交换格式，将大模型封装为单一文件，具备秒级加载能力。GGUF可直接用于llama.cpp系列工具，实现快速部署与应用。

```
# 导出float16完整权重
model.save_pretrained_merged("model-f16", tokenizer, save_method="merged_16bit")

# 导出int4量化权重，会有精度损失
model.save_pretrained_merged("model-int4", tokenizer, save_method="merged_4bit_forced")

# 导出GGUF系列
model.save_pretrained_gguf("model-q8",  tokenizer)  # 默认 Q8_0
model.save_pretrained_gguf("model-f16", tokenizer,
                           quantization_method="f16")  # 16-bit GGUF
model.save_pretrained_gguf("model-q4",  tokenizer,
                           quantization_method="q4_k_m") # 4-bit GGUF
```

# 3 参考

* [Unsloth](https://github.com)
* [Unsloth Docs](https://github.com)
* [Multi-GPU-Unsloth](https://github.com)
* [How To Fine-tune & Run LLMs](https://github.com)
* [Unsloth-requirements](https://github.com)
* [Unsloth Models](https://github.com)
* [LoRA Hyperparameters Guide](https://github.com)
* [LoRA、QLoRA、QA-LoRA 原理笔记](https://github.com)
* [Unsloth Datasets Guide](https://github.com)
* [Unsloth Notebooks](https://github.com)
