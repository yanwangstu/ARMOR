# ARMOR: Adaptive Multi-hop Reasoning with Reinforcement Learning

ARMOR 是一个基于强化学习（RL）的多跳推理问答系统，结合了检索增强生成（RAG）技术。该项目包含两个主要阶段：**WarmUp 阶段**（监督微调）和 **RL 阶段**（基于 GRPO 的强化学习）。

## 📁 项目结构

```
Code&Dataset/
├── code/
│   ├── WarmUp/              # WarmUp 阶段（监督微调）
│   │   ├── train.py         # LoRA 微调训练脚本
│   │   ├── data_utils.py    # 数据集加载和处理
│   │   ├── retriever.py     # E5 文本检索器
│   │   ├── inference.py     # RAG 推理引擎
│   │   ├── test.py          # 测试推理脚本
│   │   └── train.sh         # 训练启动脚本
│   │
│   └── RL/                  # RL 阶段（GRPO 强化学习）
│       ├── grpo_trainer.py  # GRPO 训练器主脚本
│       ├── rollout.py       # Rollout 采样模块
│       ├── reward_cal.py    # 奖励计算模块
│       ├── optimizer.py     # GRPO 优化器
│       ├── task_schedular.py# 自适应任务调度器
│       ├── rl_data_utils.py # RL 数据集加载
│       ├── json_dump.py     # JSON 数据转储工具
│       ├── llm_invoke.py    # LLM 调用工具
│       └── train-Llama3.2-3B.sh  # RL 训练启动脚本
│
└── dataset/                 # 数据集
    ├── train_warmup.json    # WarmUp 训练集
    ├── val_sectional.json   # 验证集
    ├── train_rl.json        # RL 训练集
    ├── test_2WikiMultiHopQA_sectional.json
    ├── test_HotpotQA_sectional.json
    └── test_MusiQue_sectional.json
```

## 🚀 核心功能

### 1. WarmUp 阶段（监督微调）

WarmUp 阶段使用 LoRA（Low-Rank Adaptation）对预训练语言模型进行监督微调，使模型学会多跳推理的思维链格式。

#### 主要组件

- **`train.py`**: 主训练脚本
  - 支持 LoRA 微调
  - 自动处理特殊 token（如 `<think>`, `<sub-question>`, `<search>` 等）
  - OOM 安全机制（CUDA 显存不足时自动跳过批次）
  - 分布式训练支持（通过 Accelerate）

- **`data_utils.py`**: 数据处理
  - `MultiHopDataset`: 多跳推理数据集类
  - 特殊 token 定义：`<main-question>`, `<think>`, `<sub-question>`, `<search>`, `<doc>`, `<doc-type>`, `<sub-answer>`, `<main-answer>`
  - 自动掩码文档内容（训练时不对文档内容进行反向传播）

- **`retriever.py`**: 检索模块
  - `e5_Retriever`: 基于 E5 模型的稠密检索器
  - 支持相似度计算和向量检索

- **`inference.py`**: RAG 推理引擎
  - 动态检索：根据子问题实时检索相关文档
  - 特殊 token 控制生成流程
  - 支持噪声文档注入（用于鲁棒性测试）
  - ChromaDB 向量数据库集成

- **`test.py`**: 批量测试脚本
  - 增量保存推理结果
  - 支持多种测试数据集

#### 使用方法

```bash
# WarmUp 训练
cd Code&Dataset/code/WarmUp
bash train.sh

# 或手动运行
python train.py \
    --model_path "Llama-3.2-3B-Instruct" \
    --train_data_path "warmup_dataset/train_warmup_Llama3.2-3B.json" \
    --val_data_path "warmup_dataset/val_warmup_Llama3.2-3B.json" \
    --output_dir "sft_cache/Llama3.2-3B-sft-lora-train-embedding-shot" \
    --prompt_file_path "prompt.txt" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 5

# 推理测试
python test.py \
    --base_model_path "Llama-3.2-3B-Instruct" \
    --lora_adapter_path "sft_cache/Llama3.2-3B-sft-lora-train-embedding-shot/final" \
    --e5_model_path "e5-base-v2" \
    --prompt_file_path "prompt.txt" \
    --test_file_path "../dataset/test_MusiQue_sectional.json" \
    --output_file_path "results/test_results.json" \
    --retriever_topk 3
```

### 2. RL 阶段（GRPO 强化学习）

RL 阶段使用 GRPO（Group Relative Policy Optimization）算法进一步优化模型的推理能力，通过自适应任务调度提升多跳推理性能。

#### 主要组件

- **`grpo_trainer.py`**: GRPO 训练主脚本
  - 分布式训练支持（torchrun + DDP）
  - 三阶段训练流程：Rollout → Reward Calculation → Policy Update
  - 支持全局/局部优势归一化
  - 自动检查点保存

- **`rollout.py`**: Rollout 采样模块
  - 并行采样多个推理轨迹
  - 动态文档检索和注入
  - 思维链节点解析

- **`reward_cal.py`**: 奖励计算模块
  - **QuestionSearch 任务**: 奖励正确的检索决策（是否应该检索）
  - **DocType 任务**: 奖励正确的文档类型判断（useful/useless）
  - **QuestionAnswer 任务**: 奖励正确的答案生成（0-5 分制）

- **`task_schedular.py`**: 自适应任务调度器
  - 基于多臂老虎机（MAB）的任务选择策略
  - 动态调整任务权重以平衡不同任务的训练
  - 探索 - 利用权衡（Exploration-Exploitation Tradeoff）

- **`optimizer.py`**: GRPO 优化器
  - 重要性采样比率计算
  - PPO-style 截断损失
  - KL 散度正则化
  - DAPO（Direct Alignment from Preference Optimization）损失变体

#### 训练任务类型

```python
class TrainTaskType(Enum):
    QuestionSearch = "QuestionSearchVerdict"      # 判断是否需要检索
    DocType = "DocTypeClassify"                   # 判断文档是否有用
    QuestionAnswer = "QuestionAnswerGeneration"   # 生成正确答案
```

#### 使用方法

```bash
# RL 训练（单卡）
cd Code&Dataset/code/RL
bash train-Llama3.2-3B.sh

# 或多卡训练
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 grpo_trainer.py \
    --theme "Llama3.2-3B-RL" \
    --train_dataset_path "train_rl.json" \
    --system_prompt_path "prompt.txt" \
    --policy_model_path "WarmUp/sft_cache/Llama3.2-3B-sft-lora-train-embedding-shot/final-merged" \
    --reward_model_path "Qwen3-4B" \
    --model_save_path "grpo_save/Llama3.2-3B-RL" \
    --rollout_info_save_dir "rollout_info/Llama3.2-3B-RL" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --batch_size 10 \
    --rollout_num 4 \
    --rollout_micro_batch_size_per_gpu 1 \
    --policy_ref_micro_batch_size_per_gpu 1 \
    --max_token_len 1500 \
    --temperature 0.85 \
    --top_p 0.95 \
    --top_k 50 \
    --model_save_steps 30 \
    --rollout_info_save_step 10
```

## 📊 数据格式

### WarmUp 数据格式

```json
{
  "id": "795_2Wiki_train",
  "data_source": ["2WikiMultiHopQA", "train", 795],
  "main_question": "Are director of film My Own United States and director of film Anton (1973 film) from the same country?",
  "main_answer": "no",
  "chain_of_thought": [
    {
      "think": "To determine if the directors are from the same country...",
      "sub_question": "Who is the director of the film Anton (1973 film)?",
      "retrieval": [],
      "doc": "Anton is a 1973 Norwegian drama film...",
      "doc_type": "golden",
      "sub_answer": "The 1973 film Anton was directed by Per Blom.",
      "evidence": ["Anton (1973 film)", "director", "Per Blom"]
    }
  ]
}
```

### RL 数据格式

与 WarmUp 类似，但增加了 `id` 字段用于追踪样本。

## 🔧 依赖环境

```bash
# 基础依赖
pip install torch transformers peft accelerate

# 检索相关
pip install chromadb sentence-transformers

# 其他工具
pip install python-dotenv
```

## 📝 特殊 Token 说明

| Token | 用途 |
|-------|------|
| `<main-question>` | 主问题开始标记 |
| `</main-question>` | 主问题结束标记 |
| `<think>` | 思考过程开始 |
| `</think>` | 思考过程结束 |
| `<sub-question>` | 子问题开始 |
| `</sub-question>` | 子问题结束 |
| `<search>` | 检索决策开始 |
| `</search>` | 检索决策结束 |
| `<doc>` | 文档内容开始 |
| `</doc>` | 文档内容结束 |
| `<doc-type>` | 文档类型（useful/useless） |
| `</doc-type>` | 文档类型结束 |
| `<sub-answer>` | 子答案开始 |
| `</sub-answer>` | 子答案结束 |
| `<main-answer>` | 主答案开始 |
| `</main-answer>` | 主答案结束 |

## 🎯 推理流程示例

```
用户输入：<main-question>谁导演了电影 X？</main-question>

模型输出：
<think>我需要先查找电影 X 的相关信息</think>
<sub-question>电影 X 的导演是谁？</sub-question>
<search>True</search>
<doc>电影 X 是由导演 Y 执导的...</doc>
<doc-type>useful</doc-type>
<sub-answer>电影 X 的导演是 Y</sub-answer>
<main-answer>Y</main-answer>
```

## 📈 训练参数配置

### WarmUp 阶段默认参数

```python
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.1
learning_rate = 2e-4
batch_size = 2
gradient_accumulation_steps = 8
num_train_epochs = 5
```

### RL 阶段默认参数

```python
lora_r = 8
lora_alpha = 16
batch_size = 10
rollout_num = 4
temperature = 0.85
top_p = 0.95
clip_epsilon = 0.2  # PPO 截断阈值
kl_coef = 0.0       # KL 散度系数
```

## 🔍 奖励函数设计

### 1. QuestionSearch 任务
- **不检索且答案正确**: +0.5
- **不检索且答案错误**: -1.0
- **检索且答案正确**: -0.5（惩罚不必要的检索）
- **检索且答案错误**: +1.0（鼓励检索）

### 2. DocType 任务
- **正确分类文档类型**: +1.0
- **错误分类文档类型**: -1.0

### 3. QuestionAnswer 任务
- **答案质量评分 5 分**: +1.0
- **答案质量评分 0-4 分**: 0.0

## 📄 License

本项目代码仅供研究使用。

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [E5 Embedding Model](https://huggingface.co/intfloat/e5-base-v2)
