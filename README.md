## DISTILLING THE THOUGHT, WATERMARKING THE ANSWER: A PRINCIPLE SEMANTIC GUIDED WATERMARK FOR LARGE REASONING MODELS

<div align="center">
Shuliang Liu<sup>1,2</sup>, Xingyu Li<sup>1</sup>, Hongyi Liu<sup>1</sup>, Yibo Yan<sup>1,2</sup>, Bingchen Duan<sup>1,2</sup>, Qi Zheng<sup>1,2</sup>, Dong Fang<sup>3*</sup>, Lingfeng Su<sup>3</sup>, Xuming Hu<sup>1,2*</sup>
</div>

<div align="center">
<sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou) <br/>
<sup>2</sup> The Hong Kong University of Science and Technology <br/>
<sup>3</sup> Independent Researcher
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2601.05144">
    <img src="https://img.shields.io/badge/arXiv-2601.05144-b31b1b.svg?style=flat" />
  </a>
</div>

## Abstract

Reasoning Large Language Models (RLLMs) excelling in complex tasks present unique challenges for digital watermarking, as existing methods often disrupt logical coherence or incur high computational costs. Token-based watermarking techniques can corrupt the reasoning flow by applying pseudo-random biases, while semantic-aware approaches improve quality but introduce significant latency or require auxiliary models. This paper introduces **ReasonMark**, a novel watermarking framework specifically designed for reasoning-intensive LLMs. Our approach decouples generation into an undisturbed **Thinking Phase** and a watermarked **Answering Phase**. We propose a **Criticality Score** to identify semantically pivotal tokens from the reasoning trace, which are distilled into a **Principal Semantic Vector (PSV)**. The PSV then guides a semantically-adaptive mechanism that modulates watermark strength based on token-PSV alignment, ensuring robustness without compromising logical integrity. Extensive experiments show ReasonMark surpasses state-of-the-art methods by reducing text Perplexity by 0.35, increasing translation BLEU score by 0.164, and raising mathematical accuracy by 0.67 points. These advancements are achieved alongside a 0.34% higher watermark detection AUC and stronger robustness to attacks, all with a negligible increase in latency. This work enables the traceable and trustworthy deployment of reasoning LLMs in real-world applications.

<div align="center">
  <img src="assert/ReasonMark-onepage3.pdf" alt="ReasonMark overview" width="800">
</div>

---

## 🏗️ Project Structure

```
MarkLLM-dev/
├── config/                        # 算法配置（含 config/OURS.json）
├── watermark/
│   └── ours/                      # OURS/ReasonMark 实现（watermark/ours/ours.py）
├── scripts/                       # 生成/可视化/质量/检测
│   ├── generate_hf.sh
│   ├── visualize.sh
│   ├── assess_quality.sh
│   └── assess_detectability.sh
├── dataset/                       # 评测数据（c4/gsm8k/wmt/human_eval/...）
├── outputs/                       # 生成与评测输出（自动生成）
├── generate_hf.py                    # 生成（有水印/无水印）主入口
├── assess_detectability.py        # 检测性评估主入口
├── assess_quality.py              # 文本质量评估主入口
└── visualization.py               # 可视化主入口
```

## 🚀 Quick Start

### Prerequisites
- Python **3.10**
- 依赖：`torch`、`transformers`、`vllm`、`datasets` 等（见 `requirements*.txt`）

### Installation

```bash
cd MarkLLM-dev

pip install -r requirements.txt
```

---



## 📊 How to Use (Step-by-step)

### 1) Generate (Watermarked / Unwatermarked)

```bash
# in MarkLLM-dev/

bash scripts/generate_hf.sh \
  --model-path "Qwen/Qwen3-32B" \
  --algorithm-name "OURS" \
  --dataset-name "c4" \
  --dataset-len 200 \
  --watermark-before-think
# 对于 OURS 算法, 需要加上 --watermark-before-think
```

常用参数：

- `--max-model-len`
- `--max-new-tokens` / `--min-new-tokens`
- `--temperature` / `--top-p` / `--top-k` / `--min-p`
- `--watermark-before-think`:在 `</think>` 前应用水印（适配推理模型输出格式）

### 2) Assess Text Quality

```bash
# in MarkLLM-dev/

bash scripts/assess_quality.sh \
  --algorithm "OURS" \
  --model-path "Llama/Meta-Llama-3.1-70B-bnb-4bit" \
  --dataset-name "c4" \
  --dataset-len 200
```

### 3) Assess Detectability

```bash
# in MarkLLM-dev/

bash scripts/assess_detectability.sh \
  --algorithm "OURS" \
  --model-path "Qwen/Qwen3-32B" \
  --dataset-name "c4" \
  --dataset-len 200 
```


## 🧩 Algorithm Configuration (ReasonMark / OURS)

- 配置文件：`config/OURS.json`
- 实现代码：`watermark/ours/ours.py`

---

## 📁 Dataset

数据集与任务配置入口在 `cfg.py`，常见包括：
- `c4`（文本续写）
- `cnn_dailymail`（内容概括）
- `wmt16_de_en` / `wmt19_zh_en`（机器翻译）
- `human_eval`（代码生成）
- `gsm8k` / `mmlu_pro` / `aime_2025`（推理/选择题/数学）

---

## 📄 License

本仓库核心代码（`MarkLLM-dev`）遵循 **Apache-2.0**（见 `MarkLLM-dev/LICENSE`）。

---

## 📖 Citation
```bibtex
@article{liu2026distilling,
  title={Distilling the Thought, Watermarking the Answer: A Principle Semantic Guided Watermark for Large Reasoning Models},
  author={Liu, Shuliang and Li, Xingyu and Liu, Hongyi and Yan, Yibo and Duan, Bingchen and Zheng, Qi and Fang, Dong and Su, Lingfeng and Hu, Xuming},
  journal={arXiv preprint arXiv:2601.05144},
  year={2026}
}
```

## 📧 Contact

- Email: shulianglyo@gmail.com
