import argparse
import gc
import json
import os
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob

from evaluation.dataset import (
    BaseDataset,
    C4Dataset,
    HumanEvalDataset,
    WMT16DE_ENDataset,
)
from evaluation.pipelines.detection import DetectionPipelineReturnType
from evaluation.pipelines.quality_analysis import QualityPipelineReturnType
from evaluation.pipelines.vllm_detection import WatermarkDetectionVLLMPipeline
from evaluation.pipelines.vllm_quality_analysis import (
    DirectTextQualityAnalysisVLLMPipeline,
    ExternalDiscriminatorTextQualityAnalysisVLLMPipeline,
    ReferencedTextQualityAnalysisVLLMPipeline,
)
from evaluation.tools.success_rate_calculator import (
    DynamicThresholdSuccessRateCalculator,
)
from evaluation.tools.text_quality_analyzer import (
    BERTScoreCalculator,
    BLEUCalculator,
    GPTTextDiscriminator,
    LogDiversityAnalyzer,
    PPLCalculator,
    ROUGE1Calculator,
    ROUGE2Calculator,
    ROUGELCalculator,
)
from utils.transformers_config import TransformersConfig
from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import DiscreteVisualizer
from watermark.auto_watermark import AutoWatermarkForRLLM
from watermark.base import BaseWatermark

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()


# 数据集配置（使用具体的分析器类而非字符串，并添加prompt键）
DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "c4": {
        "path": "dataset/c4/processed_c4.json",
        "class": C4Dataset,
        "metrics": {
            "direct": [PPLCalculator, LogDiversityAnalyzer],
            "referenced": [],
            "external": [],
        },
        "prompt": None,
    },
    "wmt16_de_en": {
        "path": "dataset/wmt16_de_en/validation.jsonl",
        "class": WMT16DE_ENDataset,
        "metrics": {
            "direct": [PPLCalculator],
            "referenced": [
                BLEUCalculator,
                ROUGE1Calculator,
                ROUGE2Calculator,
                ROUGELCalculator,
                BERTScoreCalculator,
            ],
            "external": [GPTTextDiscriminator],
        },
        "prompt": "You are a professional translator. Translate input text into English while preserving all original formatting, style, and special characters. Important: No explanations or comments in your output - just translation!",
    },
    "human_eval": {
        "path": "dataset/human_eval/test.jsonl",
        "class": HumanEvalDataset,
        "metrics": {"direct": [], "referenced": [], "external": []},
        "prompt": None,
    },
}


def prepare_output_dir(
    model_path: str,
    dataset_len: int,
    dataset_name: str,
    dir_name: str = "outputs",
) -> str:
    """准备输出目录并返回路径"""
    output_dir = os.path.join(os.getcwd(), dir_name)
    output_dir = os.path.join(
        output_dir,
        model_path,
        f"{dataset_name}-{dataset_len}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_result_file_paths(
    output_dir: str, algorithm_name: str
) -> dict[str, str]:
    """获取各种结果文件的路径"""
    return {
        "no_watermark_results": os.path.join(
            output_dir, "no_watermark_results.json"
        ),
        "watermark_results": os.path.join(
            output_dir, f"{algorithm_name}_watermark_results.json"
        ),
        "combined_results": os.path.join(
            output_dir, f"{algorithm_name}_combined_results.json"
        ),
        "nowatermark_img": os.path.join(output_dir, "no_watermark.png"),
        "watermark_img": os.path.join(
            output_dir, f"{algorithm_name}_watermark.png"
        ),
    }


def load_previous_no_watermark_results(
    file_paths: dict[str, str],
) -> dict[str, Any] | None:
    """尝试加载之前的无水印生成结果"""
    if os.path.exists(file_paths["no_watermark_results"]):
        print("发现之前的无水印生成结果，正在加载...")
        with open(file_paths["no_watermark_results"], "r") as f:
            results = json.load(f)
        return results
    return None


def save_results(file_path: str, results: dict[str, Any]) -> None:
    """保存结果和相关数据到 JSON 文件"""
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)


def extract_text_from_output(
    outputs: list[RequestOutput],
) -> tuple[list[str], list[str], int]:
    """从输出中提取全文和回答文本，确保无None值"""
    text_full: list[str] = []
    text_filtered: list[str] = []

    think_cnt = 0
    for output in outputs:
        text = output.outputs[0].text
        if "</think>" in text:
            text_full.append(text)
            text_filtered.append(text.split("</think>")[-1].strip())
            think_cnt += 1
        else:
            print(f"没有 </think> 的文本，已跳过: {text}")
            # 不添加到结果中，保证所有列表索引对齐

    return text_full, text_filtered, think_cnt


def process_generation(
    model,
    sampling_params,
    conversations,
    is_watermark: bool = False,
    watermark=None,
) -> tuple[list[RequestOutput], list[str], list[str]]:
    """处理文本生成，职责单一，只负责生成文本并返回结果"""
    # 设置采样参数
    if is_watermark and watermark:
        sampling_params.logits_processors = [watermark]
    elif is_watermark:
        print("警告：启用了水印但未提供水印处理器")

    # 文本类型标识
    text_type = "有水印" if is_watermark else "无水印"

    # 执行模型生成
    outputs = model.chat(
        messages=conversations,
        sampling_params=sampling_params,
        use_tqdm=True,
        add_generation_prompt=True,
    )

    # 提取文本，确保无None值
    text_full, text_filtered, think_cnt = extract_text_from_output(outputs)
    print(
        f"{text_type}情况下，含有 </think> 文本的占比: {think_cnt}/{len(outputs)}"
    )

    return outputs, text_full, text_filtered


def setup_quality_analyzers(
    model_path: str, dataset_metrics: dict, device: str = "cuda"
) -> dict[str, list]:
    """设置文本质量分析器，根据数据集配置实例化对应分析器"""
    analyzers = {"direct": [], "referenced": [], "external": []}

    # 直接文本质量分析器
    if PPLCalculator in dataset_metrics["direct"]:
        ppl_calculator = PPLCalculator(
            model=AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_path),
            device=device,
        )
        analyzers["direct"].append(ppl_calculator)

    if LogDiversityAnalyzer in dataset_metrics["direct"]:
        log_diversity_analyzer = LogDiversityAnalyzer()
        analyzers["direct"].append(log_diversity_analyzer)

    # 参考文本质量分析器
    if BLEUCalculator in dataset_metrics["referenced"]:
        analyzers["referenced"].append(BLEUCalculator())

    if ROUGE1Calculator in dataset_metrics["referenced"]:
        analyzers["referenced"].append(ROUGE1Calculator())

    if ROUGE2Calculator in dataset_metrics["referenced"]:
        analyzers["referenced"].append(ROUGE2Calculator())

    if ROUGELCalculator in dataset_metrics["referenced"]:
        analyzers["referenced"].append(ROUGELCalculator())

    if BERTScoreCalculator in dataset_metrics["referenced"]:
        analyzers["referenced"].append(
            BERTScoreCalculator(model_path="google-bert/bert-base-uncased")
        )

    # GPT判别器将在需要时创建，避免不必要的API调用
    if GPTTextDiscriminator in dataset_metrics["external"]:
        analyzers["external"] = []

    return analyzers


def evaluate_detection(
    watermark: BaseWatermark,
    watermark_texts: list[str],
    nowatermark_texts: list[str],
    dataset: BaseDataset,
    labels: list[str] = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
    rule="target_fpr",
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """处理 ROC 曲线评估，返回评估指标"""
    print("\n正在使用 ROC 曲线计算器评估水印检测效果...")

    if not watermark_texts or not nowatermark_texts:
        print("警告: 没有足够的有效文本进行 ROC 评估")
        return {}

    # 创建评估 pipeline
    pipeline = WatermarkDetectionVLLMPipeline(
        dataset=dataset,
        show_progress=True,
        return_type=DetectionPipelineReturnType.SCORES,
    )

    watermarked_scores: list[float] = pipeline.evaluate(
        watermark, watermark_texts
    )  # type: ignore
    nowatermarked_scores: list[float] = pipeline.evaluate(
        watermark, nowatermark_texts
    )  # type: ignore

    auroc = roc_auc_score(
        [1] * len(watermarked_scores) + [0] * len(nowatermarked_scores),
        watermarked_scores + nowatermarked_scores,
    )

    calculator = DynamicThresholdSuccessRateCalculator(
        labels=labels,
        rule=rule,
        target_fpr=target_fpr,
    )

    # 计算并输出评估指标
    metrics: dict[str, float] = calculator.calculate(
        watermarked_scores, nowatermarked_scores
    )
    metrics["auroc"] = float(auroc)

    print("\nROC 曲线水印检测评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def gpt_evaluation(
    prompts: list[str],
    nowatermark_text: list[str],
    watermark_text: list[str],
    dataset: BaseDataset,
    openai_api_key: str,
    openai_model: str = "text-davinci-003",
) -> dict[str, Any]:
    """使用 GPT 判别器评估水印对文本质量的影响"""
    if not nowatermark_text or not watermark_text:
        print("警告: 没有足够的有效文本进行 GPT 评估")
        return {"error": "缺少有效文本"}

    print("\n正在使用 GPT 文本判别器评估水印对文本质量的影响...")
    # 设置 OpenAI API 密钥
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # 选择部分样本进行 GPT 判别（为了控制 API 成本）
    sample_size = min(10, len(nowatermark_text))
    sample_indices = np.random.choice(
        len(nowatermark_text), sample_size, replace=False
    )

    # 获取样本文本
    sampled_nowatermark = [nowatermark_text[i] for i in sample_indices]
    sampled_watermark = [watermark_text[i] for i in sample_indices]

    # 实例化 GPT 文本判别器
    gpt_discriminator = GPTTextDiscriminator(
        openai_model=openai_model,
        task_description="评估文本的自然度、流畅度和整体质量",
    )

    # 创建外部判别器 pipeline，修改返回类型为 SCORES 而非 FULL
    pipeline = ExternalDiscriminatorTextQualityAnalysisVLLMPipeline(
        dataset=dataset,
        analyzers=[gpt_discriminator],
        show_progress=True,
        return_type=QualityPipelineReturnType.SCORES,
    )

    # 执行评估
    eval_results = pipeline.evaluate(sampled_nowatermark, sampled_watermark)

    # 统计 GPT 判别结果
    gpt_results = []

    # 正确处理评估结果 - 处理 SCORES 返回类型
    if isinstance(eval_results, list):
        for result in eval_results:
            if (
                isinstance(result, dict)
                and "watermarked" in result
                and "unwatermarked" in result
            ):
                watermarked_scores = result["watermarked"]
                if "GPTTextDiscriminator" in watermarked_scores:
                    gpt_results.append(
                        watermarked_scores["GPTTextDiscriminator"]
                    )

    no_watermark_better = gpt_results.count(1)
    watermark_better = gpt_results.count(2)
    equal_quality = gpt_results.count(0)

    print(f"GPT 判别结果 (样本数: {sample_size}):")
    print(
        f"无水印文本更好: {no_watermark_better} ({no_watermark_better/sample_size:.2%})"
    )
    print(
        f"有水印文本更好: {watermark_better} ({watermark_better/sample_size:.2%})"
    )
    print(f"质量相当: {equal_quality} ({equal_quality/sample_size:.2%})")

    # 返回 GPT 判别结果
    return {
        "sample_size": sample_size,
        "no_watermark_better": no_watermark_better,
        "watermark_better": watermark_better,
        "equal_quality": equal_quality,
        "percentages": {
            "no_watermark_better": no_watermark_better / sample_size,
            "watermark_better": watermark_better / sample_size,
            "equal_quality": equal_quality / sample_size,
        },
    }


def visualization(
    img_path: str,
    watermark: Any,  # 改为Any类型，避免类型兼容问题
    text: str,
    force_create: bool = False,
) -> None:
    """创建文本可视化图像并保存到指定路径"""
    if not force_create and os.path.exists(img_path):
        print(f"可视化图片已存在: {img_path}，跳过生成")
        return

    # 创建可视化对象
    visualizer = DiscreteVisualizer(
        color_scheme=ColorSchemeForDiscreteVisualization(),
        font_settings=FontSettings(),
        page_layout_settings=PageLayoutSettings(),
        legend_settings=DiscreteLegendSettings(),
    )

    # 生成可视化图像
    img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=text),
        show_text=True,
        visualize_weight=True,
        display_legend=True,
    )
    img.save(img_path)


def main(args):
    # 获取数据集配置
    if args.dataset not in DATASET_CONFIG:
        print(
            f"错误: 未知的数据集 {args.dataset}，可用选项: {list(DATASET_CONFIG.keys())}"
        )
        return

    dataset_info = DATASET_CONFIG[args.dataset]

    # 准备输出目录
    output_dir: str = prepare_output_dir(
        model_path=args.model_path,
        dataset_len=args.dataset_len,
        dataset_name=args.dataset,
    )
    file_paths: dict[str, str] = get_result_file_paths(
        output_dir, args.algorithm_name
    )

    # 初始化模型
    model = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        dtype=args.dtype,
        disable_custom_all_reduce=False,
        disable_log_stats=False,
        trust_remote_code=True,
        swap_space=32,
        seed=args.seed,
    )

    # 准备采样参数
    sampling_params = model.get_default_sampling_params()
    sampling_params.n = 1  # 生成序列数量始终为 1
    sampling_params.logprobs = 0  # 确保能获取 logprobs
    if args.max_tokens is not None:
        sampling_params.max_tokens = args.max_tokens
    if args.min_tokens is not None:
        sampling_params.min_tokens = args.min_tokens
    if args.presence_penalty is not None:
        sampling_params.presence_penalty = args.presence_penalty
    if args.seed is not None:
        sampling_params.seed = args.seed
    if args.temperature is not None:
        sampling_params.temperature = args.temperature
    if args.min_p is not None:
        sampling_params.min_p = args.min_p
    if args.top_p is not None:
        sampling_params.top_p = args.top_p
    if args.top_k is not None:
        sampling_params.top_k = args.top_k

    # 加载模型配置和 tokenizer
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(args.model_path),
        tokenizer=tokenizer,
        vocab_size=config.vocab_size,
        device="cuda",
        max_new_tokens=args.max_tokens,
        max_length=args.max_model_len,
        do_sample=True,
        no_repeat_ngram_size=4,
    )

    # 初始化水印
    print("初始化水印...")
    watermark = AutoWatermarkForRLLM(
        algorithm_name=args.algorithm_name,
        algorithm_config=f"config/{args.algorithm_name}.json",
        transformers_config=transformers_config,
    )

    # 加载数据集
    print(f"加载数据集 {args.dataset}...")
    dataset_class = dataset_info["class"]
    dataset: BaseDataset = dataset_class(
        dataset_info["path"], max_samples=args.dataset_len
    )

    # 获取提示和参考文本
    prompts: list[str] = dataset.prompts
    references: list[str] | None = None
    if dataset.references:
        references = dataset.references
    elif dataset.natural_texts:
        references = dataset.natural_texts

    # 创建对话格式，考虑是否需要添加提示
    pre_prompt = dataset_info["prompt"]
    conversations = []

    if pre_prompt:
        for prompt in prompts:
            conversation = [
                {"role": "user", "content": pre_prompt},
                {"role": "user", "content": prompt},
            ]
            conversations.append(conversation)
    else:
        # 标准对话格式
        for prompt in prompts:
            conversation = [{"role": "user", "content": prompt}]
            conversations.append(conversation)

    # 设置分析器
    text_quality_analyzers = setup_quality_analyzers(
        args.model_path, dataset_info["metrics"]
    )

    # 首先尝试加载之前的无水印结果
    previous_nowatermark_results = load_previous_no_watermark_results(
        file_paths
    )

    if previous_nowatermark_results:
        print("使用之前的无水印生成结果，跳过重新生成...")
        nowatermark_results = previous_nowatermark_results
        # 提取之前保存的文本
        nowatermark_full_text = nowatermark_results.get("outputs", {}).get(
            "full_text", []
        )
        nowatermark_answer_text = nowatermark_results.get("outputs", {}).get(
            "answer_text", []
        )
    else:
        print("未找到之前的无水印生成结果，开始生成...")
        # 处理无水印生成
        nowatermark_outputs, nowatermark_full_text, nowatermark_answer_text = (
            process_generation(
                model=model,
                sampling_params=sampling_params,
                conversations=conversations,
                is_watermark=False,
            )
        )

        nowatermark_results = {
            "outputs": {
                "full_text": nowatermark_full_text,
                "answer_text": nowatermark_answer_text,
            },
            "quality_metrics": {},
        }

    # 处理有水印生成
    watermark_outputs, watermark_full_text, watermark_answer_text = (
        process_generation(
            model=model,
            sampling_params=sampling_params,
            conversations=conversations,
            is_watermark=True,
            watermark=watermark,
        )
    )

    # 保存有水印结果基础信息
    watermark_results = {
        "outputs": {
            "full_text": watermark_full_text,
            "answer_text": watermark_answer_text,
        },
        "quality_metrics": {},
    }

    # 直接文本质量分析 (PPL, Log Diversity)
    if text_quality_analyzers["direct"]:
        print("\n进行直接文本质量分析...")
        direct_pipeline = DirectTextQualityAnalysisVLLMPipeline(
            dataset=dataset,
            analyzers=text_quality_analyzers["direct"],
            show_progress=True,
            return_type=QualityPipelineReturnType.MEAN_SCORES,
            unwatermarked_text_source="generated",  # 确保正确设置
        )

        metrics = direct_pipeline.evaluate(
            watermark_answer_text, nowatermark_answer_text
        )

        # 保存结果
        if isinstance(metrics, dict):
            if "watermarked" in metrics:
                for metric_name, value in metrics["watermarked"].items():
                    watermark_results["quality_metrics"][
                        metric_name.lower()
                    ] = value
            if "unwatermarked" in metrics:
                for metric_name, value in metrics["unwatermarked"].items():
                    nowatermark_results["quality_metrics"][
                        metric_name.lower()
                    ] = value

    # 参考文本质量分析 (BLEU, ROUGE, BERTScore等)
    if text_quality_analyzers["referenced"] and references:
        print("\n进行参考文本质量分析...")
        # 确保参考文本和生成文本数量相同
        assert (
            len(references)
            == len(watermark_answer_text)
            == len(nowatermark_answer_text)
        )
        ref_pipeline = ReferencedTextQualityAnalysisVLLMPipeline(
            dataset=dataset,
            analyzers=text_quality_analyzers["referenced"],
            show_progress=True,
            return_type=QualityPipelineReturnType.MEAN_SCORES,
            unwatermarked_text_source="natural" if references else "generated",
        )

        ref_metrics = ref_pipeline.evaluate(
            watermark_answer_text, nowatermark_answer_text
        )

        # 保存结果
        if isinstance(ref_metrics, dict):
            if "watermarked" in ref_metrics:
                for metric_name, value in ref_metrics["watermarked"].items():
                    watermark_results["quality_metrics"][
                        metric_name.lower()
                    ] = value
            if "unwatermarked" in ref_metrics:
                for metric_name, value in ref_metrics["unwatermarked"].items():
                    nowatermark_results["quality_metrics"][
                        metric_name.lower()
                    ] = value

    # 打印指标汇总
    print("\n有水印文本指标汇总:")
    for metric_name, value in watermark_results["quality_metrics"].items():
        print(f"{metric_name}: {value:.4f}")

    print("\n无水印文本指标汇总:")
    for metric_name, value in nowatermark_results["quality_metrics"].items():
        print(f"{metric_name}: {value:.4f}")

    # 处理 ROC 评估并添加到结果中
    roc_metrics = evaluate_detection(
        watermark.watermark,
        watermark_answer_text,
        nowatermark_answer_text,
        dataset,
        labels=["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
        rule=args.threshold_rule,
        target_fpr=args.target_fpr,
    )

    # 处理 GPT 评估并添加到结果中
    if (
        GPTTextDiscriminator in dataset_info["metrics"]["external"]
        and args.openai_api_key
    ):
        gpt_metrics = gpt_evaluation(
            prompts,
            nowatermark_answer_text,
            watermark_answer_text,
            dataset,
            args.openai_api_key,
            args.openai_model,
        )

    # 保存结果
    save_results(
        file_paths["no_watermark_results"],
        nowatermark_results,
    )

    save_results(
        file_paths["watermark_results"],
        watermark_results,
    )

    # 保存合并结果
    combined_results: dict[str, dict[str, Any]] = {
        "no_watermark": nowatermark_results,
        "watermark": watermark_results,
        "detection": roc_metrics,
        "gpt_discrimination": gpt_metrics,
    }
    save_results(
        file_paths["combined_results"],
        combined_results,
    )

    # 创建可视化
    if watermark_answer_text and nowatermark_answer_text:
        # 创建无水印文本的可视化
        if not os.path.exists(file_paths["nowatermark_img"]):
            visualization(
                file_paths["nowatermark_img"],
                watermark,
                nowatermark_answer_text[0],
            )

        # 创建有水印文本的可视化
        visualization(
            file_paths["watermark_img"],
            watermark,
            watermark_answer_text[0],
            force_create=True,  # 有水印图片始终重新创建
        )
    else:
        print("警告: 没有足够的有效文本进行可视化")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="模型路径，例如 deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument(
        "--algorithm-name",
        type=str,
        default="KGW",
        choices=["KGW", "UPV", "Unigram"],
        help="算法名称，例如 KGW, UPV, Unigram",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        choices=["c4", "wmt16_de_en", "human_eval"],
        help="评估数据集类型",
    )
    parser.add_argument(
        "--dataset-len",
        type=int,
        default=100,
        help="数据集大小",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="张量数据类型",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="模型最长上下文",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=7800,
        help="最大生成 tokens 数",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=32,
        help="最小生成 tokens 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="采样温度",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="top_k 参数",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="核采样 top_p",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="min_p 参数",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="presence_penalty 参数",
    )
    parser.add_argument(
        "--threshold-rule",
        type=str,
        default="target_fpr",
        choices=["best", "target_fpr"],
        help="动态阈值确定规则: best 或 target_fpr",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="目标假阳性率 (当 threshold-rule 为 target_fpr 时使用)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API 密钥",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-davinci-003",
        help="OpenAI 模型名称，例如 text-davinci-003",
    )
    args = parser.parse_args()
    main(args)
