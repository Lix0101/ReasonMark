import argparse
import gc
import json
import os
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob

from evaluation.dataset import BaseDataset, C4Dataset
from evaluation.tools.text_quality_analyzer import (
    BERTScoreCalculator,
    BLEUCalculator,
    GPTTextDiscriminator,
    LogDiversityAnalyzer,
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
from watermark.auto_watermark import AutoWatermarkForRLLM, AutoWatermarkForVLLM
from watermark.base import BaseWatermark

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()


# Initialize quality analyzers
bleu_calculator = BLEUCalculator()
rouge1_calculator = ROUGE1Calculator()
rouge2_calculator = ROUGE2Calculator()
rougel_calculator = ROUGELCalculator()
log_diversity_analyzer = LogDiversityAnalyzer()
bert_score_calculator = BERTScoreCalculator(
    model_path="google-bert/bert-base-uncased"
)


def prepare_output_dir(
    model_path: str,
    dataset_len: int,
    dir_name: str = "outputs",
    dataset_name: str = "C4",
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
) -> tuple[list[str], list[str | None], int]:
    """从输出中提取全文和回答文本"""
    text_full: list[str] = [output.outputs[0].text for output in outputs]
    text_filtered: list[str | None] = []

    think_cnt = 0
    for text in text_full:
        if "</think>" in text:
            text_filtered.append(text.split("</think>")[-1].strip())
            think_cnt += 1
        else:
            print(f"没有 </think> 的文本: {text}")
            text_filtered.append(None)

    return text_full, text_filtered, think_cnt


def extract_outputs_data(
    outputs: list[RequestOutput],
    text_full: list[str],
    text_filtered: list[str | None],
) -> dict[str, Any]:
    """从 outputs 提取可序列化的数据"""
    outputs_data: dict[str, Any] = {
        # 保存原始文本
        "text_full": text_full,
        "text_filtered": text_filtered,
        # 从 outputs 提取关键信息
        "logprobs": [],
        "cumulative_logprobs": [],
    }

    for output in outputs:
        try:
            # 提取 logprobs
            if (
                hasattr(output.outputs[0], "logprobs")
                and output.outputs[0].logprobs
            ):
                logprobs_data = []
                for token_logprob in output.outputs[0].logprobs:
                    for token, data in token_logprob.items():
                        logprobs_data.append(
                            {
                                "token": token,
                                "logprob": data.logprob,
                            }
                        )
                outputs_data["logprobs"].append(logprobs_data)
            else:
                outputs_data["logprobs"].append(None)

            # 提取 cumulative_logprob
            outputs_data["cumulative_logprobs"].append(
                output.outputs[0].cumulative_logprob
                if hasattr(output.outputs[0], "cumulative_logprob")
                else None
            )
        except Exception as e:
            print(f"提取输出数据时出错: {e}")
            outputs_data["logprobs"].append(None)
            outputs_data["cumulative_logprobs"].append(None)

    return outputs_data


def calculate_ppl(
    answer_text: str | None,
    tokenizer,
    output_token_logprobs: list[dict[str, Logprob]] | None,
) -> float | None:
    """计算困惑度 (PPL)"""
    if answer_text is None or output_token_logprobs is None:
        return None

    try:
        # 仅对答案部分的 token 计算 PPL
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        answer_token_count = len(answer_tokens)

        # 获取对应的 token logprobs
        token_logprobs = output_token_logprobs[-answer_token_count:]
        answer_logprob = sum(
            next(iter(token_logprob.values())).logprob
            for token_logprob in token_logprobs
        )

        return -answer_logprob / max(1, answer_token_count)
    except Exception as e:
        print(f"计算 PPL 时出错: {e}")
        return None


def calculate_log_diversity(text: str | None) -> float | None:
    """计算对数多样性"""
    if text is None:
        return None

    try:
        return log_diversity_analyzer.analyze(text)
    except Exception as e:
        print(f"计算 log_diversity 时出错: {e}")
        return None


def calculate_reference_metrics(
    generated_text: str | None,
    reference_text: str,
    labels: list[str] = ["bleu", "rouge_1", "rouge_2", "rouge_l", "bert_score"],
) -> dict[str, float | None]:
    """计算文本相似度指标"""
    results: dict[str, float | None] = {}

    if generated_text is None:
        # 如果生成文本为 None，所有指标都设为 None
        for label in labels:
            results[label] = None
        return results

    # 计算各个指标
    calculators: dict[str, Callable[[], float]] = {
        "bleu": lambda: bleu_calculator.analyze(generated_text, reference_text),
        "rouge_1": lambda: rouge1_calculator.analyze(
            generated_text, reference_text
        ),
        "rouge_2": lambda: rouge2_calculator.analyze(
            generated_text, reference_text
        ),
        "rouge_l": lambda: rougel_calculator.analyze(
            generated_text, reference_text
        ),
        "bert_score": lambda: bert_score_calculator.analyze(
            generated_text, reference_text
        ),
    }

    for label in labels:
        if label in calculators:
            try:
                results[label] = calculators[label]()
            except Exception as e:
                print(f"计算 {label} 时出错: {e}")
                results[label] = None
        else:
            print(f"未知的文本相似度指标: {label}")
            results[label] = None

    return results


def calculate_avg_metrics(
    metric_results: dict[str, list[float | None]],
) -> dict[str, float | None]:
    """计算平均指标，忽略 None 值"""
    avg_metrics: dict[str, float | None] = {}

    for metric_name, values in metric_results.items():
        # 过滤掉 None 值
        valid_values = [v for v in values if v is not None]
        if valid_values:
            avg_metrics[f"avg_{metric_name}"] = np.mean(valid_values).item()
        else:
            avg_metrics[f"avg_{metric_name}"] = None

    return avg_metrics


def process_generation(
    model,
    sampling_params,
    conversations,
    tokenizer,
    reference,
    is_watermark: bool = False,
    watermark=None,
) -> tuple[
    list[RequestOutput], list[str], list[str | None], dict[str, dict[str, Any]]
]:
    """处理文本生成和评估"""
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

    # 提取文本
    text_full, text_filtered, think_cnt = extract_text_from_output(outputs)
    print(
        f"{text_type}情况下，含有 </think> 文本的占比: {think_cnt}/{len(text_full)}"
    )

    # 准备结果字典
    results: dict[str, dict[str, Any]] = {
        "outputs": {
            "full_text": text_full,
            "answer_text": text_filtered,
        },
        "metrics": {
            "ppl": [],
            "bleu": [],
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
            "bert_score": [],
            "log_diversity": [],
        },
        "avg_metrics": {},
    }

    # 计算各种指标
    for i, answer_text in enumerate(text_filtered):
        # 计算 PPL
        output_token_logprobs = (
            outputs[i].outputs[0].logprobs
            if hasattr(outputs[i].outputs[0], "logprobs")
            else None
        )
        ppl = calculate_ppl(
            answer_text,
            tokenizer,
            output_token_logprobs,
        )
        results["metrics"]["ppl"].append(ppl)

        # 计算 log_diversity
        log_diversity = calculate_log_diversity(answer_text)
        results["metrics"]["log_diversity"].append(log_diversity)

        # 计算与参考文本的相似度指标
        if i < len(reference):
            reference_metrics = calculate_reference_metrics(
                answer_text, reference[i]
            )
            for metric_name, value in reference_metrics.items():
                results["metrics"][metric_name].append(value)

    # 计算平均指标并添加到结果中
    avg_metrics: dict[str, float | None] = calculate_avg_metrics(
        results["metrics"]
    )
    results["avg_metrics"] = avg_metrics

    # 打印平均指标
    print(f"\n{text_type}指标汇总:")
    for metric_name, value in avg_metrics.items():
        if value is not None:
            print(f"{text_type} {metric_name[4:]}: {value:.4f}")
        else:
            print(f"{text_type} {metric_name[4:]}: N/A")

    return outputs, text_full, text_filtered, results


def roc_evaluation(
    watermark: BaseWatermark,
    watermark_text: list[str | None],
    nowatermark_text: list[str | None],
    dataset: BaseDataset,
    labels: list[str] = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
    rule="best",
    target_fpr: float | None = None,
    reverse: bool = False,
) -> dict[str, float]:
    """处理 ROC 曲线评估，返回评估指标"""
    print("\n正在使用 ROC 曲线计算器评估水印检测效果...")
    from evaluation.pipelines.detection import DetectionPipelineReturnType
    from evaluation.pipelines.vllm_detection import (
        WatermarkDetectionVLLMPipeline,
    )
    from evaluation.tools.success_rate_calculator import (
        DynamicThresholdSuccessRateCalculator,
    )

    # 创建评估 pipeline
    pipeline = WatermarkDetectionVLLMPipeline(
        dataset=dataset,
        show_progress=True,
        return_type=DetectionPipelineReturnType.SCORES,
    )

    # 过滤掉 None 值
    watermark_text_filtered = [t for t in watermark_text if t is not None]
    nowatermark_text_filtered = [t for t in nowatermark_text if t is not None]

    if not watermark_text_filtered or not nowatermark_text_filtered:
        print("警告: 没有足够的有效文本进行 ROC 评估")
        return {}

    watermarked_scores: list[float] = pipeline.evaluate(
        watermark, watermark_text_filtered
    )  # type: ignore
    nowatermarked_scores: list[float] = pipeline.evaluate(
        watermark, nowatermark_text_filtered
    )  # type: ignore

    auroc = roc_auc_score(
        [1] * len(watermarked_scores) + [0] * len(nowatermarked_scores),
        watermarked_scores + nowatermarked_scores,
    )

    calculator = DynamicThresholdSuccessRateCalculator(
        labels=labels,
        rule=rule,
        target_fpr=target_fpr,
        reverse=reverse,
    )

    # 计算并输出评估指标
    metrics: dict[str, float] = calculator.calculate(
        watermarked_scores, nowatermarked_scores
    )
    metrics["AUROC"] = float(auroc)

    print("\nROC 曲线水印检测评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def gpt_evaluation(
    prompts: list[str],
    nowatermark_text: list[str | None],
    watermark_text: list[str | None],
    openai_api_key: str,
    openai_model: str = "text-davinci-003",
) -> dict[str, Any]:
    """处理 GPT 文本判别评估，返回评估结果"""
    # 过滤掉 None 值
    nowatermark_text_filtered = [t for t in nowatermark_text if t is not None]
    watermark_text_filtered = [t for t in watermark_text if t is not None]

    if not nowatermark_text_filtered or not watermark_text_filtered:
        print("警告: 没有足够的有效文本进行 GPT 评估")
        return {"error": "缺少有效文本"}

    print("\n正在使用 GPT 文本判别器评估水印对文本质量的影响...")
    # 设置 OpenAI API 密钥
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # 实例化 GPT 文本判别器
    gpt_discriminator = GPTTextDiscriminator(
        openai_model=openai_model,
        task_description="评估文本的自然度、流畅度和整体质量",
    )

    # 选择部分样本进行 GPT 判别（为了控制 API 成本）
    sample_size = min(10, len(nowatermark_text_filtered))
    sample_indices = np.random.choice(
        len(nowatermark_text_filtered), sample_size, replace=False
    )

    gpt_results = []
    for idx in sample_indices:
        # 评估无水印文本和有水印文本的质量差异
        prompt_idx = min(idx, len(prompts) - 1)
        prompt = prompts[prompt_idx]
        result = gpt_discriminator.analyze(
            nowatermark_text_filtered[idx], watermark_text_filtered[idx], prompt
        )
        gpt_results.append(result)

    # 统计 GPT 判别结果
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
    img_path: str, watermark: Any, text: str, force_create: bool = False
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
    # 准备输出目录
    output_dir: str = prepare_output_dir(
        model_path=args.model_path, dataset_len=args.dataset_len
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
    if args.watermark_after_think:
        print("只在 </think> 之后使用水印")
        watermark = AutoWatermarkForRLLM(
            algorithm_name=args.algorithm_name,
            algorithm_config=f"config/{args.algorithm_name}.json",
            transformers_config=transformers_config,
        )
    else:
        print("在整个文本中使用水印")
        watermark = AutoWatermarkForVLLM(
            algorithm_name=args.algorithm_name,
            algorithm_config=f"config/{args.algorithm_name}.json",
            transformers_config=transformers_config,
        )

    # 加载数据集
    dataset = C4Dataset(
        data_source="dataset/c4/processed_c4.json", max_samples=args.dataset_len
    )
    prompts = dataset.prompts
    natural = dataset.natural_texts
    conversations: list[ChatCompletionMessageParam] = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]  # type: ignore

    # 首先尝试加载之前的无水印结果
    previous_nowatermark_results = load_previous_no_watermark_results(
        file_paths
    )

    if previous_nowatermark_results:
        print("使用之前的无水印生成结果，跳过重新生成...")
        nowatermark_results = previous_nowatermark_results
        # 提取之前保存的文本
        # nowatermark_text_full = nowatermark_results.get("text_full", [])
        nowatermark_text = nowatermark_results.get("text_filtered", [])
    else:
        print("未找到之前的无水印生成结果，开始生成...")
        # 处理无水印生成
        (
            _,
            _,
            nowatermark_text,
            nowatermark_results,
        ) = process_generation(
            model=model,
            sampling_params=sampling_params,
            conversations=conversations,
            tokenizer=tokenizer,
            reference=natural,
            is_watermark=False,
        )

        # 保存无水印结果
        save_results(
            file_paths["no_watermark_results"],
            nowatermark_results,
        )

    # 处理有水印生成
    (
        _,
        _,
        watermark_text,
        watermark_results,
    ) = process_generation(
        model=model,
        sampling_params=sampling_params,
        conversations=conversations,
        tokenizer=tokenizer,
        reference=natural,
        is_watermark=True,
        watermark=watermark,
    )

    # 处理 ROC 评估并添加到结果中
    if args.use_dynamic_threshold:
        roc_metrics = roc_evaluation(
            watermark.watermark,
            watermark_text,
            nowatermark_text,
            dataset,
            labels=["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
            rule=args.threshold_rule,
            target_fpr=args.target_fpr,
            reverse=False,
        )
        watermark_results["roc_metrics"] = roc_metrics

    # 处理 GPT 评估并添加到结果中
    if args.use_gpt_discriminator and args.openai_api_key:
        gpt_metrics = gpt_evaluation(
            prompts,
            nowatermark_text,
            watermark_text,
            args.openai_api_key,
            args.openai_model,
        )
        watermark_results["gpt_discrimination"] = gpt_metrics

    # 保存有水印结果
    save_results(
        file_paths["watermark_results"],
        watermark_results,
    )

    # 保存合并结果
    combined_results: dict[str, dict[str, Any]] = {
        "no_watermark": nowatermark_results,
        "watermark": watermark_results,
    }
    save_results(
        file_paths["combined_results"],
        combined_results,
    )

    # 创建可视化
    # 过滤掉 None 值以便可视化
    nowatermark_text_filtered = [t for t in nowatermark_text if t is not None]
    watermark_text_filtered = [t for t in watermark_text if t is not None]

    if nowatermark_text_filtered and watermark_text_filtered:
        # 创建无水印文本的可视化
        if nowatermark_text_filtered and not os.path.exists(
            file_paths["nowatermark_img"]
        ):
            visualization(
                file_paths["nowatermark_img"],
                watermark,
                nowatermark_text_filtered[0],
            )

        # 创建有水印文本的可视化
        if watermark_text_filtered:
            visualization(
                file_paths["watermark_img"],
                watermark,
                watermark_text_filtered[0],
                force_create=True,  # 有水印图片始终重新创建
            )
    else:
        print("警告: 没有足够的有效文本进行可视化")


if __name__ == "__main__":
    model_path = sys.argv[-2]  # "meta-llama/Meta-Llama-3-8B-Instruct"
    method = sys.argv[-1]  # "UPV" "KGW" "Unigram"
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
        "--watermark-after-think",
        action="store_true",
        default=False,
        help="是否仅在 </think> 后添加水印",
    )
    parser.add_argument(
        "--dataset-len",
        type=int,
        default="100",
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
        "--use-dynamic-threshold",
        action="store_true",
        default=True,
        help="是否使用动态阈值评估水印效果",
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
        "--use-gpt-discriminator",
        action="store_true",
        default=False,
        help="是否使用 GPT 文本判别器评估文本质量",
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
