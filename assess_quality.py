import argparse
import os
from typing import Any, cast

import torch
from transformers import set_seed

from cfg import (
    get_dataset,
    get_evaluators,
    get_result_file_paths,
    get_task,
    load_results,
    prepare_output_dir,
    save_results,
)
from evaluation.dataset import BaseDataset
from evaluation.pipelines.quality_analysis import QualityPipelineReturnType
from evaluation.pipelines.vllm_quality_analysis import (
    DirectTextQualityAnalysisVLLMPipeline,
    ExternalDiscriminatorTextQualityAnalysisVLLMPipeline,
    ReferencedTextQualityAnalysisVLLMPipeline,
)
from evaluation.tools.text_editor import (
    CodeGenerationTextEditor,
    CodeGenerationV2TextEditor,
    MathReasoningTextEditor,
    TextEditor,
    TruncatePromptTextEditor,
)
from evaluation.tools.text_quality_analyzer import (
    DirectTextQualityAnalyzer,
    ExternalDiscriminatorTextQualityAnalyzer,
    ReferencedTextQualityAnalyzer,
    TextQualityAnalyzer,
)


def assess_quality(
    dataset: BaseDataset,
    task: str,
    unwatermark_texts: list[str],
    watermark_texts: list[str],
    direct_analyzers: list[TextQualityAnalyzer],
    referenced_analyzers: list[TextQualityAnalyzer],
    external_analyzers: list[TextQualityAnalyzer],
) -> dict[str, dict]:
    """
    评估水印对文本质量的影响
    Args:
        dataset: 数据集对象
        task: 任务类型
        unwatermark_texts: 无水印文本列表
        watermark_texts: 有水印文本列表
        direct_analyzers: 直接质量分析器列表
        referenced_analyzers: 参考质量分析器列表
        external_analyzers: 外部判别器分析器列表
    Returns:
        包含评估结果的字典
    """
    results: dict[str, dict] = {
        "direct": {},
        "referenced": {},
        "external": {},
    }

    
    if direct_analyzers:
        direct_pipeline = DirectTextQualityAnalysisVLLMPipeline(
            dataset=dataset,
            watermarked_text_editor_list=[],
            unwatermarked_text_editor_list=[],
            analyzers=cast(list[DirectTextQualityAnalyzer], direct_analyzers),
            show_progress=True,
            return_type=QualityPipelineReturnType.MEAN_SCORES,
            unwatermarked_text_source="generated",
        )

        metrics = direct_pipeline.evaluate(watermark_texts, unwatermark_texts)

        
        if isinstance(metrics, dict):
            if "watermarked" in metrics:
                for metric_name, value in metrics["watermarked"].items():
                    results["direct"][
                        f"watermarked_{metric_name.lower()}"
                    ] = value
            if "unwatermarked" in metrics:
                for metric_name, value in metrics["unwatermarked"].items():
                    results["direct"][
                        f"unwatermarked_{metric_name.lower()}"
                    ] = value

    
    if referenced_analyzers and dataset.references:
        references = dataset.references

        
        if len(references) != len(watermark_texts) or len(references) != len(
            unwatermark_texts
        ):
            
            min_len = min(
                len(references), len(watermark_texts), len(unwatermark_texts)
            )
            references = references[:min_len]
            watermark_texts = watermark_texts[:min_len]
            unwatermark_texts = unwatermark_texts[:min_len]

        if task == "code-generation":
            ref_editor_list: list[TextEditor] = [
                CodeGenerationV2TextEditor(),
                TruncatePromptTextEditor(),
                CodeGenerationTextEditor(),
            ]
        elif task == "math-reasoning":
            ref_editor_list: list[TextEditor] = [
                MathReasoningTextEditor(),
            ]
        else:
            ref_editor_list: list[TextEditor] = []

        ref_pipeline = ReferencedTextQualityAnalysisVLLMPipeline(
            dataset=dataset,
            watermarked_text_editor_list=ref_editor_list,
            unwatermarked_text_editor_list=ref_editor_list,
            analyzers=cast(
                list[ReferencedTextQualityAnalyzer], referenced_analyzers
            ),
            show_progress=True,
            return_type=QualityPipelineReturnType.MEAN_SCORES,
            unwatermarked_text_source="generated",
        )

        ref_metrics = ref_pipeline.evaluate(watermark_texts, unwatermark_texts)

        
        if isinstance(ref_metrics, dict):
            if "watermarked" in ref_metrics:
                for metric_name, value in ref_metrics["watermarked"].items():
                    results["referenced"][
                        f"watermarked_{metric_name.lower()}"
                    ] = value
            if "unwatermarked" in ref_metrics:
                for metric_name, value in ref_metrics["unwatermarked"].items():
                    results["referenced"][
                        f"unwatermarked_{metric_name.lower()}"
                    ] = value

    
    if external_analyzers:
        if False:
            
            sample_size = min(10, len(watermark_texts), len(unwatermark_texts))
            sample_indices = torch.randperm(len(watermark_texts))[
                :sample_size
            ].tolist()

            sampled_watermark_texts = [
                watermark_texts[i] for i in sample_indices
            ]
            sampled_unwatermark_texts = [
                unwatermark_texts[i] for i in sample_indices
            ]
        else:
            sampled_watermark_texts = watermark_texts
            sampled_unwatermark_texts = unwatermark_texts

        external_pipeline = (
            ExternalDiscriminatorTextQualityAnalysisVLLMPipeline(
                dataset=dataset,
                analyzers=cast(
                    list[ExternalDiscriminatorTextQualityAnalyzer],
                    external_analyzers,
                ),
                show_progress=True,
                return_type=QualityPipelineReturnType.MEAN_SCORES,
                unwatermarked_text_source="generated",
            )
        )

        ext_metrics = external_pipeline.evaluate(
            watermarked_texts=sampled_watermark_texts,
            unwatermarked_texts=sampled_unwatermark_texts,
        )

        
        if isinstance(ext_metrics, dict):
            if "watermarked" in ext_metrics:
                for metric_name, value in ext_metrics["watermarked"].items():
                    results["external"][
                        f"watermarked_{metric_name.lower()}"
                    ] = value
            if "unwatermarked" in ext_metrics:
                for metric_name, value in ext_metrics["unwatermarked"].items():
                    results["external"][
                        f"unwatermarked_{metric_name.lower()}"
                    ] = value

    return results


def main(args) -> None:
    """主函数"""
    set_seed(args.seed)
    
    output_dir = prepare_output_dir(
        model_path=args.model_path,
        dataset_len=args.dataset_len,
        dataset_name=args.dataset_name,
    )
    file_paths = get_result_file_paths(output_dir, args.algorithm)

    
    prev_quality_results = load_results(file_paths["quality_results"])

    if prev_quality_results:
        print(
            f"质量评估结果文件已存在: {file_paths['quality_results']}，跳过评估"
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    
    dataset = get_dataset(args.dataset_name, args.dataset_len, args.seed)

    task = get_task(args.dataset_name)

    
    analyzers = get_evaluators(
        args.dataset_name, args.model_path, device
    )

    watermark_results = load_results(file_paths["watermark_results"])
    nowatermark_results = load_results(file_paths["no_watermark_results"])

    if not watermark_results or not nowatermark_results:
        print("错误: 未找到生成的文本结果，请先运行 generate.py 生成文本")
        return

    
    watermark_texts = watermark_results.get("answer_text", [])
    nowatermark_texts = nowatermark_results.get("answer_text", [])

    # if args.prompt_file:
    #     import json, types
    #     with open(args.prompt_file, "r", encoding="utf-8") as f:
    #         prompt_list = json.load(f)
    
    #             prompt_list = prompt_list.get("prompts", [])
    #     print(len(prompt_list),len(watermark_texts),len(nowatermark_texts))
    #     assert len(prompt_list) == len(watermark_texts) == len(nowatermark_texts), \
    

    
    #     dataset.prompts = prompt_list                      
    #     dataset.get_prompt = types.MethodType(
    #         lambda self, idx, lst=prompt_list: lst[idx],
    #         dataset,
    #     )

    
    quality_results: dict[str, Any] = assess_quality(
        dataset=dataset,
        task=task,
        unwatermark_texts=nowatermark_texts,
        watermark_texts=watermark_texts,
        direct_analyzers=analyzers["direct"],
        referenced_analyzers=analyzers["referenced"],
        external_analyzers=analyzers["external"],
    )

    from evaluation.pipelines.quality_analysis import QualityPipelineReturnType
    from evaluation.pipelines.vllm_quality_analysis import (
        DirectTextQualityAnalysisVLLMPipeline,
        ReferencedTextQualityAnalysisVLLMPipeline,
        ExternalDiscriminatorTextQualityAnalysisVLLMPipeline,
    )

    
    direct_pipeline = DirectTextQualityAnalysisVLLMPipeline(
        dataset=dataset,
        analyzers=analyzers["direct"],
        show_progress=False,
        return_type=QualityPipelineReturnType.SCORES,
    )
    direct_scores = direct_pipeline.evaluate(watermark_texts, nowatermark_texts)

    
    referenced_scores: list[dict] = []
    if analyzers["referenced"] and dataset.references:
        ref_pipeline = ReferencedTextQualityAnalysisVLLMPipeline(
            dataset=dataset,
            analyzers=analyzers["referenced"],
            show_progress=False,
            return_type=QualityPipelineReturnType.SCORES,
        )
        referenced_scores = ref_pipeline.evaluate(watermark_texts, nowatermark_texts)

    
    external_scores: list[dict] = []
    if analyzers["external"]:
        ext_pipeline = ExternalDiscriminatorTextQualityAnalysisVLLMPipeline(
            dataset=dataset,
            analyzers=analyzers["external"],
            show_progress=False,
            return_type=QualityPipelineReturnType.SCORES,
        )
        external_scores = ext_pipeline.evaluate(watermark_texts, nowatermark_texts)

    
    per_sample_records = []
    sample_num = len(direct_scores)  
    for idx in range(sample_num):
        record = {
            "index": idx,
            "watermarked_text": watermark_texts[idx],
            "unwatermarked_text": nowatermark_texts[idx],
            "direct_metrics": {
                "watermarked_prefixppl": direct_scores[idx]["watermarked"]["PrefixPPLCalculator"],
                "unwatermarked_prefixppl": direct_scores[idx]["unwatermarked"]["PrefixPPLCalculator"],
            },
            "referenced_metrics": referenced_scores[idx]["watermarked"]
            if referenced_scores
            else {},
            "external_metrics": external_scores[idx]["watermarked"]
            if external_scores
            else {},
        }
        per_sample_records.append(record)

    
    os.makedirs(file_paths["output_dir"], exist_ok=True)
    save_results(
        os.path.join(file_paths["output_dir"], "quality_per_sample.json"),
        {"records": per_sample_records},
    )
    
    print("\n直接文本质量分析结果:")
    for metric_name, value in quality_results["direct"].items():
        print(f"{metric_name}: {value:.4f}")

    print("\n参考文本质量分析结果:")
    for metric_name, value in quality_results["referenced"].items():
        print(f"{metric_name}: {value:.4f}")

    print("\n外部判别器质量分析结果:")
    for metric_name, value in quality_results["external"].items():
        print(f"{metric_name}: {value:.4f}")

    
    save_results(file_paths["quality_results"], quality_results)
    print(f"\n评估结果已保存至 {file_paths['quality_results']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="KGW")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="c4",
        help="数据集类型",
    )
    parser.add_argument("--dataset-len", type=int, default=200)
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/QwQ-32B",
        help="模型路径，例如 Qwen/QwQ-32B",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()
    main(args)