import argparse
import json
import os                        # ← 新增
import types 
import torch
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForMaskedLM,
    BertTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
    AutoModelForCausalLM,
)

from cfg import (
    get_dataset,
    get_result_file_paths,
    get_task,
    load_results,
    prepare_output_dir,
    save_results,
)
from evaluation.dataset import BaseDataset
from evaluation.pipelines.detection import DetectionPipelineReturnType
from evaluation.pipelines.vllm_detection import WatermarkDetectionVLLMPipeline
from evaluation.tools.success_rate_calculator import (
    DynamicThresholdSuccessRateCalculator,
)
from evaluation.tools.text_editor import (
    BackTranslationTextEditor,
    CodeGenerationV2TextEditor,
    ContextAwareSynonymSubstitution,
    DeepSeekParaphraser,           # ← 新增
    DipperParaphraser,
    GPTParaphraser,
    SynonymSubstitution,
    TextEditor,
    Translator,
    TruncateTaskTextEditor,
    WordDeletion,
    WordInsertion,
    OpenAI,
    # YoudaoBackTranslationTextEditor,
    DeepSeekBackTranslationTextEditor,
)
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from watermark.base import BaseWatermark


def assess_detectability(
    watermark: BaseWatermark,
    watermark_texts: list[str],
    nowatermark_texts: list[str],
    dataset: BaseDataset,
    task: str,
    labels: list[str] = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
    rule="target_fpr",
    target_fpr: float = 0.01,
    reverse: bool = False,
    attack_name: str | None = None,
    device: str = "cuda",
    min_answer_tokens: int | None = None,
    max_answer_tokens: int | None = None,
    save_scores_path: str | None = None,        # ← 新增
    save_edited_path: str | None = None,        # ← 新增
) -> dict[str, float]:
    """
    评估水印的检测性

    Args:
        watermark: 水印对象
        watermark_texts: 有水印的文本列表
        nowatermark_texts: 无水印的文本列表
        dataset: 数据集对象
        labels: 评估指标标签
        rule: 阈值确定规则
        target_fpr: 目标假阳性率
        reverse: 是否反转评估结果
        attack_name: 攻击名称
        device: 设备类型
        min_answer_tokens: 最小token数量，小于此值的文本将被过滤
        max_answer_tokens: 最大token数量，大于此值的文本将被截断
        save_scores_path: 若不为 None，则把每条样本的得分与标签保存到该路径

    Returns:
        评估指标字典
    """
    print("\n正在使用 ROC 曲线计算器评估水印检测效果...")

    if not watermark_texts or not nowatermark_texts:
        print("警告: 没有足够的有效文本进行 ROC 评估")
        return {}

    # 根据攻击名称选择攻击方式
    attack: TextEditor | None
    match attack_name:
        case "Word-D":
            attack = WordDeletion(ratio=0.3)
        case "Word-S":
            attack = SynonymSubstitution(ratio=0.5)
        case "Word-I":
            attack = WordInsertion(ratio=0.3, attack_type="dispersed")
        case "Word-I(Local)":
            attack = WordInsertion(ratio=0.3, attack_type="local")
        case "Word-S(Context)":
            attack = ContextAwareSynonymSubstitution(
                ratio=0.5,
                tokenizer=BertTokenizer.from_pretrained(
                    "google-bert/bert-large-uncased"
                ),
                model=BertForMaskedLM.from_pretrained(
                    "google-bert/bert-large-uncased"
                ).to(device),
            )
        case "Doc-P(GPT-3.5)":
            attack = GPTParaphraser(
                openai_model="gpt-3.5-turbo",
                prompt="Please rewrite the following text: ",
            )
        case "Doc-P(Dipper)":
            attack = DipperParaphraser(
                tokenizer=T5Tokenizer.from_pretrained("google/t5-v1_1-xxl"),
                model=T5ForConditionalGeneration.from_pretrained(
                    "kalpeshk2011/dipper-paraphraser-xxl",
                    device_map="auto",
                ),
                lex_diversity=60,
                order_diversity=0,
                sent_interval=1,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.75,
                top_k=None,
            )
        case "Doc-P-deepseek":      # ← 新增
            attack = DeepSeekParaphraser(
                api_key="xxx",
                temperature=0.0,
            )
        case "Translation-deepseek":      # ← 新增
            attack = DeepSeekBackTranslationTextEditor(
                api_key="xxx",
                temperature=0.0,
            )
        # case "Translation-Youdao":
        #     attack = YoudaoBackTranslationTextEditor(
        #         app_key="71068ae4f61fd8de",
        #         app_secret="mxFDTgIBxN3NjVVZkHfNzys2R6QtbDF5",
        #     )
        case "Translation":
            attack = BackTranslationTextEditor(
                translate_to_intermediary=Translator(
                    from_lang="en", to_lang="zh"
                ).translate,
                translate_to_source=Translator(
                    from_lang="zh", to_lang="en"
                ).translate,
            )
        case _:
            attack = None
    if attack is not None:
        rule = "best"

    text_editor_list: list[TextEditor] = [attack] if attack else []
    if task == "code-generation":
        text_editor_list.append(CodeGenerationV2TextEditor(language="python"))
        text_editor_list.append(TruncateTaskTextEditor())

    
    store_attacks = {"Translation", "Doc-P-deepseek", "Translation-Youdao","Translation-deepseek"}
    need_store_edit = attack_name in store_attacks and save_edited_path is not None
    pipeline_ret_type = (
        DetectionPipelineReturnType.FULL
        if need_store_edit
        else DetectionPipelineReturnType.SCORES
    )

    # 构建 pipeline（使用 pipeline_ret_type）
    watermark_pipeline = WatermarkDetectionVLLMPipeline(
        dataset=dataset,
        text_editor_list=text_editor_list,
        show_progress=True,
        return_type=pipeline_ret_type,
    )
    unwatermark_pipeline = WatermarkDetectionVLLMPipeline(
        dataset=dataset,
        text_editor_list=text_editor_list,
        show_progress=True,
        return_type=pipeline_ret_type,
    )

    watermarked_eval = watermark_pipeline.evaluate(
        watermark,
        watermark_texts,
        min_answer_tokens=min_answer_tokens,
        max_answer_tokens=max_answer_tokens,
    )
    nowatermarked_eval = unwatermark_pipeline.evaluate(
        watermark,
        nowatermark_texts,
        min_answer_tokens=min_answer_tokens,
        max_answer_tokens=max_answer_tokens,
    )

    # 根据返回类型提取得分
    if pipeline_ret_type is DetectionPipelineReturnType.FULL:
        watermarked_scores = [
            r.detect_result["score"] for r in watermarked_eval
        ]
        nowatermarked_scores = [
            r.detect_result["score"] for r in nowatermarked_eval
        ]
        # --------- 保存改写文本 ----------
        if need_store_edit:
            edited_dict = {
                "watermarked_edited": [r.edited_text for r in watermarked_eval],
                "nowatermarked_edited": [
                    r.edited_text for r in nowatermarked_eval
                ],
            }
            os.makedirs(os.path.dirname(save_edited_path), exist_ok=True)
            with open(save_edited_path, "w", encoding="utf-8") as f:
                json.dump(edited_dict, f, ensure_ascii=False, indent=2)
            print(f"改写文本已保存至 {save_edited_path}")
        # ---------------------------------
    else:
        watermarked_scores = watermarked_eval     # 已是 list[float]
        nowatermarked_scores = nowatermarked_eval

    if not watermarked_scores or not nowatermarked_scores:
        print("警告: 过滤后没有足够的有效文本进行 ROC 评估")
        return {}

    scores = watermarked_scores + nowatermarked_scores
    if reverse:
        scores = [-score for score in scores]

    auroc = roc_auc_score(
        [1] * len(watermarked_scores) + [0] * len(nowatermarked_scores),
        scores,
    )

    calculator = DynamicThresholdSuccessRateCalculator(
        labels=labels,
        rule=rule,
        target_fpr=target_fpr,
        reverse=reverse,
    )

    metrics: dict[str, float] = calculator.calculate(
        watermarked_scores, nowatermarked_scores
    )
    metrics["auroc"] = float(auroc)

    if save_scores_path:
        os.makedirs(os.path.dirname(save_scores_path), exist_ok=True)
        sample_scores = {
            "watermarked_scores": watermarked_scores,
            "nowatermarked_scores": nowatermarked_scores,
            "labels": [1] * len(watermarked_scores)
            + [0] * len(nowatermarked_scores),
            "scores": scores,  # 按顺序拼接，方便直接绘 ROC
        }
        with open(save_scores_path, "w", encoding="utf-8") as f:
            json.dump(sample_scores, f, indent=2)
        print(f"样本级得分已保存至 {save_scores_path}")


    return metrics


def main(args) -> None:
    """主函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    # 获取数据集
    dataset = get_dataset(args.dataset_name, args.dataset_len, args.seed)

    task = get_task(args.dataset_name)

    # 初始化配置
    need_model = args.algorithm.lower() in ["unbiased", "ewd", "sweet"]   # 按需列举
    model_obj = None
    if need_model:
        model_obj = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",           # 多 GPU 自动切分；显存不够可换成 cpu / 4bit
            torch_dtype="auto"           # 或 torch.float16
        )
    # 初始化配置
    transformers_config = TransformersConfig(
        model=model_obj,
        tokenizer=AutoTokenizer.from_pretrained(args.model_path),
        vocab_size=AutoConfig.from_pretrained(args.model_path).vocab_size,
        device=device,
    )
    watermark: BaseWatermark = AutoWatermark.load(
        algorithm_name=args.algorithm,
        algorithm_config=f"config/{args.algorithm}.json",
        transformers_config=transformers_config,
    )

    # 使用 dataset_cfg 中的方法准备输出目录和文件路径
    output_dir = prepare_output_dir(
        model_path=args.model_path,
        dataset_len=args.dataset_len,
        dataset_name=args.dataset_name,
    )
    file_paths = get_result_file_paths(output_dir, args.algorithm)

    # 确定样本级得分保存路径
    scores_save_path = os.path.join(
        file_paths["output_dir"], "detection_sample_scores.json"
    )
    edited_save_path = os.path.join(              # ← 新增
        file_paths["output_dir"], "edited_texts.json"
    )

    # 加载生成的文本
    watermark_results = load_results(file_paths["watermark_results"])
    nowatermark_results = load_results(file_paths["no_watermark_results"])

    if not watermark_results or not nowatermark_results:
        print("错误: 未找到生成的文本结果，请先运行 generate.py 生成文本")
        return

    # 提取文本
    watermark_texts = watermark_results.get("answer_text", [])
    nowatermark_texts = nowatermark_results.get("answer_text", [])

    # 评估
    metrics = assess_detectability(
        watermark=watermark,
        watermark_texts=watermark_texts,
        nowatermark_texts=nowatermark_texts,
        dataset=dataset,
        task=task,
        labels=args.labels,
        rule=args.rules,
        target_fpr=args.target_fpr,
        reverse=args.reverse,
        attack_name=args.attack_name,
        device=device,
        min_answer_tokens=args.min_answer_tokens,
        max_answer_tokens=args.max_answer_tokens,
        save_scores_path=scores_save_path,           
        save_edited_path=edited_save_path,       
    )

    print("\nROC 曲线水印检测评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 保存评估结果
    save_results(file_paths["detection_results"], metrics)
    print(f"评估结果已保存至 {file_paths['detection_results']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="KGW")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/QwQ-32B",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="c4",
        help="数据集类型",
    )
    parser.add_argument("--dataset-len", type=int, default=200)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
    )
    parser.add_argument("--rules", type=str, default="best")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--attack-name", type=str, default=None)
    parser.add_argument(
        "--min-answer-tokens",
        type=int,
        default=None,
        help="最小token数量，小于此值的文本将被过滤",
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=None,
        help="最大token数量，大于此值的文本将被截断",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--save-edited-path",
        type=str,
        default=None,
        help="若不为 None，则把改写后的文本保存到该路径",
    )
    args = parser.parse_args()
    main(args)
