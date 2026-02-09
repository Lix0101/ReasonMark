import json
import os
import torch
from collections.abc import Callable
from functools import partial
from typing import Any


from evaluation.dataset import (
    AIME2025Dataset,
    BaseDataset,
    C4Dataset,
    CNN_DailyMailDataset,
    GSM8KDataset,
    HumanEvalDataset,
    MMLU_ProDataset,
    WMT16DE_ENDataset,
    WMT19ZH_ENDataset,
)
from evaluation.tools.text_quality_analyzer import (
    BERTScoreCalculator,
    BLEUCalculator,
    GPTTextDiscriminator,
    LogDiversityAnalyzer,
    MathAccuracyCalculator,
    PassOrNotJudger,
    PrefixPPLCalculator,
    ROUGE1Calculator,
    ROUGE2Calculator,
    ROUGELCalculator,
    TextQualityAnalyzer,
)
from visualize.visualizer import (
    BaseVisualizer,
    ColorSchemeForContinuousVisualization,
    ColorSchemeForDiscreteVisualization,
    ContinuousLegendSettings,
    ContinuousVisualizer,
    DiscreteLegendSettings,
    DiscreteVisualizer,
    FontSettings,
    PageLayoutSettings,
)

KGW_FAMILY: list[str] = [
    "KGW",
    "SWEET",
    "DIP",
    "Unigram",
    "SIR",
    "XSIR",
    "TS",
    "EWD",
    "UPV",
    "Unbiased",
    "SynthID",
    "OURS",
]
CHRIST_FAMILY: list[str] = [
    "EXP",
    "EXPEdit",
    "EXPGumbel",
    "ITSEdit",
    "PF",
]


def build_code_generation_instruction(
    question: str, languge: str = "python"
) -> str:
    return """
Please continue to complete the function. Do not modify any given code (including the docstring), and only provide completion without explanation or comments. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
        languge.lower(), question.strip()
    )


def build_text_completion_instruction(question: str) -> str:
    return f"Please continue the following text and provide only the continuation without any explanations or comments. Here is the given text to do completion:\n{question}"


def build_machine_translation_instruction(
    question: str, src_lang: str, dst_lang: str
) -> str:
    return f"Please translate the following {src_lang} text into {dst_lang} while preserving all original formatting, style, and special characters. Provide only the translation without any explanations or comments. Here is the given text to translate:\n{question}"


def build_math_reasoning_instruction(question: str) -> str:
    return (
        "Please reason step by step, and put your final answer within \\boxed{}. Here is the problem:\n"
        + question
    )


def build_text_summarization_instruction(question: str) -> str:
    return f"Please summarize the following article in 50 words, and provide only the summary without any explanations or comments:\n{question}"


def build_mcq_instruction(
    example: dict[str, Any], choices_key="options", question_key="question"
) -> str:
    """构建多项选择题指令。

    Args:
        example: 包含问题和选项的字典
        choices_key: 选项在字典中的键名
        question_key: 问题在字典中的键名

    Returns:
        格式化的多项选择题指令
    """
    question: str = example[question_key]
    options: list[str] = example[choices_key]
    formatted_options = "\n".join(
        f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options)
    )

    instruction = f"""Question: {question}
Options:
{formatted_options}
Note: Only one option may be selected. Please reason step by step, and put your final answer (a single letter) within \\boxed{{}}."""

    return instruction


DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "c4": {
        "path": "dataset/c4/processed_c4.json",
        "class": C4Dataset,
        "metrics": {
            # "direct": [PrefixPPLCalculator, LogDiversityAnalyzer],
            "direct": [PrefixPPLCalculator],
            "referenced": [],
            "external": [],
        },
        "task": "text-completion",
        "task_prompt": build_text_completion_instruction,
        "rating_prompt": "",
    },
    "cnn_dailymail": {
        "path": "dataset/cnn_dailymail/test-00000-of-00001.jsonl",
        "class": partial(CNN_DailyMailDataset, global_prompt=""),
        "metrics": {
            "direct": [],
            "referenced": [
                BLEUCalculator,
                ROUGE1Calculator,
                ROUGE2Calculator,
                ROUGELCalculator,
                BERTScoreCalculator,
            ],
            # "external": [GPTTextDiscriminator],
            "external": [],
        },
        "task": "text-summarization",
        "task_prompt": build_text_summarization_instruction,
        "rating_prompt": "Summarize the following article, and provide only the summary without any explanations or comments.",
    },
    "wmt16_de_en": {
        "path": "dataset/wmt16_de_en/validation.jsonl",
        "class": WMT16DE_ENDataset,
        "metrics": {
            "direct": [],
            "referenced": [
                BLEUCalculator,
                ROUGE1Calculator,
                ROUGE2Calculator,
                ROUGELCalculator,
                BERTScoreCalculator,
            ],
            # "external": [GPTTextDiscriminator],
            "external": [],
        },
        "task": "machine-translation",
        "task_prompt": partial(
            build_machine_translation_instruction,
            src_lang="German",
            dst_lang="English",
        ),
        "rating_prompt": "Translate the following German text into English, and provide only the translation without any explanations or comments.",
    },
    "wmt19_zh_en": {
        "path": "dataset/wmt19_zh_en/validation.jsonl",
        "class": WMT19ZH_ENDataset,
        "metrics": {
            "direct": [],
            "referenced": [
                BLEUCalculator,
                ROUGE1Calculator,
                ROUGE2Calculator,
                ROUGELCalculator,
                BERTScoreCalculator,
            ],
            # "external": [GPTTextDiscriminator],
            "external": [],
        },
        "task": "machine-translation",
        "task_prompt": partial(
            build_machine_translation_instruction,
            src_lang="Chinese",
            dst_lang="English",
        ),
        "rating_prompt": "Translate the following Chinese text into English, and provide only the translation without any explanations or comments.",
    },
    "human_eval": {
        "path": "dataset/human_eval/test.jsonl",
        "class": HumanEvalDataset,
        "metrics": {
            "direct": [],
            "referenced": [PassOrNotJudger],
            "external": [],
        },
        "task": "code-generation",
        "task_prompt": partial(
            build_code_generation_instruction, languge="python"
        ),
        "rating_prompt": "",
    },
    "gsm8k": {
        "path": "dataset/gsm8k/test.jsonl",
        "class": GSM8KDataset,
        "metrics": {
            "direct": [],
            "referenced": [MathAccuracyCalculator],
            "external": [],
        },
        "task": "math-reasoning",
        "task_prompt": build_math_reasoning_instruction,
        "rating_prompt": "",
    },
    "mmlu_pro": {
        "path": "dataset/mmlu_pro/validation.jsonl",
        "class": MMLU_ProDataset,
        "metrics": {
            "direct": [],
            "referenced": [MathAccuracyCalculator],
            "external": [],
        },
        "task": "multiple-choice",
        "task_prompt": partial(
            build_mcq_instruction,
            choices_key="options",
            question_key="question",
        ),
        "rating_prompt": "",
    },
    "aime_2025": {
        "path": "dataset/aime_2025/train.jsonl",
        "class": AIME2025Dataset,
        "metrics": {
            "direct": [],
            "referenced": [MathAccuracyCalculator],
            "external": [],
        },
        "task": "math-reasoning",
        "task_prompt": build_math_reasoning_instruction,
        "rating_prompt": "",
    },
}


def get_dataset(
    dataset_name: str,
    max_samples: int | None = None,
    seed: int | None = None,
) -> BaseDataset:
    """
    根据数据集名称获取数据集实例

    Args:
        dataset_name: 数据集名称
        max_samples: 最大样本数量

    Returns:
        数据集实例，类型为 BaseDataset
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"未知的数据集 {dataset_name}，可用选项: {list(DATASET_CONFIG.keys())}"
        )

    dataset_info = DATASET_CONFIG[dataset_name]
    dataset_class = dataset_info["class"]
    return dataset_class(
        dataset_info["path"], max_samples=max_samples, seed=seed
    )


def get_evaluators(
    dataset_name: str,
    oracle_model_path: str = "unsloth/Qwen2.5-72B-bnb-4bit",
    device: str = "cuda",
) -> dict[str, list[TextQualityAnalyzer]]:
    """
    根据数据集名称获取评估器列表

    Args:
        dataset_name: 数据集名称
        oracle_model_path: 模型路径
        device: 设备类型

    Returns:
        评估器字典，类型为 dict[str, list[TextQualityAnalyzer]]
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"未知的数据集 {dataset_name}，可用选项: {list(DATASET_CONFIG.keys())}"
        )

    dataset_info = DATASET_CONFIG[dataset_name]
    dataset_metrics = dataset_info["metrics"]

    evaluators = {"direct": [], "referenced": [], "external": []}

    # 直接文本质量分析器
    if PrefixPPLCalculator in dataset_metrics["direct"]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ppl_calculator = PrefixPPLCalculator(
            model=AutoModelForCausalLM.from_pretrained(
                oracle_model_path, device_map="auto",torch_dtype=torch.float16,
            ),
            tokenizer=AutoTokenizer.from_pretrained(oracle_model_path),
            device=device,
        )
        evaluators["direct"].append(ppl_calculator)

    if LogDiversityAnalyzer in dataset_metrics["direct"]:
        log_diversity_analyzer = LogDiversityAnalyzer()
        evaluators["direct"].append(log_diversity_analyzer)

    # 参考文本质量分析器
    if BLEUCalculator in dataset_metrics["referenced"]:
        evaluators["referenced"].append(BLEUCalculator())

    if ROUGE1Calculator in dataset_metrics["referenced"]:
        evaluators["referenced"].append(ROUGE1Calculator())

    if ROUGE2Calculator in dataset_metrics["referenced"]:
        evaluators["referenced"].append(ROUGE2Calculator())

    if ROUGELCalculator in dataset_metrics["referenced"]:
        evaluators["referenced"].append(ROUGELCalculator())

    if BERTScoreCalculator in dataset_metrics["referenced"]:
        evaluators["referenced"].append(
            BERTScoreCalculator(model_path="google-bert/bert-base-uncased")
        )

    if PassOrNotJudger in dataset_metrics["referenced"]:
        evaluators["referenced"].append(PassOrNotJudger())

    if MathAccuracyCalculator in dataset_metrics["referenced"]:
        evaluators["referenced"].append(MathAccuracyCalculator())

    rating_prompt = get_rating_prompt(dataset_name)
    if GPTTextDiscriminator in dataset_metrics["external"] and rating_prompt:
        evaluators["external"].append(
            GPTTextDiscriminator(
                openai_model="gpt-4",
                task_description=rating_prompt,
            )
        )

    return evaluators


def get_task_prompt_builder(dataset_name: str) -> Callable[[str], str]:
    """
    根据数据集名称获取任务提示处理函数

    Args:
        dataset_name: 数据集名称

    Returns:
        返回处理函数
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"未知的数据集 {dataset_name}，可用选项: {list(DATASET_CONFIG.keys())}"
        )

    return DATASET_CONFIG[dataset_name]["task_prompt"]


def get_rating_prompt(dataset_name: str) -> str | None:
    """
    根据数据集名称获取评分提示

    Args:
        dataset_name: 数据集名称

    Returns:
        评分提示，类型为 str | None
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"未知的数据集 {dataset_name}，可用选项: {list(DATASET_CONFIG.keys())}"
        )

    return DATASET_CONFIG[dataset_name]["rating_prompt"]


def get_task(dataset_name: str) -> str:
    """
    根据数据集名称获取任务类型

    Args:
        dataset_name: 数据集名称

    Returns:
        任务类型，类型为 str
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"未知的数据集 {dataset_name}，可用选项: {list(DATASET_CONFIG.keys())}"
        )

    return DATASET_CONFIG[dataset_name]["task"]


def prepare_output_dir(
    model_path: str,
    dataset_len: int,
    dataset_name: str,
    dir_name: str = "outputs",
    special_chars: list[str] = ["."],
    replacement_char: str = "-",
) -> str:
    """准备输出目录并返回路径"""
    output_dir = os.path.join(os.getcwd(), dir_name)

    # Extract model name based on whether '/' exists in model_path
    if "/" in model_path:
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path

    # Replace special characters with replacement_char
    for char in special_chars:
        model_name = model_name.replace(char, replacement_char)

    output_dir = os.path.join(
        output_dir,
        model_name,
        f"{dataset_name}-{dataset_len}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_result_file_paths(
    output_dir: str, algorithm_name: str
) -> dict[str, str]:
    """获取各种结果文件的路径"""
    watermark_output_dir = os.path.join(output_dir, algorithm_name)
    return {
        "no_watermark_results": os.path.join(
            output_dir, "no_watermark_texts.json"
        ),
        "watermark_results": os.path.join(
            watermark_output_dir, "watermark_texts.json"
        ),
        "combined_results": os.path.join(
            watermark_output_dir, "combined_texts.json"
        ),
        "nowatermark_img": os.path.join(output_dir, "no_watermark.png"),
        "watermark_img": os.path.join(watermark_output_dir, "watermark.png"),
        "detection_results": os.path.join(
            watermark_output_dir, "detectabitily.json"
        ),
        "quality_results": os.path.join(watermark_output_dir, "quality.json"),
        "output_dir": os.path.join(watermark_output_dir, "each_sample"),
    }


def save_results(file_path: str, results: dict[str, Any]) -> None:
    """保存结果到 JSON 文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 写入文件
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(file_path: str) -> dict[str, Any] | None:
    """加载结果文件"""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)
        return results
    return None


def get_visualizer(algorithm_name: str) -> BaseVisualizer:
    """
    Returns the appropriate visualizer based on the algorithm family.

    Args:
        algorithm_name: Name of the watermarking algorithm

    Returns:
        Appropriate visualizer instance for the algorithm
    """
    if algorithm_name in KGW_FAMILY:
        return DiscreteVisualizer(
            color_scheme=ColorSchemeForDiscreteVisualization(),
            font_settings=FontSettings(),
            page_layout_settings=PageLayoutSettings(),
            legend_settings=DiscreteLegendSettings(),
        )
    elif algorithm_name in CHRIST_FAMILY:
        return ContinuousVisualizer(
            color_scheme=ColorSchemeForContinuousVisualization(),
            font_settings=FontSettings(),
            page_layout_settings=PageLayoutSettings(),
            legend_settings=ContinuousLegendSettings(),
        )
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Cannot determine appropriate visualizer."
        )
