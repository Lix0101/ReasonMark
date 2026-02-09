import os
from typing import Any, Optional

import torch
from PIL import ImageDraw
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.generation import GenerateDecoderOnlyOutput

from cfg import get_dataset, get_task_prompt_builder, prepare_output_dir
from visualize.color_scheme import ColorSchemeForContinuousVisualization
from visualize.data_for_visualization import DataForVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import ContinuousLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import ContinuousVisualizer


def chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    prompt: str,
) -> dict[str, Any]:
    """
    处理单个文本生成，返回生成结果和logits

    Args:
        model: 模型
        tokenizer: 分词器
        gen_config: 生成配置
        prompt: 提示文本

    Returns:
        包含生成文本、token_ids和logits的字典
    """
    # 将提示文本转换为模型输入
    conversation = [{"role": "user", "content": prompt}]
    input_text: str = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )  # type: ignore

    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]

    # 生成文本，确保返回logits
    outputs: GenerateDecoderOnlyOutput = model.generate(
        **model_inputs,  # type: ignore
        generation_config=gen_config,
        tokenizer=tokenizer,
    )

    # 提取生成的token ids
    generated_token_ids = outputs.sequences[0, input_length:]

    # 解码生成的文本
    generated_text = tokenizer.decode(
        generated_token_ids, skip_special_tokens=True
    )

    logits = torch.stack(outputs.logits)  # [seq_len, batch_size, vocab_size]
    logits = logits.squeeze(1)  # [seq_len, vocab_size]

    # 计算logprobs和probs
    probs = torch.softmax(logits, dim=-1)  # [seq_len, vocab_size]
    logprobs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]

    # 直接使用所有token ids
    filtered_tokens = generated_token_ids.tolist()
    filtered_logits = logits
    filtered_probs = probs

    # 丢弃最后一个token (</think>)
    if len(filtered_tokens) > 0:
        filtered_tokens = filtered_tokens[:-1]
        filtered_logits = (
            filtered_logits[:-1]
            if len(filtered_logits) > 0
            else filtered_logits
        )
        filtered_probs = (
            filtered_probs[:-1] if len(filtered_probs) > 0 else filtered_probs
        )

    return {
        "text": generated_text,
        "token_ids": filtered_tokens,
        "logits": filtered_logits,
        "probs": filtered_probs,
        "prompt": prompt,
    }


def plot_text_logits_with_visualizer(
    prompt: str,
    token_values: list[float],
    token_texts: list[str],
    plot_name: str,
    output_dir: str,
    top_k_indices: Optional[list[int]] = None,
    value_type: str = "logits",
):
    """
    使用visualizer库绘制文本logits/probs热力图，并用红框标注top k token

    Args:
        prompt: 原始提示文本
        token_values: 每个token的logits或probs值
        token_texts: 每个token对应的文本
        plot_name: 图的名称
        output_dir: 输出目录
        top_k_indices: top k token的索引
        value_type: 值类型，"logits"或"probs"
    """
    # 创建Visualizer所需对象
    font_settings = FontSettings()
    page_layout_settings = PageLayoutSettings(
        max_width=800
    )  # 加宽以适应更多文本
    legend_settings = ContinuousLegendSettings()
    color_scheme = ColorSchemeForContinuousVisualization()

    # 归一化值到[0,1]范围，从而使用热力图
    if token_values:
        min_value = min(token_values)
        max_value = max(token_values)
        # 如果是logits，高值显示为蓝色(0)，低值显示为红色(1)
        # 如果是probs，高值显示为蓝色(0)，低值显示为红色(1)
        # 一致处理，都是高值=重要=蓝色
        normalized_values = [
            (
                1 - (v - min_value) / (max_value - min_value)
                if max_value > min_value
                else 0.5
            )
            for v in token_values
        ]
    else:
        normalized_values = []

    # 创建数据对象
    data = DataForVisualization(
        decoded_tokens=token_texts,
        highlight_values=normalized_values,
    )

    # 创建可视化器并生成图像
    visualizer = ContinuousVisualizer(
        color_scheme=color_scheme,
        font_settings=font_settings,
        page_layout_settings=page_layout_settings,
        legend_settings=legend_settings,
    )

    # 生成基本图像
    image = visualizer.visualize(
        data=data,
        show_text=True,
        visualize_weight=False,
        display_legend=True,
    )

    # 后处理图像添加红框标注top k tokens
    if top_k_indices:
        draw = ImageDraw.Draw(image)
        boxes = {}
        line_height = (
            font_settings.font_size + page_layout_settings.line_spacing
        )
        current_line = 0
        current_x = page_layout_settings.margin_l

        for i, token in enumerate(token_texts):
            # 获取token尺寸
            bbox = font_settings.font.getbbox(token)
            token_width = bbox[2] - bbox[0]

            # 计算token位置
            if current_x + token_width > page_layout_settings.max_width:
                current_line += 1
                current_x = page_layout_settings.margin_l

            token_y = page_layout_settings.margin_t + current_line * line_height

            # 存储位置信息
            boxes[i] = (
                current_x,
                token_y,
                current_x + token_width,
                token_y + font_settings.font_size,
            )

            # 更新x坐标
            current_x += token_width + page_layout_settings.token_spacing

        # 绘制top k tokens红框
        for idx in top_k_indices:
            if idx in boxes:
                x1, y1, x2, y2 = boxes[idx]
                draw.rectangle(
                    [x1 - 2, y1 - 2, x2 + 2, y2 + 2],
                    outline=(255, 0, 0),
                    width=2,
                )

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, plot_name))


def analyze_logits_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    prompts: list[str],
    output_dir: str = "",
    token_ratio: float = 0.001,
    plot_indices: list[int] = [],
) -> list[dict[str, Any]]:
    """
    分析思考阶段的logits和probs

    Args:
        model: transformers模型
        tokenizer: 分词器
        gen_config: 生成配置
        prompts: 提示文本列表
        output_dir: 输出目录
        token_ratio: 筛选出top k个token的比例（相对于词表大小）
        plot_indices: 需要绘图的样本索引

    Returns:
        包含分析结果的字典列表
    """
    per_request_results = []

    # 创建输出目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 计算需要筛选的token数量
    vocab_size: int = tokenizer.vocab_size  # type: ignore
    top_k = max(1, int(vocab_size * token_ratio))
    print(
        f"词表大小: {vocab_size}, 选取top {top_k}个token（比例{token_ratio}）"
    )

    # 逐个处理每个提示
    for i, prompt in enumerate(prompts):
        print(f"处理提示 {i+1}/{len(prompts)}...")

        # 生成文本并获取logits和probs
        output: dict[str, Any] = chat(
            model=model,
            tokenizer=tokenizer,
            gen_config=gen_config,
            prompt=prompt,
        )

        # 获取数据
        text = output["text"]
        token_ids = output["token_ids"]
        token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]
        logits = output["logits"]
        probs = output["probs"]

        # 计算所有时间步的logits和probs求和
        sum_logits = torch.sum(logits, dim=0)  # [vocab_size]
        sum_probs = torch.sum(probs, dim=0)  # [vocab_size]

        # 获取top k的索引
        top_k_logits_indices = torch.topk(sum_logits, top_k).indices.tolist()
        top_k_probs_indices = torch.topk(sum_probs, top_k).indices.tolist()

        # 获取top k的token
        top_k_logits_tokens = [
            tokenizer.decode([idx]) for idx in top_k_logits_indices
        ]
        top_k_probs_tokens = [
            tokenizer.decode([idx]) for idx in top_k_probs_indices
        ]

        # 计算每个token位置的logits和probs值（用于热力图）
        token_logits_values = []
        token_probs_values = []

        for pos, token_id in enumerate(token_ids):
            # 获取该token在top k中的排名
            logits_value = sum_logits[token_id].item()
            probs_value = sum_probs[token_id].item()

            token_logits_values.append(logits_value)
            token_probs_values.append(probs_value)

        # 标记出现在top k中的token位置
        top_k_logits_token_positions = []
        top_k_probs_token_positions = []

        for pos, token_id in enumerate(token_ids):
            if token_id in top_k_logits_indices:
                top_k_logits_token_positions.append(pos)
            if token_id in top_k_probs_indices:
                top_k_probs_token_positions.append(pos)

        # 保存结果
        result = {
            "prompt": prompt,
            "think_text": text,
            "top_k_logits_token": top_k_logits_tokens,
            "top_k_probs_token": top_k_probs_tokens,
        }
        per_request_results.append(result)

        # 绘制指定索引请求的图表
        if i in plot_indices:
            # 绘制logits热力图
            plot_text_logits_with_visualizer(
                prompt,
                token_logits_values,
                token_texts,
                f"text_logits_{i}.png",
                plots_dir,
                top_k_logits_token_positions,
                "logits",
            )

            # 绘制probs热力图
            plot_text_logits_with_visualizer(
                prompt,
                token_probs_values,
                token_texts,
                f"text_probs_{i}.png",
                plots_dir,
                top_k_probs_token_positions,
                "probs",
            )

    return per_request_results


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 准备输出目录
    output_dir = prepare_output_dir(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_len=args.dataset_len,
        dir_name="outputs-exp",
    )

    # 创建plots目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 获取数据集和提示构建器
    dataset = get_dataset(args.dataset_name, args.dataset_len, args.seed)
    prompts = dataset.prompts
    task_prompt_builder = get_task_prompt_builder(args.dataset_name)

    # 构建提示
    prompts = [task_prompt_builder(prompt) for prompt in prompts]

    # 加载模型和tokenizer
    print(f"正在加载模型 {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        trust_remote_code=True,
        device_map=device,
    )

    # 确保设置pad_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    # 配置生成参数
    gen_config = GenerationConfig.from_pretrained(args.model_path)

    # 设置必要的参数以获取logits
    gen_config.output_logits = True
    gen_config.return_dict_in_generate = True

    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.do_sample = True
    gen_config.no_repeat_ngram_size = 4
    gen_config.stop_strings = "</think>"

    # 根据命令行参数设置
    if args.max_model_len is not None:
        gen_config.max_length = args.max_model_len
    if args.temperature is not None:
        gen_config.temperature = args.temperature
    if args.min_p is not None:
        gen_config.min_p = args.min_p
    if args.top_p is not None:
        gen_config.top_p = args.top_p
    if args.top_k is not None:
        gen_config.top_k = args.top_k
    if args.repetition_penalty is not None:
        gen_config.repetition_penalty = args.repetition_penalty

    # 分析思考阶段的logits和probs
    print("开始分析思考阶段的logits和probs...")
    results = analyze_logits_probs(
        model=model,
        tokenizer=tokenizer,
        gen_config=gen_config,
        prompts=prompts,
        token_ratio=args.token_ratio,
        plot_indices=args.plot_indices,
        output_dir=output_dir,
    )

    # 保存结果
    import json

    class CompactJSONEncoder(json.JSONEncoder):
        """自定义JSON编码器，确保列表为单行格式"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def encode(self, obj):
            if isinstance(obj, list):
                return "[" + ", ".join(self.encode(item) for item in obj) + "]"
            return super().encode(obj)

    results_path = os.path.join(output_dir, "text_logits.json")
    with open(results_path, "w") as f:
        json.dump(results, f, cls=CompactJSONEncoder, indent=2)

    print(f"分析结果已保存至 {results_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/QwQ-32B",
        help="模型路径，例如 Qwen/QwQ-32B",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="c4",
        choices=[
            "c4",
            "wmt16_de_en",
            "wmt19_zh_en",
            "human_eval",
            "gsm8k",
            "cnn_dailymail",
        ],
        help="数据集类型",
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
        "--quantization",
        type=str,
        default=None,
        help="量化方法，例如 awq, awq_marlin 等",
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
        default=4096,
        help="模型最长上下文",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3800,
        help="最大生成 tokens 数",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=32,
        help="最小生成 tokens 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="采样温度",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top_k 参数",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="核采样 top_p",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="min_p 参数",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="presence_penalty 参数",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="repetition_penalty 参数",
    )
    parser.add_argument(
        "--token-ratio",
        type=float,
        default=0.001,
        help="筛选出top k个token的比例（相对于词表大小），默认0.001",
    )
    parser.add_argument(
        "--plot-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="需要绘制图表的样本索引列表，例如 --plot-indices 0 5 10",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
