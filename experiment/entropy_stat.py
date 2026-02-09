import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
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


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算给定 logits 的熵

    参数:
        logits (torch.Tensor): 模型输出的 logits，可以是 [vocab_size] 或 [seq_len, vocab_size]

    返回:
        torch.Tensor: 熵值，如果输入是 [vocab_size]，则返回标量；如果输入是 [seq_len, vocab_size]，则返回 [seq_len]
    """
    # 处理单个logits向量 [vocab_size] 或批量logits [seq_len, vocab_size]
    # 批量logits [seq_len, vocab_size]
    probs = torch.softmax(logits, dim=-1)  # [seq_len, vocab_size]
    log_probs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [seq_len]
    if entropy.dim() == 0:
        return entropy.unsqueeze(0)
    return entropy


def filter_special_tokens(token_ids: list[int], tokenizer) -> list[int]:
    """
    过滤掉特殊token ids

    Args:
        token_ids: token id列表
        tokenizer: 使用的tokenizer

    Returns:
        过滤后的token id列表
    """
    # 获取所有特殊token id
    special_ids = tokenizer.all_special_ids
    # 过滤掉特殊token
    filtered_ids = [tid for tid in token_ids if tid not in special_ids]
    return filtered_ids


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

    # 计算logprobs
    logprobs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]

    if False:
        # 过滤掉特殊token
        special_ids: list[int] = tokenizer.all_special_ids
        token_ids: list[int] = generated_token_ids.tolist()
        filtered_token_ids: list[int] = [
            token_id for token_id in token_ids if token_id not in special_ids
        ]

        # 获取过滤后token的索引，以便提取对应的logprobs
        filtered_indices = []
        filtered_tokens = []

        for i, token_id in enumerate(token_ids):
            if token_id in filtered_token_ids:
                filtered_indices.append(i)
                filtered_tokens.append(token_id)

        # 提取这些位置的logprobs
        filtered_logprobs = logprobs[filtered_indices]
    else:
        # 直接使用所有token ids
        filtered_tokens = generated_token_ids.tolist()
        filtered_logprobs = logprobs

    # 丢弃最后一个token (</think>)
    if len(filtered_tokens) > 0:
        filtered_tokens = filtered_tokens[:-1]
        filtered_logprobs = (
            filtered_logprobs[:-1]
            if len(filtered_logprobs) > 0
            else filtered_logprobs
        )

    return {
        "text": generated_text,
        "token_ids": filtered_tokens,
        "logprobs": filtered_logprobs,
        "prompt": prompt,
    }


def plot_text_entropy_with_visualizer(
    prompt: str,
    token_entropies: list[float],
    token_texts: list[str],
    thresholds: dict[str, float],
    plot_name: str,
    output_dir: str,
    low_entropy_indices: Optional[list[int]] = None,
    mid_entropy_indices: Optional[list[int]] = None,
):
    """
    使用visualizer库绘制文本熵热力图，并用彩色框标注不同熵值范围的token

    Args:
        prompt: 原始提示文本
        token_entropies: 每个token的熵值
        token_texts: 每个token对应的文本
        thresholds: 熵阈值字典
        plot_name: 图的名称
        output_dir: 输出目录
        low_entropy_indices: 低熵值token的索引（top 10%）
        mid_entropy_indices: 中熵值token的索引（top 10-20%）
    """
    # 创建Visualizer所需对象
    font_settings = FontSettings()
    page_layout_settings = PageLayoutSettings(
        max_width=800
    )  # 加宽以适应更多文本
    legend_settings = ContinuousLegendSettings()
    color_scheme = ColorSchemeForContinuousVisualization()

    # 归一化熵值到[0,1]范围，从而使用热力图
    if token_entropies:
        min_entropy = min(token_entropies)
        max_entropy = max(token_entropies)
        # 反转归一化，使低熵值显示为蓝色（0接近0，高熵值为1）
        normalized_entropies = [
            (
                1 - (e - min_entropy) / (max_entropy - min_entropy)
                if max_entropy > min_entropy
                else 0.5
            )
            for e in token_entropies
        ]
    else:
        normalized_entropies = []

    # 创建数据对象
    data = DataForVisualization(
        decoded_tokens=token_texts,
        highlight_values=normalized_entropies,
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

    # 后处理图像添加彩色框
    if low_entropy_indices or mid_entropy_indices:
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

        # 绘制低熵区域框（红色）
        if low_entropy_indices:
            for idx in low_entropy_indices:
                if idx in boxes:
                    x1, y1, x2, y2 = boxes[idx]
                    draw.rectangle(
                        [x1 - 2, y1 - 2, x2 + 2, y2 + 2],
                        outline=(255, 0, 0),
                        width=2,
                    )

        # 绘制中熵区域框（蓝色）
        if mid_entropy_indices:
            for idx in mid_entropy_indices:
                if idx in boxes:
                    x1, y1, x2, y2 = boxes[idx]
                    draw.rectangle(
                        [x1 - 2, y1 - 2, x2 + 2, y2 + 2],
                        outline=(0, 0, 255),
                        width=2,
                    )

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, plot_name))


def plot_prompt_and_reference(
    prompt: str,
    reference: str,
    plot_name: str,
    output_dir: str,
):
    """
    绘制prompt和reference对比图，用不同颜色标注文本

    Args:
        prompt: 原始提示文本
        reference: 参考回答
        plot_name: 图的名称
        output_dir: 输出目录
    """
    # 创建Visualizer所需对象
    font_settings = FontSettings()
    page_layout_settings = PageLayoutSettings(
        max_width=800
    )  # 加宽以适应更多文本
    legend_settings = ContinuousLegendSettings()
    color_scheme = ColorSchemeForContinuousVisualization()

    # 将prompt和reference分词
    prompt_tokens = [char for char in prompt]
    reference_tokens = [char for char in reference]

    # 创建数据对象 - 合并两个部分的token
    all_tokens = prompt_tokens + ["[SEP]"] + reference_tokens

    # 创建颜色值 - 只用于占位
    highlight_values = [0.5] * len(all_tokens)

    # 创建数据对象
    image = Image.new(
        "RGB",
        (page_layout_settings.max_width + 100, 1000),
        color=color_scheme.background_color,
    )
    draw = ImageDraw.Draw(image)

    # 绘制提示文本（绿色）
    current_x = page_layout_settings.margin_l
    current_y = page_layout_settings.margin_t
    prompt_color = (0, 128, 0)  # 绿色
    ref_color = (128, 0, 128)  # 紫色
    line_height = font_settings.font_size + page_layout_settings.line_spacing

    # 绘制提示标题
    draw.text(
        (current_x, current_y),
        "提示文本:",
        fill="black",
        font=font_settings.font,
    )
    current_y += line_height

    # 绘制提示内容
    for char in prompt:
        # 获取字符尺寸
        bbox = font_settings.font.getbbox(char)
        char_width = bbox[2] - bbox[0]

        # 检查是否需要换行
        if current_x + char_width > page_layout_settings.max_width:
            current_x = page_layout_settings.margin_l
            current_y += line_height

        # 绘制字符
        draw.text(
            (current_x, current_y),
            char,
            fill=prompt_color,
            font=font_settings.font,
        )
        current_x += char_width

    # 移动到下一部分
    current_x = page_layout_settings.margin_l
    current_y += line_height * 2

    # 绘制参考标题
    draw.text(
        (current_x, current_y),
        "参考回答:",
        fill="black",
        font=font_settings.font,
    )
    current_y += line_height

    # 绘制参考内容
    for char in reference:
        # 获取字符尺寸
        bbox = font_settings.font.getbbox(char)
        char_width = bbox[2] - bbox[0]

        # 检查是否需要换行
        if current_x + char_width > page_layout_settings.max_width:
            current_x = page_layout_settings.margin_l
            current_y += line_height

        # 绘制字符
        draw.text(
            (current_x, current_y),
            char,
            fill=ref_color,
            font=font_settings.font,
        )
        current_x += char_width

    # 裁剪图像到实际内容
    image = image.crop(
        (0, 0, page_layout_settings.max_width + 100, current_y + line_height)
    )

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, plot_name))


def plot_entropy_trend(
    token_texts: list[str],
    entropies: list[float],
    output_dir: str,
    request_idx: int,
    tokens_per_plot: int = 50,
):
    """
    绘制熵值趋势折线图，每张图包含固定数量的token

    Args:
        token_texts: token文本列表
        entropies: 对应的熵值列表
        output_dir: 输出目录
        request_idx: 请求索引
        tokens_per_plot: 每张图包含的token数量
    """
    if not token_texts or not entropies:
        print(f"请求 {request_idx} 没有足够的数据来绘制熵值趋势图")
        return

    # 创建请求专用目录
    request_dir = os.path.join(output_dir, str(request_idx))
    os.makedirs(request_dir, exist_ok=True)

    # 计算需要绘制的图表数量
    total_tokens = len(token_texts)
    num_plots = (
        total_tokens + tokens_per_plot - 1
    ) // tokens_per_plot  # 向上取整

    for plot_idx in range(num_plots):
        # 计算当前图表包含的token范围
        start_idx = plot_idx * tokens_per_plot
        end_idx = min(start_idx + tokens_per_plot, total_tokens)

        # 提取当前范围的数据
        current_tokens = token_texts[start_idx:end_idx]
        current_entropies = entropies[start_idx:end_idx]

        # 创建图表
        plt.figure(figsize=(15, 6))

        # 绘制折线图
        plt.plot(
            range(len(current_entropies)),
            current_entropies,
            marker="o",
            linestyle="-",
            markersize=8,
        )

        # 添加数据标签显示熵值
        for i, entropy in enumerate(current_entropies):
            plt.annotate(
                f"{entropy:.2f}",
                (i, entropy),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # 设置x轴标签为token文本，设置旋转角度以避免重叠
        plt.xticks(
            range(len(current_tokens)), current_tokens, rotation=45, ha="right"
        )

        # 添加网格线使图表更易读
        plt.grid(True, linestyle="--", alpha=0.7)

        # 设置图表标题和轴标签
        plt.title(
            f"Token熵值趋势 (请求 {request_idx}, token {start_idx+1}-{end_idx})"
        )
        plt.xlabel("Token")
        plt.ylabel("熵值")

        # 调整布局以适应旋转后的标签
        plt.tight_layout()

        # 保存图表
        plot_filename = f"entropy_trend_{start_idx+1}_{end_idx}.png"
        plt.savefig(os.path.join(request_dir, plot_filename))
        plt.close()

        print(f"熵值趋势图已保存: {os.path.join(request_dir, plot_filename)}")


def analyze_think_phase(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    prompts: list[str],
    references: Optional[list[str]] = None,
    output_dir: str = "",
    entropy_threshold_percent: float = 0.2,
    plot_indices: list[int] = [],
    tokens_per_plot: int = 50,
) -> dict[str, Any]:
    """
    分析思考阶段的熵值

    Args:
        model: transformers模型
        tokenizer: 分词器
        gen_config: 生成配置
        prompts: 提示文本列表
        references: 参考回答列表，可选
        output_dir: 输出目录
        entropy_threshold_percent: 熵阈值百分比位置
        plot_indices: 需要绘图的样本索引
        tokens_per_plot: 每张熵值趋势图中显示的token数量

    Returns:
        包含分析结果的字典
    """
    per_request_results = []
    all_entropies = []

    # 创建输出目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 逐个处理每个提示
    for i, prompt in enumerate(prompts):
        print(f"处理提示 {i+1}/{len(prompts)}...")

        # 生成文本并获取logits
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
        logprobs = output["logprobs"]

        # 计算熵值
        entropies = compute_entropy(logprobs).tolist()
        all_entropies.extend(entropies)

        # 保存结果 - 不保存token_ids和entropy字段
        result = {
            "text": text,
            "decoded_texts": list(set(token_texts)),  # 去重
            "prompt": prompt,
        }
        per_request_results.append(result)

    # 计算全局熵阈值
    global_percentiles = {}
    for percent in [10, 20, 30, 40, 50]:
        threshold_key = f"top_{percent}_entropy"
        global_percentiles[threshold_key] = (
            float(np.percentile(all_entropies, percent))
            if all_entropies
            else 0.0
        )

    # 为每个请求添加基于全局阈值的文本提取
    for req_idx, req_result in enumerate(per_request_results):
        # 由于我们不再保存entropy，需要重新计算
        output = chat(
            model=model,
            tokenizer=tokenizer,
            gen_config=gen_config,
            prompt=req_result["prompt"],
        )
        token_ids = output["token_ids"]
        token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]
        logprobs = output["logprobs"]
        entropies = compute_entropy(logprobs).tolist()

        # 提取低熵文本 (top 10%)
        low_entropy_indices = [
            i
            for i, e in enumerate(entropies)
            if e <= global_percentiles["top_10_entropy"]
        ]
        low_entropy_tokens = [token_texts[i] for i in low_entropy_indices]

        # 提取中熵文本 (top 10%-20%)
        mid_entropy_indices = [
            i
            for i, e in enumerate(entropies)
            if global_percentiles["top_10_entropy"]
            < e
            <= global_percentiles["top_20_entropy"]
        ]
        mid_entropy_tokens = [token_texts[i] for i in mid_entropy_indices]

        # 计算比率
        total_tokens = len(token_texts)
        low_entropy_ratio = (
            len(low_entropy_indices) / total_tokens if total_tokens > 0 else 0
        )
        mid_entropy_ratio = (
            len(mid_entropy_indices) / total_tokens if total_tokens > 0 else 0
        )

        # 保存到结果中 - 添加熵值比率统计，并确保为去重列表
        req_result["text_top_10_entropy"] = list(set(low_entropy_tokens))
        req_result["text_top_10_20_entropy"] = list(set(mid_entropy_tokens))
        req_result["text_top_10_entropy_ratio"] = low_entropy_ratio
        req_result["text_top_20_entropy_ratio"] = mid_entropy_ratio

        # 绘制指定索引请求的图表
        if req_idx in plot_indices:
            # 使用项目中的可视化库绘制熵值分布图
            plot_text_entropy_with_visualizer(
                req_result["prompt"],
                entropies,
                token_texts,
                global_percentiles,
                f"text_entropy_{req_idx}.png",
                plots_dir,
                low_entropy_indices,
                mid_entropy_indices,
            )

            # 绘制熵值变化趋势折线图
            plot_entropy_trend(
                token_texts,
                entropies,
                plots_dir,
                req_idx,
                tokens_per_plot=tokens_per_plot,  # 使用参数传递的token数量
            )

            # 如果有参考回答，绘制prompt和reference对比图
            if references and req_idx < len(references):
                plot_prompt_and_reference(
                    req_result["prompt"],
                    references[req_idx],
                    f"prompt_reference_{req_idx}.png",
                    plots_dir,
                )

    # 返回最终结果
    return {
        "global_stat": global_percentiles,
        "per_request_stat": per_request_results,
    }


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
    references = dataset.references if hasattr(dataset, "references") else None
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
    # if args.max_new_tokens is not None:
    #     gen_config.max_new_tokens = args.max_new_tokens
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

    # 分析思考阶段
    print("开始分析思考阶段的熵值...")
    results = analyze_think_phase(
        model=model,
        tokenizer=tokenizer,
        gen_config=gen_config,
        prompts=prompts,
        references=references,
        entropy_threshold_percent=args.entropy_threshold,
        plot_indices=args.plot_indices,
        output_dir=output_dir,
        tokens_per_plot=args.tokens_per_plot,  # 传递tokens_per_plot参数
    )

    # 保存结果 - 使JSON中的列表单行存储
    import json

    class CompactJSONEncoder(json.JSONEncoder):
        """自定义JSON编码器，确保列表为单行格式"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def encode(self, obj):
            if isinstance(obj, list):
                return "[" + ", ".join(self.encode(item) for item in obj) + "]"
            return super().encode(obj)

    results_path = os.path.join(output_dir, "think_text_entropy.json")
    with open(results_path, "w") as f:
        json.dump(results, f, cls=CompactJSONEncoder, indent=2)

    print(f"分析结果已保存至 {results_path}")
    print(
        f"全局熵阈值 (top 10%): {results['global_stat']['top_10_entropy']:.4f}"
    )
    print(
        f"全局熵阈值 (top 20%): {results['global_stat']['top_20_entropy']:.4f}"
    )


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
        "--entropy-threshold",
        type=float,
        default=0.2,
        help="熵阈值百分比位置，默认0.2表示20%",
    )
    parser.add_argument(
        "--plot-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="需要绘制图表的样本索引列表，例如 --plot-indices 0 5 10",
    )
    parser.add_argument(
        "--tokens-per-plot",
        type=int,
        default=50,
        help="熵值趋势图中每张图显示的token数量",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
