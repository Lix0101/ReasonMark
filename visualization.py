import os

import torch
from transformers import AutoConfig, AutoTokenizer

from cfg import (
    get_result_file_paths,
    get_visualizer,
    load_results,
    prepare_output_dir,
)
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from watermark.base import BaseWatermark


def visualize(
    img_path: str,
    watermark: BaseWatermark,
    text: str,
    force: bool = False,
    figure_width: int | None = None,
    figure_line_spacing: int | None = None,
    save_pdf: bool = False,
    pdf_upscale: int = 4,
) -> None:
    """创建文本可视化图像并保存到指定路径"""
    if not force and os.path.exists(img_path) and not save_pdf:
        print(f"可视化图片已存在: {img_path}，跳过生成")
        return
    algorithm_name = watermark.config.algorithm_name
    visualizer = get_visualizer(algorithm_name)

    try:
        if hasattr(visualizer, "page_layout_settings"):
            if figure_width is not None:
                setattr(visualizer.page_layout_settings, "max_width", int(figure_width))
            if figure_line_spacing is not None:
                setattr(visualizer.page_layout_settings, "line_spacing", int(figure_line_spacing))
    except Exception:
        pass
    img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text),
        show_text=True,
        visualize_weight=True,
        display_legend=True,
    )
    img.save(img_path)
    if save_pdf:
        base, _ = os.path.splitext(img_path)
        pdf_path = f"{base}.pdf"
        try:
            from PIL import Image as _PILImage
            scale = max(1, int(pdf_upscale))
            w, h = img.size
            pdf_img = img.resize((w * scale, h * scale), resample=_PILImage.Resampling.LANCZOS)
            pdf_img.save(pdf_path, format="PDF")
            print(f"PDF 已保存: {pdf_path}")
        except Exception:
            img.convert("RGB").save(pdf_path, format="PDF")
            print(f"PDF 已保存(降级模式): {pdf_path}")


def main(args) -> None:
    """主函数，处理命令行参数并调用可视化函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = prepare_output_dir(
        model_path=args.model_path,
        dataset_len=args.dataset_len,
        dataset_name=args.dataset_name,
    )

    # 获取文件路径
    file_paths = get_result_file_paths(output_dir, args.algorithm_name)

    # 加载生成的文本
    watermark_results = load_results(file_paths["watermark_results"])
    nowatermark_results = load_results(file_paths["no_watermark_results"])

    if not watermark_results or not nowatermark_results:
        print("错误: 未找到生成的文本结果，请先运行 generate.py 生成文本")
        return

    # 提取文本
    watermark_answer_text = watermark_results.get("answer_text", [])
    nowatermark_answer_text = nowatermark_results.get("answer_text", [])

    print(f"水印文本数量: {len(watermark_answer_text)}")
    print(f"无水印文本数量: {len(nowatermark_answer_text)}")

    if not watermark_answer_text or not nowatermark_answer_text:
        print("警告: 没有足够的有效文本进行可视化")
        return

    # 初始化水印
    print(f"初始化水印用于可视化...")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    transformers_config = TransformersConfig(
        model=None,  # 不需要加载模型，只需要tokenizer
        tokenizer=tokenizer,
        vocab_size=config.vocab_size,
        device=device,
    )
    watermark: BaseWatermark = AutoWatermark.load(
        algorithm_name=args.algorithm_name,
        algorithm_config=f"config/{args.algorithm_name}.json",
        transformers_config=transformers_config,
    )

    # 生成可视化图片
    print("创建有水印文本的可视化...")
    visualize(
        img_path=file_paths["watermark_img"],
        watermark=watermark,
        text=watermark_answer_text[args.sample_index],
        force=args.force,
        figure_width=args.figure_width,
        figure_line_spacing=args.figure_line_spacing,
        save_pdf=args.save_pdf,
        pdf_upscale=args.pdf_upscale,
    )

    print("创建无水印文本的可视化...")
    visualize(
        img_path=file_paths["nowatermark_img"],
        watermark=watermark,
        text=nowatermark_answer_text[args.sample_index],
        force=args.force,
        figure_width=args.figure_width,
        figure_line_spacing=args.figure_line_spacing,
        save_pdf=args.save_pdf,
        pdf_upscale=args.pdf_upscale,
    )

    print(f"可视化完成，图片已保存至 {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="为生成的文本创建水印可视化")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/QwQ-32B",
        help="模型路径，例如 Qwen/QwQ-32B",
    )
    parser.add_argument(
        "--algorithm-name",
        type=str,
        default="KGW",
        help="算法名称，例如 KGW, UPV, Unigram",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="c4",
        help="数据集类型",
    )
    parser.add_argument(
        "--dataset-len",
        type=int,
        default=200,
        help="数据集大小",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="要可视化的样本索引",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已存在的图片",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=800,
        help="最大文本宽度（像素），与 cs_stat 对齐",
    )
    parser.add_argument(
        "--figure-line-spacing",
        type=int,
        default=6,
        help="行间距（像素），与 cs_stat 对齐",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="同时导出 PDF（与 PNG 同名）",
    )
    parser.add_argument(
        "--pdf-upscale",
        type=int,
        default=4,
        help="PDF 导出时的放大倍数（提高清晰度）",
    )

    args = parser.parse_args()
    main(args)
