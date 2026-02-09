import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
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

from watermark.utils.text_filters import should_filter_by_pos


def stack_logits(outputs: GenerateDecoderOnlyOutput) -> torch.Tensor:
    logits = torch.stack(outputs.logits)  # [gen_len, batch, V]
    logits = logits.squeeze(1)  # [gen_len, V]
    return logits


def compute_cs_scores(
    think_tokens: torch.Tensor,
    think_logits: torch.Tensor,
    beta: float = 1.0,
    top_k_candidates: int = 10,
) -> torch.Tensor:
    """
    Compute CS scores over the vocabulary given think segment.
    CS(w) = GCC(w) * log(1 + CPS(w))
    Ported to avoid coupling with watermark.OURS classes.
    """
    device = think_logits.device
    think_logits = think_logits.float()  # [N, V]
    probs = torch.softmax(think_logits, dim=-1)  # [N, V]
    N, vocab_size = probs.shape

    # 1) lambda weights via KL(p_t || p_{t-1})
    lambda_weights = torch.zeros(N, device=device, dtype=probs.dtype)
    if N > 1:
        for i in range(1, N):
            lambda_weights[i] = torch.sum(
                probs[i] * (torch.log(probs[i] + 1e-10) - torch.log(probs[i - 1] + 1e-10))
            )

    # 2) alpha matrix via cosine similarity over probs
    probs_norm = torch.nn.functional.normalize(probs, dim=-1, eps=1e-10)
    similarities = torch.mm(probs_norm, probs_norm.T)  # [N, N]
    upper_tri_mask = torch.triu(torch.ones(N, N, device=device, dtype=probs.dtype), diagonal=1)
    sim_upper = similarities * upper_tri_mask
    row_sum = sim_upper.sum(dim=-1, keepdim=True) + 1e-10
    alpha_matrix = sim_upper / row_sum  # [N, N]

    # 3) B matrix and GCC over vocab (streaming by chunks)
    B = lambda_weights.unsqueeze(1) * alpha_matrix  # [N, N]
    gcc_scores = torch.zeros(vocab_size, device=device, dtype=probs.dtype)
    chunk_size = 4096
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        idx = torch.arange(start, end, device=device)
        P_chunk = probs[:, idx]
        BP_chunk = torch.mm(B, P_chunk)
        gcc_chunk = (P_chunk * BP_chunk).sum(dim=0)
        gcc_scores[idx] = gcc_chunk

    # 4) CPS streaming on top-k
    _, top_k_indices = torch.topk(think_logits, top_k_candidates, dim=-1)  # [N, k]

    # Precompute surprise and competition
    generated_probs = probs[torch.arange(N, device=device), think_tokens]
    denom = -torch.log(generated_probs.clamp_min(1e-12))
    surprise_rewards = 1.0 / torch.clamp(denom, min=1e-6)

    top2_vals, _ = torch.topk(think_logits, 2, dim=1)
    delta_selected = top2_vals[:, 0] - top2_vals[:, 1]
    comp_selected = torch.exp(-beta * delta_selected)

    future_counts = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    cps_scores = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    for i in range(N - 1, -1, -1):
        S_inv = surprise_rewards[i]
        t_i = int(think_tokens[i].item())
        cps_scores[t_i] += S_inv * comp_selected[i] * future_counts[t_i].float()

        tk = top_k_indices[i]
        mask_other = tk != t_i
        tk_other = tk[mask_other]
        if tk_other.numel() > 0:
            logits_candidate = think_logits[i, tk_other]
            delta = torch.abs(think_logits[i, t_i] - logits_candidate)
            comp_other = torch.exp(-beta * delta)
            cps_scores[tk_other] += S_inv * comp_other * future_counts[tk_other].float()

        future_counts[tk] += 1

    cs_scores = gcc_scores * torch.log(1 + cps_scores)
    return cs_scores


def plot_text_cs_with_visualizer(
    prompt: str,
    token_scores: list[float],
    token_texts: list[str],
    plot_name: str,
    output_dir: str,
    red_indices: Optional[list[int]] = None,
    blue_indices: Optional[list[int]] = None,
    max_width: int = 1000,
    line_spacing: int = 6,
    pdf_upscale: int = 4,
    add_think_marker: bool = True,
    think_label: str = "</think>",
):
    font_settings = FontSettings()
    # 控制整体宽高比例与行距，以匹配论文版式
    page_layout_settings = PageLayoutSettings(
        max_width=max_width,
        line_spacing=line_spacing,
    )
    legend_settings = ContinuousLegendSettings()
    color_scheme = ColorSchemeForContinuousVisualization()

    # normalize scores to [0, 1]
        # normalize scores to [0, 1] with log compress + percentile clip
    if token_scores:
        vals = np.array(token_scores, dtype=np.float32)
        vals = np.log1p(vals)  # 压缩长尾：log(1+x)
        pos = vals[vals > 0]
        if pos.size >= 5:
            lo, hi = np.percentile(pos, [5, 95])
        else:
            lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        vals = np.clip(vals, lo, hi)
        normalized = [0.5 if hi == lo else float((v - lo) / (hi - lo)) for v in vals]
    else:
        normalized = []

    data = DataForVisualization(decoded_tokens=token_texts, highlight_values=normalized)
    visualizer = ContinuousVisualizer(
        color_scheme=color_scheme,
        font_settings=font_settings,
        page_layout_settings=page_layout_settings,
        legend_settings=legend_settings,
    )
    image = visualizer.visualize(data=data, show_text=True, visualize_weight=False, display_legend=True)

    # Draw boxes approximating layout used by ContinuousVisualizer (same as entropy_stat)
    from PIL import ImageDraw

    # 计算 token 布局与绘制红/蓝框、思考结束标记
    draw = ImageDraw.Draw(image)
    boxes = {}
    line_height = font_settings.font_size + page_layout_settings.line_spacing
    current_line = 0
    current_x = page_layout_settings.margin_l
    for i, token in enumerate(token_texts):
        bbox = font_settings.font.getbbox(token)
        token_width = bbox[2] - bbox[0]
        if current_x + token_width > page_layout_settings.max_width:
            current_line += 1
            current_x = page_layout_settings.margin_l
        token_y = page_layout_settings.margin_t + current_line * line_height
        boxes[i] = (
            current_x,
            token_y,
            current_x + token_width,
            token_y + font_settings.font_size,
        )
        current_x += token_width + page_layout_settings.token_spacing

    if red_indices:
        for idx in red_indices:
            if idx in boxes:
                x1, y1, x2, y2 = boxes[idx]
                draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(255, 0, 0), width=2)

    if blue_indices:
        for idx in blue_indices:
            if idx in boxes:
                x1, y1, x2, y2 = boxes[idx]
                draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(0, 0, 255), width=2)

    # 在思考段末尾标注 </think>（使用归一化值0对应的颜色）
    if add_think_marker and len(token_texts) > 0:
        last_idx = len(token_texts) - 1
        if last_idx in boxes:
            x1, y1, x2, y2 = boxes[last_idx]
            label_bbox = font_settings.font.getbbox(think_label)
            label_w = label_bbox[2] - label_bbox[0]
            next_x = x2 + page_layout_settings.token_spacing
            next_line = current_line
            # 若超出行宽则换行（不扩展画布，保持原逻辑）
            if next_x + label_w > page_layout_settings.max_width:
                next_line += 1
                next_x = page_layout_settings.margin_l
            label_y = page_layout_settings.margin_t + next_line * line_height
            zero_color = color_scheme.get_color_from_axis(0.0)
            draw.text((next_x, label_y), think_label, fill=zero_color, font=font_settings.font)

    os.makedirs(output_dir, exist_ok=True)
    # save PNG
    png_path = os.path.join(output_dir, plot_name)
    image.save(png_path)
    # also save PDF (upscale raster for sharper appearance in PDF viewers)
    base, _ = os.path.splitext(plot_name)
    pdf_path = os.path.join(output_dir, f"{base}.pdf")
    try:
        from PIL import Image
        scale = max(1, int(pdf_upscale))  # upscale factor for PDF clarity
        w, h = image.size
        pdf_image = image.resize((w * scale, h * scale), resample=Image.Resampling.LANCZOS)
        pdf_image.save(pdf_path, format="PDF")
    except Exception:
        # fallback: convert to RGB explicitly and save
        image.convert("RGB").save(pdf_path, format="PDF")


def chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    prompt: str,
) -> dict[str, Any]:
    conversation = [{"role": "user", "content": prompt}]
    input_text: str = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )  # type: ignore

    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]

    outputs: GenerateDecoderOnlyOutput = model.generate(
        **model_inputs,  # type: ignore
        generation_config=gen_config,
        tokenizer=tokenizer,
    )

    generated_token_ids = outputs.sequences[0, input_length:]
    logits = stack_logits(outputs)

    return {
        "token_ids": generated_token_ids.tolist(),
        "logits": logits,
        "prompt": prompt,
        "text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
    }


def analyze_cs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    prompts: list[str],
    output_dir: str,
    plot_indices: list[int],
    k_red: int = 10,
    k_blue: int = 10,
    beta: float = 1.0,
    top_k_candidates: int = 10,
) -> dict[str, Any]:
    results = []

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    think_end_tid = tokenizer.encode("</think>", add_special_tokens=False)
    think_end_tid = think_end_tid[-1] if len(think_end_tid) > 0 else None

    for i, prompt in enumerate(prompts):
        out = chat(model, tokenizer, gen_config, prompt)
        token_ids_all: list[int] = out["token_ids"]
        logits_all: torch.Tensor = out["logits"]  # [gen_len, V]

        # cut at </think>
        end_pos = len(token_ids_all)
        if think_end_tid is not None and think_end_tid in token_ids_all:
            end_pos = token_ids_all.index(think_end_tid)

        think_ids = token_ids_all[:end_pos]
        think_logits = logits_all[:end_pos]

        if len(think_ids) == 0:
            results.append({"prompt": prompt, "text": out["text"], "critical_tokens_texts": [], "secondary_tokens_texts": []})
            continue

        think_tokens = torch.tensor(think_ids, device=think_logits.device, dtype=torch.long)
        cs_scores = compute_cs_scores(think_tokens, think_logits, beta=beta, top_k_candidates=top_k_candidates)  # [V]

        # map scores to occurrence positions
        token_texts = [tokenizer.decode([tid]) for tid in think_ids]
        token_scores = [float(cs_scores[tid].item()) for tid in think_ids]

        # rank unique tids by CS desc
        unique_tids = list(dict.fromkeys(think_ids))
        unique_tids_sorted = sorted(unique_tids, key=lambda t: float(cs_scores[t].item()), reverse=True)
        # exclude whitespace-only tokens explicitly, then apply POS/stopword filter
        def _is_content_token(tok_text: str) -> bool:
            return bool(tok_text.strip()) and (not should_filter_by_pos(tok_text))
        filtered_ids = [
            tid for tid in unique_tids_sorted
            if _is_content_token(tokenizer.decode([tid]))
        ]

        red_ids = set(filtered_ids[:k_red])
        blue_ids = set(filtered_ids[k_red:k_red + k_blue])

        red_indices = [idx for idx, tid in enumerate(think_ids) if tid in red_ids]
        blue_indices = [idx for idx, tid in enumerate(think_ids) if tid in blue_ids]

        # visualization for selected samples
        if i in plot_indices:
            plot_text_cs_with_visualizer(
                prompt=prompt,
                token_scores=token_scores,
                token_texts=token_texts,
                plot_name=f"text_cs_{i}.png",
                output_dir=plots_dir,
                red_indices=red_indices,
                blue_indices=blue_indices,
                max_width=args.figure_width,
                line_spacing=args.figure_line_spacing,
                pdf_upscale=args.pdf_upscale,
                add_think_marker=True,
                think_label="</think>",
            )

        results.append({
            "prompt": prompt,
            "text": out["text"],
            "critical_tokens_texts": list(dict.fromkeys([tokenizer.decode([tid]) for tid in red_ids])),
            "secondary_tokens_texts": list(dict.fromkeys([tokenizer.decode([tid]) for tid in blue_ids])),
        })

    return {"per_request_stat": results}


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    output_dir = prepare_output_dir(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_len=args.dataset_len,
        dir_name="outputs-exp",
    )

    dataset = get_dataset(args.dataset_name, args.dataset_len, args.seed)
    prompts = dataset.prompts
    task_prompt_builder = get_task_prompt_builder(args.dataset_name)
    prompts = [task_prompt_builder(p) for p in prompts]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    gen_config = GenerationConfig.from_pretrained(args.model_path)
    gen_config.output_logits = True
    gen_config.return_dict_in_generate = True
    gen_config.pad_token_id = tokenizer.pad_token_id
    # gen_config.no_repeat_ngram_size = 3
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

    print("开始基于 CS 分数的可视化分析...")
    results = analyze_cs(
        model=model,
        tokenizer=tokenizer,
        gen_config=gen_config,
        prompts=prompts,
        output_dir=output_dir,
        plot_indices=args.plot_indices,
        k_red=args.cs_red_k,
        k_blue=args.cs_blue_k,
        beta=args.beta,
        top_k_candidates=args.top_k_candidates,
    )

    import json

    class CompactJSONEncoder(json.JSONEncoder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
        def encode(self, obj):
            if isinstance(obj, list):
                return "[" + ", ".join(self.encode(it) for it in obj) + "]"
            return super().encode(obj)

    results_path = os.path.join(output_dir, "think_text_cs.json")
    with open(results_path, "w") as f:
        json.dump(results, f, cls=CompactJSONEncoder, indent=2)

    print(f"分析结果已保存至 {results_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/QwQ-32B")
    parser.add_argument(
        "--dataset-name", type=str, default="c4",
        choices=["c4", "wmt16_de_en", "wmt19_zh_en", "human_eval", "gsm8k", "cnn_dailymail","aime_2025"],
    )
    parser.add_argument("--dataset-len", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--plot-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cs-red-k", type=int, default=10)
    parser.add_argument("--cs-blue-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--top-k-candidates", type=int, default=10)
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential", "cpu"],
        help="Device mapping strategy for from_pretrained; 'auto' will shard across multiple GPUs if available.",
    )
    # figure controls
    parser.add_argument("--figure-width", type=int, default=900, help="Max text width in pixels")
    parser.add_argument("--figure-line-spacing", type=int, default=8, help="Line spacing in pixels")
    parser.add_argument("--pdf-upscale", type=int, default=4, help="Upscale factor for PDF raster clarity")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


#python -m experiment.cs_stat  --model-path /root/autodl-tmp/qwen3_32B    --dataset-name wmt16_de_en   --dataset-len 5   --plot-indices 0 1 2 3 4   --cs-red-k 10   --cs-blue-k 10