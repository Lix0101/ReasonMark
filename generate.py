import gc

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from cfg import (
    get_dataset,
    get_result_file_paths,
    get_task_prompt_builder,
    load_results,
    prepare_output_dir,
    save_results,
)
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermarkForRLLM

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()


def extract_text_from_output(
    outputs: list[RequestOutput],
) -> tuple[list[str], list[str], list[dict]]:
    """从输出中提取全文和回答文本，区分有/无</think>标签的文本"""
    text_full: list[str] = []
    text_filtered: list[str] = []
    no_think_text: list[dict] = []

    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        # 从RequestOutput中获取prompt
        prompt = output.prompt

        if "</think>" in text:
            text_full.append(text)
            text_filtered.append(text.split("</think>")[-1].strip())
        else:
            # 将无</think>标签的文本添加到no_think_text
            no_think_dict = {"id": i, "prompt": prompt, "text": text}
            no_think_text.append(no_think_dict)

    return text_full, text_filtered, no_think_text


def chat(
    model: LLM,
    sampling_params: SamplingParams,
    conversations: list[list[dict[str, str]]],
    algorithm_name: str = None,  # 添加算法名称参数
    # is_watermark: bool = False,
    # watermark: AutoWatermarkForRLLM | None = None,
) -> tuple[list[RequestOutput], list[str], list[str], list[dict]]:
    """处理文本生成，只负责生成文本并返回结果"""
    # # 设置采样参数
    # if is_watermark and watermark:
    #     sampling_params.logits_processors = [watermark]
    # elif is_watermark:
    #     print("警告：启用了水印但未提供水印处理器")

    # 文本类型标识
    text_type = (
        "有水印" if sampling_params.logits_processors is not None else "无水印"
    )

        # 根据算法确定批处理大小
    if algorithm_name in ["OURS", "OURS_Decrease"]:
        batch_size = 20  
        
        # 分批处理
        all_outputs = []
        all_text_full = []
        all_text_filtered = []
        all_no_think_text = []
        
        for i in range(0, len(conversations), batch_size):
            batch_conversations = conversations[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(conversations) + batch_size - 1) // batch_size
            
            #print(f"处理批次 {batch_num}/{total_batches} (包含 {len(batch_conversations)} 个对话)")
            
            # 执行模型生成
            outputs1: list[RequestOutput] = model.chat(
                messages=batch_conversations,
                sampling_params=sampling_params,
                use_tqdm=True,
                add_generation_prompt=True,
            )
            
            # 提取文本
            text_full, text_filtered, no_think_text = extract_text_from_output(outputs1)
            
            # 收集结果
            all_outputs.extend(outputs1)
            all_text_full.extend(text_full)
            all_text_filtered.extend(text_filtered)
            all_no_think_text.extend(no_think_text)
            
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_outputs, all_text_full, all_text_filtered, all_no_think_text
    
    else:
        # 其他算法使用标准处理方式
        # print(f"[{algorithm_name}算法] 使用标准批处理")
        
        outputs: list[RequestOutput] = model.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=True,
            add_generation_prompt=True,
        )

    # 提取文本
    text_full, text_filtered, no_think_text = extract_text_from_output(outputs)
    print(
        f"{text_type}情况下，不含 </think> 文本的占比: {len(no_think_text)}/{len(outputs)}"
    )

    return outputs, text_full, text_filtered, no_think_text


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    output_dir = prepare_output_dir(
        model_path=args.model_path,
        dataset_len=args.dataset_len,
        dataset_name=args.dataset_name,
    )

    file_paths = get_result_file_paths(output_dir, args.algorithm_name)

    # 判断结果文件是否存在
    watermark_results = load_results(file_paths["watermark_results"])
    nowatermark_results = load_results(file_paths["no_watermark_results"])

    watermark_exists = watermark_results is not None
    nowatermark_exists = nowatermark_results is not None

    if watermark_exists and nowatermark_exists:
        print("有水印和无水印文本结果文件都已存在，跳过生成步骤")
    else:
        # 获取数据集
        dataset = get_dataset(args.dataset_name, args.dataset_len, args.seed)
        prompts = dataset.prompts

        # 获取任务提示函数
        task_prompt_builder = get_task_prompt_builder(args.dataset_name)

        # 初始化模型
        print(f"正在加载模型 {args.model_path}...")
        model = LLM(
            model=args.model_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.8,
            enforce_eager=False,
            dtype=args.dtype,
            disable_custom_all_reduce=True,
            disable_log_stats=False,
            trust_remote_code=True,
            swap_space=32,
            seed=args.seed,
            quantization=args.quantization if args.quantization else None,
            tensor_parallel_size=2,
        )

        # 准备采样参数
        sampling_params = model.get_default_sampling_params()
        sampling_params.n = 1  # 生成序列数量始终为 1
        # sampling_params.logprobs = 0  # 确保能获取 logprobs
        if args.max_new_tokens is not None:
            sampling_params.max_tokens = args.max_new_tokens
        if args.min_new_tokens is not None:
            sampling_params.min_tokens = args.min_new_tokens
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
        if args.repetition_penalty is not None:
            sampling_params.repetition_penalty = args.repetition_penalty
        # print("cuda:{args.watermark_device}")
        # 加载模型配置和 tokenizer
        config = AutoConfig.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained(args.model_path),
            tokenizer=tokenizer,
            vocab_size=config.vocab_size,
            device=(f"cuda:{args.watermark_device}" if args.algorithm_name in ["OURS", "OURS_Decrease"] else device),
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_model_len,
            do_sample=True,
            no_repeat_ngram_size=3,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
        )

        # 初始化水印
        print("初始化水印...")
        watermark = AutoWatermarkForRLLM(
            algorithm_name=args.algorithm_name,
            algorithm_config=f"config/{args.algorithm_name}.json",
            transformers_config=transformers_config,
            watermark_before_think=args.watermark_before_think,
        )

        conversations = [
            [{"role": "user", "content": task_prompt_builder(prompt)}]
            for prompt in prompts
        ]

        # 处理无水印生成
        if not nowatermark_exists:
            print("开始生成无水印文本...")
            (
                _,
                nowatermark_full_text,
                nowatermark_answer_text,
                nowatermark_no_think_text,
            ) = chat(
                model=model,
                sampling_params=sampling_params,
                conversations=conversations,
                # is_watermark=False,
            )

            # 保存无水印结果
            nowatermark_results = {
                "full_text": nowatermark_full_text,
                "answer_text": nowatermark_answer_text,
                "no_think_text": nowatermark_no_think_text,
            }
            save_results(
                file_paths["no_watermark_results"], nowatermark_results
            )
            print(f"无水印文本已保存至 {file_paths['no_watermark_results']}")
        else:
            print(
                f"无水印文本结果文件已存在: {file_paths['no_watermark_results']}，跳过生成"
            )

        # 处理有水印生成
        if not watermark_exists:
            print("开始生成有水印文本...")
            sampling_params.logits_processors = [watermark]
            (
                _,
                watermark_full_text,
                watermark_answer_text,
                watermark_no_think_text,
            ) = chat(
                model=model,
                sampling_params=sampling_params,
                conversations=conversations,
                algorithm_name=args.algorithm_name,
                # is_watermark=True,
                # watermark=watermark,
            )

            # 保存有水印结果
            watermark_results = {
                "full_text": watermark_full_text,
                "answer_text": watermark_answer_text,
                "no_think_text": watermark_no_think_text,
            }
            save_results(file_paths["watermark_results"], watermark_results)
            print(f"有水印文本已保存至 {file_paths['watermark_results']}")
        else:
            print(
                f"有水印文本结果文件已存在: {file_paths['watermark_results']}，跳过生成"
            )

    print(f"全部完成，结果已保存至 {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        default=8192,
        help="模型最长上下文",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
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
        "--watermark-before-think",
        action="store_true",
    )
    parser.add_argument(
        "--watermark-device",
        type=int,
        default=2,
        help="Cuda:4 index ID for ours",
    )
    args = parser.parse_args()
    import os

    os.environ["VLLM_USE_V1"] = "0"
    main(args)
