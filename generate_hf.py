import gc
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from tqdm import tqdm

from cfg import (
    CHRIST_FAMILY,
    KGW_FAMILY,
    get_dataset,
    get_result_file_paths,
    get_task_prompt_builder,
    load_results,
    prepare_output_dir,
    save_results,
)
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermarkForRLLMHF

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()


def extract_text_from_output(
    outputs: list[str],
) -> tuple[list[str], list[str], list[dict]]:
    """从输出文本中提取全文和回答文本，区分有/无</think>标签的文本"""
    text_full: list[str] = []
    text_filtered: list[str] = []
    no_think_text: list[dict] = []

    for i, text in enumerate(outputs):
        if "</think>" in text:
            text_full.append(text)
            text_filtered.append(text.split("</think>")[-1].strip())
        else:
            # 将无</think>标签的文本添加到no_think_text
            no_think_dict = {"id": i, "text": text}
            no_think_text.append(no_think_dict)

    return text_full, text_filtered, no_think_text


def chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    conversations: list[list[dict[str, str]]],
    watermark: AutoWatermarkForRLLMHF | None = None,
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """处理文本生成，只负责生成文本并返回结果"""
    # 文本类型标识
    text_type = "有水印" if watermark is not None else "无水印"
    outputs = []

    for conversation in tqdm(conversations, desc=f"生成{text_type}文本"):
        input_text: str = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )  # type: ignore
        try:
            model_inputs = tokenizer([input_text], return_tensors="pt").to(
                model.device
            )
            input_length = model_inputs.input_ids.shape[1]
            if watermark is None:
                output_ids = model.generate(
                    **model_inputs, generation_config=gen_config,#no_repeat_ngram_size=3, # type: ignore
                )
                generated_text: str = tokenizer.decode(
                    output_ids[0, input_length:], skip_special_tokens=True
                )
            else:
                watermark_instance = watermark.watermark
                if watermark_instance.config.algorithm_name in KGW_FAMILY:  # type: ignore
                    # logit-based watermark
                    assert hasattr(watermark_instance, "logits_processor")
                    output_ids = model.generate(
                        **model_inputs,  # type: ignore
                        generation_config=gen_config,
                        logits_processor=LogitsProcessorList([watermark]),
                        #no_repeat_ngram_size=3,
                    )
                    generated_text: str = tokenizer.decode(
                        output_ids[0, input_length:], skip_special_tokens=True
                    )
                elif watermark_instance.config.algorithm_name in CHRIST_FAMILY:  # type: ignore
                    # sampling-based watermark
                    assert hasattr(
                        watermark_instance, "generate_watermarked_text_rllm"
                    )
                    generated_text = (
                        watermark_instance.generate_watermarked_text_rllm(
                            input_text
                        ).removeprefix(input_text)
                    )
                else:
                    raise NotImplementedError(
                        f"Algorithm {watermark_instance.config.algorithm_name} is not configured for rllm watermarking."  # type: ignore
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[OOM] Skipped one sample at index {len(outputs)} ({text_type})")
            generated_text = ""
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"[OOM] Skipped one sample at index {len(outputs)} ({text_type})")
                generated_text = ""
            else:
                raise
        outputs.append(generated_text)

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
        dir_name="outputs-hf",
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
        
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            device_map = "auto"  # 自动多GPU分配
            print(f"使用 {num_gpus} 个GPU进行模型并行")
        else:
            device_map = device  # 单GPU模式

        # 初始化模型
        print(f"正在加载模型 {args.model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=args.dtype,
            trust_remote_code=True,
            #device_map=device_map,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        config = AutoConfig.from_pretrained(args.model_path)

        # 设置默认的 pad_token_id
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

        # 首先加载模型的默认生成配置
        gen_config = GenerationConfig.from_pretrained(args.model_path)

        # 确保设置pad_token_id
        gen_config.pad_token_id = tokenizer.pad_token_id
        gen_config.do_sample = True
        gen_config.no_repeat_ngram_size = 3
        if args.max_model_len is not None:
            gen_config.max_length = args.max_model_len
        if args.max_new_tokens is not None:
            print(f"设置 max_new_tokens = {args.max_new_tokens}")
            gen_config.max_new_tokens = args.max_new_tokens
        if args.min_new_tokens is not None:
            gen_config.min_new_tokens = args.min_new_tokens
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
        # 初始化 transformers 配置
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=config.vocab_size,
            device=(f"cuda:{args.watermark_device}" if args.algorithm_name in ["OURS"] else device),
            **gen_config.to_dict(),
        )
        print(f"TransformersConfig gen_kwargs:")
        print(f"  {transformers_config.gen_kwargs}")
        # 初始化水印
        print("初始化水印...")
        watermark = AutoWatermarkForRLLMHF(
            algorithm_name=args.algorithm_name,
            algorithm_config=f"config/{args.algorithm_name}.json",
            transformers_config=transformers_config,
            watermark_before_think=args.watermark_before_think,
        )

        conversations: list[list[dict[str, str]]] = [
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
                tokenizer=tokenizer,
                gen_config=gen_config,
                conversations=conversations,
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
            (
                _,
                watermark_full_text,
                watermark_answer_text,
                watermark_no_think_text,
            ) = chat(
                model=model,
                tokenizer=tokenizer,
                gen_config=gen_config,
                conversations=conversations,
                watermark=watermark,
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
        help="算法名称，例如 KGW, UPV, Unigram, EXP 等",
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
        default=32768,
        help="模型最长上下文",
    )
    # unused argument
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24576,
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
    # unused argument
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="presence_penalty 参数",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="repetition_penalty 参数",
    )
    parser.add_argument(
        "--watermark-before-think",
        action="store_true",
    )
    parser.add_argument(
        "--watermark-device",
        type=int,
        default=1,
        help="cuda:1 ,index ID for ours",
    )
    args = parser.parse_args()
    main(args)
