#!/bin/bash

# generate.sh - 用于运行generate.py生成有水印和无水印文本
#
# 可用参数:
# --model-path: 模型路径，如Qwen/QwQ-32B
# --algorithm-name: 水印算法名称，如KGW, UPV, Unigram等
# --dataset-name: 数据集名称，如c4, wmt16_de_en, human_eval
# --dataset-len: 数据集样本数量
# --dtype: 张量数据类型，默认auto
# --quantization: 量化方法，如awq, awq_marlin等
# --seed: 随机种子，默认42
# --max-model-len: 模型最大上下文长度
# --max-new-tokens: 最大生成token数量
# --min-new-tokens: 最小生成token数量
# --temperature: 生成温度
# --top-p: 核采样参数
# --top-k: top-k采样参数
# --min-p: 最小概率阈值
# --presence-penalty: 惩罚已出现token的参数
# --repetition-penalty: 重复惩罚参数，默认1.0
# --watermark-before-think: 在</think>标签前应用水印

# 默认参数值
MODEL_PATH="Qwen/QwQ-32B"
ALGORITHM_NAME="KGW"
DATASET_NAME="c4"
DATASET_LEN=200
DTYPE="auto"
QUANTIZATION=""
SEED=42
MAX_MODEL_LEN=32768

MAX_NEW_TOKENS=32768
MIN_NEW_TOKENS=32

# 以下采样参数默认不传递
TEMPERATURE=""
TOP_P=""
TOP_K=""
MIN_P=""
PRESENCE_PENALTY=""
REPETITION_PENALTY=""
WATERMARK_BEFORE_THINK=""

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
    --model-path)
        MODEL_PATH="$2"
        shift 2
        ;;
    --algorithm-name)
        ALGORITHM_NAME="$2"
        shift 2
        ;;
    --dataset-name)
        DATASET_NAME="$2"
        shift 2
        ;;
    --dataset-len)
        DATASET_LEN="$2"
        shift 2
        ;;
    --dtype)
        DTYPE="$2"
        shift 2
        ;;
    --quantization)
        QUANTIZATION="$2"
        shift 2
        ;;
    --seed)
        SEED="$2"
        shift 2
        ;;
    --max-model-len)
        MAX_MODEL_LEN="$2"
        shift 2
        ;;
    --max-new-tokens)
        MAX_NEW_TOKENS="$2"
        shift 2
        ;;
    --min-new-tokens)
        MIN_NEW_TOKENS="$2"
        shift 2
        ;;
    --temperature)
        TEMPERATURE="$2"
        shift 2
        ;;
    --top-p)
        TOP_P="$2"
        shift 2
        ;;
    --top-k)
        TOP_K="$2"
        shift 2
        ;;
    --min-p)
        MIN_P="$2"
        shift 2
        ;;
    --presence-penalty)
        PRESENCE_PENALTY="$2"
        shift 2
        ;;
    --repetition-penalty)
        REPETITION_PENALTY="$2"
        shift 2
        ;;
    --watermark-before-think)
        WATERMARK_BEFORE_THINK="--watermark-before-think"
        shift 1
        ;;
    *)
        echo "未知参数: $1"
        exit 1
        ;;
    esac
done

# 打印运行信息
echo "开始生成文本..."
echo "模型: $MODEL_PATH"
echo "水印算法: $ALGORITHM_NAME"
echo "数据集: $DATASET_NAME ($DATASET_LEN 样本)"

# 构建命令
CMD="python generate.py \
    --model-path \"$MODEL_PATH\" \
    --algorithm-name \"$ALGORITHM_NAME\" \
    --dataset-name \"$DATASET_NAME\" \
    --dataset-len \"$DATASET_LEN\" \
    --dtype \"$DTYPE\" \
    --seed \"$SEED\" \
    --max-model-len \"$MAX_MODEL_LEN\" \
    --max-new-tokens \"$MAX_NEW_TOKENS\" \
    --min-new-tokens \"$MIN_NEW_TOKENS\""

# 如果设置了量化方法，则添加量化参数
if [[ -n "$QUANTIZATION" ]]; then
    CMD="$CMD --quantization \"$QUANTIZATION\""
    echo "使用量化方法: $QUANTIZATION"
fi

# 只有在明确指定这些采样参数时才添加对应参数
if [[ -n "$TEMPERATURE" ]]; then
    CMD="$CMD --temperature \"$TEMPERATURE\""
fi

if [[ -n "$TOP_P" ]]; then
    CMD="$CMD --top-p \"$TOP_P\""
fi

if [[ -n "$TOP_K" ]]; then
    CMD="$CMD --top-k \"$TOP_K\""
fi

if [[ -n "$MIN_P" ]]; then
    CMD="$CMD --min-p \"$MIN_P\""
fi

if [[ -n "$PRESENCE_PENALTY" ]]; then
    CMD="$CMD --presence-penalty \"$PRESENCE_PENALTY\""
fi

if [[ -n "$REPETITION_PENALTY" ]]; then
    CMD="$CMD --repetition-penalty \"$REPETITION_PENALTY\""
fi

# 如果设置了watermark-before-think，添加该参数
if [[ -n "$WATERMARK_BEFORE_THINK" ]]; then
    CMD="$CMD $WATERMARK_BEFORE_THINK"
    echo "在</think>标签前应用水印"
fi

# 执行命令
eval "$CMD"

echo "文本生成完成！"
