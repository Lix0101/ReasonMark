#!/bin/bash

# assess_quality.sh - 用于运行assess_quality.py评估文本质量
#
# 可用参数:
# --algorithm: 水印算法名称，如KGW, UPV, Unigram等
# --model-path: 模型路径，如Qwen/QwQ-32B
# --dataset-name: 数据集名称，如c4, wmt16_de_en, human_eval
# --dataset-len: 数据集样本数量
# --openai-api-key: OpenAI API密钥（可选，用于GPT评估）
# --seed: 随机种子，默认42

# 默认参数值
ALGORITHM="KGW"
MODEL_PATH="Qwen/QwQ-32B"
DATASET_NAME="c4"
DATASET_LEN=200
OPENAI_API_KEY=""
SEED=42

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
    --algorithm)
        ALGORITHM="$2"
        shift 2
        ;;
    --model-path)
        MODEL_PATH="$2"
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
    --openai-api-key)
        OPENAI_API_KEY="$2"
        shift 2
        ;;
    --seed)
        SEED="$2"
        shift 2
        ;;
    *)
        echo "未知参数: $1"
        exit 1
        ;;
    esac
done

# 打印运行信息
echo "开始评估文本质量..."
echo "水印算法: $ALGORITHM"
echo "模型: $MODEL_PATH"
echo "数据集: $DATASET_NAME ($DATASET_LEN 样本)"
echo "随机种子: $SEED"

# 构建运行命令
CMD="python assess_quality.py \
    --algorithm \"$ALGORITHM\" \
    --model-path \"$MODEL_PATH\" \
    --dataset-name \"$DATASET_NAME\" \
    --dataset-len \"$DATASET_LEN\" \
    --seed \"$SEED\""

# 如果提供了OpenAI API密钥，则添加到命令中
if [[ -n "$OPENAI_API_KEY" ]]; then
    CMD="$CMD --openai-api-key \"$OPENAI_API_KEY\""
    echo "使用OpenAI API进行评估"
fi

# 执行命令
eval "$CMD"

echo "文本质量评估完成！"
