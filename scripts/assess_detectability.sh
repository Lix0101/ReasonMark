#!/bin/bash

# assess_detectability.sh - 用于运行assess_detectability.py评估水印检测性能
#
# 可用参数:
# --algorithm: 水印算法名称，如KGW, UPV, Unigram等
# --model-path: 模型路径，如Qwen/QwQ-32B
# --dataset-name: 数据集名称，如c4, wmt16_de_en, human_eval
# --dataset-len: 数据集样本数量
# --labels: 评估指标标签，多个指标用空格分隔
# --rules: 阈值确定规则，如best或target_fpr
# --target-fpr: 目标假阳性率，默认0.01
# --reverse: 是否反转评估结果，默认false
# --attack-name: 攻击名称（可选），如Word-D, Word-S, Translation等
# --min-answer-tokens: 最小token数量，小于此值的文本将被过滤
# --max-answer-tokens: 最大token数量，大于此值的文本将被截断
# --seed: 随机种子，默认42

# 默认参数值
ALGORITHM="KGW"
MODEL_PATH="Qwen/QwQ-32B"
DATASET_NAME="c4"
DATASET_LEN=200
LABELS="TPR TNR FPR FNR P R F1 ACC"
RULES="best"
TARGET_FPR=0.01
REVERSE=""
ATTACK_NAME=""
MIN_ANSWER_TOKENS=""
MAX_ANSWER_TOKENS=""
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
    --labels)
        LABELS="$2"
        shift 2
        ;;
    --rules)
        RULES="$2"
        shift 2
        ;;
    --target-fpr)
        TARGET_FPR="$2"
        shift 2
        ;;
    --reverse)
        REVERSE="--reverse"
        shift 1
        ;;
    --attack-name)
        ATTACK_NAME="$2"
        shift 2
        ;;
    --min-answer-tokens)
        MIN_ANSWER_TOKENS="$2"
        shift 2
        ;;
    --max-answer-tokens)
        MAX_ANSWER_TOKENS="$2"
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
echo "开始评估水印检测性能..."
echo "水印算法: $ALGORITHM"
echo "模型: $MODEL_PATH"
echo "数据集: $DATASET_NAME ($DATASET_LEN 样本)"
echo "评估规则: $RULES"
echo "随机种子: $SEED"

# 构建运行命令
CMD="python assess_detectability.py \
    --algorithm \"$ALGORITHM\" \
    --model-path \"$MODEL_PATH\" \
    --dataset-name \"$DATASET_NAME\" \
    --dataset-len \"$DATASET_LEN\" \
    --rules \"$RULES\" \
    --target-fpr \"$TARGET_FPR\" \
    --seed \"$SEED\" \
    $REVERSE \
    --labels "

# 添加评估指标标签
for label in $LABELS; do
    CMD="$CMD \"$label\""
done

# 如果提供了攻击名称，则添加到命令中
if [[ -n "$ATTACK_NAME" ]]; then
    CMD="$CMD --attack-name \"$ATTACK_NAME\""
    echo "使用攻击方式: $ATTACK_NAME"
fi

# 如果提供了最小token数量，添加到命令中
if [[ -n "$MIN_ANSWER_TOKENS" ]]; then
    CMD="$CMD --min-answer-tokens \"$MIN_ANSWER_TOKENS\""
    echo "设置最小token数量: $MIN_ANSWER_TOKENS"
fi

# 如果提供了最大token数量，添加到命令中
if [[ -n "$MAX_ANSWER_TOKENS" ]]; then
    CMD="$CMD --max-answer-tokens \"$MAX_ANSWER_TOKENS\""
    echo "设置最大token数量: $MAX_ANSWER_TOKENS"
fi

# 执行命令
eval "$CMD"

echo "水印检测性能评估完成！"
