#!/bin/bash

# visualize.sh - 用于运行visualization.py进行文本水印可视化
#
# 可用参数:
# --model-path: 模型路径，如Qwen/QwQ-32B
# --algorithm-name: 水印算法名称，如KGW, UPV, Unigram等
# --dataset-name: 数据集名称，如c4, wmt16_de_en, human_eval
# --dataset-len: 数据集样本数量
# --sample-index: 指定要可视化的样本索引，默认为0
# --force: 强制覆盖已存在的图片

# 默认参数值
MODEL_PATH="Qwen/QwQ-32B"
ALGORITHM_NAME="KGW"
DATASET_NAME="c4"
DATASET_LEN=200
SAMPLE_INDEX=0
FORCE=""

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
    --sample-index)
        SAMPLE_INDEX="$2"
        shift 2
        ;;
    --force)
        FORCE="--force"
        shift 1
        ;;
    *)
        echo "未知参数: $1"
        exit 1
        ;;
    esac
done

# 打印运行信息
echo "开始文本水印可视化..."
echo "模型: $MODEL_PATH"
echo "水印算法: $ALGORITHM_NAME"
echo "数据集: $DATASET_NAME ($DATASET_LEN 样本)"
echo "样本索引: $SAMPLE_INDEX"

# 运行visualization.py
python visualization.py \
    --model-path "$MODEL_PATH" \
    --algorithm-name "$ALGORITHM_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-len "$DATASET_LEN" \
    --sample-index "$SAMPLE_INDEX" \
    $FORCE

echo "文本水印可视化完成！"
