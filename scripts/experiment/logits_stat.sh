#!/bin/bash

# logits_stat.sh - 统计LLM思考阶段的logits和probs分布

# 默认参数
MODEL_PATH="Qwen/QwQ-32B"
DATASET_NAME="c4"
DATASET_LEN=3
TOKEN_RATIO=0.001
PLOT_INDICES="0 1 2"

# 帮助信息
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model-path MODEL_PATH    Model path (default: $MODEL_PATH)"
    echo "  -d, --dataset-name DATASET     Dataset name (default: $DATASET_NAME)"
    echo "  -l, --dataset-len LEN          Dataset length (default: $DATASET_LEN)"
    echo "  -r, --token-ratio RATIO        Token ratio for top-k selection (default: $TOKEN_RATIO)"
    echo "  -p, --plot-indices INDICES     Sample indices to plot, space separated (default: $PLOT_INDICES)"
    echo "  -h, --help                     Show this help message"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
    -m | --model-path)
        MODEL_PATH="$2"
        shift 2
        ;;
    -d | --dataset-name)
        DATASET_NAME="$2"
        shift 2
        ;;
    -l | --dataset-len)
        DATASET_LEN="$2"
        shift 2
        ;;
    -r | --token-ratio)
        TOKEN_RATIO="$2"
        shift 2
        ;;
    -p | --plot-indices)
        PLOT_INDICES="$2"
        shift 2
        ;;
    -h | --help)
        print_help
        ;;
    *)
        echo "Unknown option: $1"
        print_help
        ;;
    esac
done

echo "Running logits statistics experiment with the following settings:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Dataset Length: $DATASET_LEN"
echo "  Token Ratio: $TOKEN_RATIO"
echo "  Plot Indices: $PLOT_INDICES"
echo

# 运行实验
python experiment/logits_stat.py \
    --model-path "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --dataset-len "$DATASET_LEN" \
    --token-ratio "$TOKEN_RATIO" \
    --plot-indices $PLOT_INDICES

echo "Experiment completed."
