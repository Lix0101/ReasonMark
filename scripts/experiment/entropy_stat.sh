#!/bin/bash

# entropy_stat.sh - 统计LLM思考阶段的熵值分布

# 默认参数
MODEL_PATH="Qwen/QwQ-32B"
DATASET_NAME="c4"
DATASET_LEN=50
ENTROPY_THRESHOLD=0.2
PLOT_INDICES="0 1 2"
TOKENS_PER_PLOT=50

# 帮助信息
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model-path MODEL_PATH    Model path (default: $MODEL_PATH)"
    echo "  -d, --dataset-name DATASET     Dataset name (default: $DATASET_NAME)"
    echo "  -l, --dataset-len LEN          Dataset length (default: $DATASET_LEN)"
    echo "  -e, --entropy-threshold VAL    Entropy threshold percentile (default: $ENTROPY_THRESHOLD)"
    echo "  -p, --plot-indices INDICES     Sample indices to plot, space separated (default: $PLOT_INDICES)"
    echo "  -t, --tokens-per-plot COUNT    Number of tokens per entropy trend plot (default: $TOKENS_PER_PLOT)"
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
    -e | --entropy-threshold)
        ENTROPY_THRESHOLD="$2"
        shift 2
        ;;
    -p | --plot-indices)
        PLOT_INDICES="$2"
        shift 2
        ;;
    -t | --tokens-per-plot)
        TOKENS_PER_PLOT="$2"
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

echo "Running entropy statistics experiment with the following settings:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Dataset Length: $DATASET_LEN"
echo "  Entropy Threshold: $ENTROPY_THRESHOLD"
echo "  Plot Indices: $PLOT_INDICES"
echo "  Tokens per Plot: $TOKENS_PER_PLOT"
echo

# 运行实验
python experiment/entropy_stat.py \
    --model-path "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --dataset-len "$DATASET_LEN" \
    --entropy-threshold "$ENTROPY_THRESHOLD" \
    --plot-indices $PLOT_INDICES \
    --tokens-per-plot $TOKENS_PER_PLOT

echo "Experiment completed."
