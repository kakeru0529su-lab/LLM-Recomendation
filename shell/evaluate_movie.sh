# 使用方法: bash shell/evaluate_movie.sh <GPU_ID> <checkpoint_path>
# 例: bash shell/evaluate_movie.sh 0,1 ./experiments/multi_MeTA/experiments_multi_MeTA_5_64/checkpoint-160

CUDA_ID=$1
checkpoint_path=$2
base_model='baffo32/decapoda-research-llama-7B-hf'
test_data='./data/movie/test.json'

if [ -z "$checkpoint_path" ]; then
    echo "Usage: bash shell/evaluate_movie.sh <GPU_ID> <checkpoint_path>"
    echo "Example: bash shell/evaluate_movie.sh 0,1 ./experiments/multi_MeTA/experiments_multi_MeTA_5_64/checkpoint-160"
    exit 1
fi

# チェックポイントパスから親ディレクトリとステップ番号を取得
parent_dir=$(dirname "$checkpoint_path")
checkpoint_name=$(basename "$checkpoint_path")

# music用のディレクトリを作成
result_dir="${parent_dir}/results_movie_0"
mkdir -p "$result_dir"

# resultファイル名を生成
result_file="${result_dir}/${checkpoint_name}.json"

if [ -d "$checkpoint_path" ]; then
    echo "Evaluating: $checkpoint_path"
    echo "MAML Adaptation: ENABLED"
    echo "  - Learning rate: 1e-2 (same as training)"
    echo "  - Steps: 10 (same as training)"
    echo "  - Support size: 5 (same as training k_shot)"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate_maml_my.py \
        --base_model $base_model \
        --lora_weights $checkpoint_path \
        --test_data_path $test_data \
        --result_json_data "$result_file" \
        --use_adaptation=False \
        --support_size=5 \
        --inner_steps=10 \
        --inner_lr=1e-2

    echo "Result saved to: $result_file"
else
    echo "Checkpoint not found: $checkpoint_path"
    exit 1
fi
