echo $1
output_dir='experiments/multi_MeTA/experiments_multi_MeTA_Kinner_8'
base_model='baffo32/decapoda-research-llama-7B-hf'
train_data2='./data/movie/train.json'
train_data='./data/book/train.json'
val_data2='./data/movie/valid.json'
val_data='./data/book/valid.json'
instruction_model='alpaca-lora-7B'
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 16
        do
            for seed in 15
            do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_multi_rec_MeTA_MAML.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --train_data_path2 $train_data2 \
                    --val_data_path $val_data \
                    --val_data_path2 $val_data2 \
                    --output_dir ${output_dir}_${seed}_${sample}\
                    --batch_size 64 \
                    --micro_batch_size 32 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r  8 \
                    --lora_alpha 16 \
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $seed \
                    --inner_adaptation_steps 8 \
                    --inner_learning_rate 1e-2 \
                    --early_stopping_patience 3 \
                    --k_shot 5
            done
        done
    done
done
