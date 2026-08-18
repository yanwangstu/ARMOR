# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 grpo_trainer.py
MODEL_NAME=Llama3.2-3B

CUDA_VISIBLE_DEVICES=3 nohup torchrun --nproc_per_node=1 --master_port=29501 grpo_trainer.py \
    --theme "${MODEL_NAME}"-RL \
    --train_dataset_path train_rl.json \
    --system_prompt_path prompt.txt \
    --policy_model_path WarmUp/sft_cache/"${MODEL_NAME}"-sft-lora-train-embedding-shot/final-merged \
    --reward_model_path model_cache/Qwen/Qwen3-4B \
    --model_save_path grpo_save/"${MODEL_NAME}"-RL \
    --rollout_info_save_dir rollout_info/"${MODEL_NAME}"-RL \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --batch_size 10 \
    --rollout_num 2 \
    --rollout_micro_batch_size_per_gpu 1 \
    --policy_ref_micro_batch_size_per_gpu 1 \
    --max_token_len 1500 \
    --temperature 0.85 \
    --top_p 0.95 \
    --top_k 50 \
    --model_save_steps 30 \
    --rollout_info_save_step 10 \
    > logs/grpo_trainer-"${MODEL_NAME}"-RL.log 2>&1 &