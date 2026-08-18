# === Path Config ===
MODEL_NAME="Llama3.2-3B"
MODEL_PATH="Llama-3.2-3B-Instruct"
TRAIN_DATA_PATH="warmup_dataset/train_warmup_${MODEL_NAME}.json"
VAL_DATA_PATH="warmup_dataset/val_warmup_${MODEL_NAME}.json"
OUTPUT_DIR="sft_cache/${MODEL_NAME}-sft-lora-train-embedding-shot"
PROMPT_FILE_PATH="prompt.txt"

# === LoRA Config ===
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# === Training Config ===
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=5
LOGGING_STEPS=100
SAVE_STEPS=100
EVAL_STEPS=100
CUDA_DEVICES=3


LOGFILE="logs/${MODEL_NAME}-sft-lora-train-embedding-shot.log"
mkdir -p logs

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" nohup python -u train.py \
    --model_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_file_path "$PROMPT_FILE_PATH" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
> "$LOGFILE" 2>&1 &

tail -f "$LOGFILE"
