import os
import argparse
# 设置可见 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import torch
import random
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from accelerate import Accelerator
from data_utils import MultiHopDataset, SPECIAL_TOKENS
from transformers import DataCollatorForSeq2Seq


class OOMSafeTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            # 调用父类方法计算 loss 并 backward
            loss = super().training_step(model, inputs, num_items_in_batch)
            return loss
        except RuntimeError as e:
            if "CUDA out of memory." in str(e):
                print("⚠️ CUDA OOM encountered in training_step. Skipping this step.")
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device=model.device)
            else:
                raise e
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("⚠️ CUDA OOM encountered in prediction_step. Skipping this batch.")
                torch.cuda.empty_cache()
                # 返回 (loss=None, logits=None, labels=None)
                return (None, None, None)
            else:
                raise e


class WarmUp_Trainer:
    # === init ===
    def __init__(
        self,
        model_path: str,
        train_data_path: str,
        val_data_path: str,
        output_dir: str,
        prompt_file_path: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        logging_steps: int,
        save_steps: int,
        eval_steps: int,
        add_special_tokens: bool = True,
        seed: int = 42,
        lora_target_modules: list = ["q_proj", "v_proj"],
    ):
        set_seed(seed)
        self.add_special_tokens = add_special_tokens
        self.accelerator = Accelerator()

        # 路径配置
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.output_dir = output_dir

        # Prompt 配置
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

        # LoRA 配置
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        # 训练参数
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps

    # === 加载 tokenizer 和 LLM ===
    def _load_tokenizer_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print("Initial tokenizer length:", len(self.tokenizer))
        if self.add_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
            print("Tokenizer length after add special tokens:", len(self.tokenizer))


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": self.accelerator.local_process_index} # 模型加载在当前进程分配的 GPU 上
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    # === 加载 Dataset, 添加 Tokenizer special token===
    def _load_dataset(self):
        # 设置 train_dataset & val_dataset
        self.train_dataset = MultiHopDataset(
            tokenizer=self.tokenizer,
            data_path=self.train_data_path,
            system_prompt=self.system_prompt,
            add_special_tokens=False
        )
        self.val_dataset = MultiHopDataset(
            tokenizer=self.tokenizer,
            data_path=self.val_data_path,
            system_prompt=self.system_prompt,
            add_special_tokens=False
        )

    def model_train(self):
        self._load_tokenizer_model()
        self._load_dataset()
        
        # 添加新 token 后需要调整 embedding 和 LLM head 
        self.model.resize_token_embeddings(len(self.tokenizer))

        # === 准备模型用于 LoRA ===
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Print model Info
        if self.accelerator.is_local_main_process:
            self.model.print_trainable_parameters()
            print(self.model)

        # === 训练参数设置 ===
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True, # 在训练结束后自动加载训练期间在验证指标上最好的 checkpoint 到 model，并且需要正确设置 metric_for_best_model 和 greater_is_better
            metric_for_best_model="eval_loss",
            greater_is_better=False, # 指定 metric_for_best_model 是否越大越好 （eval_loss 越小越好， accuracy/bleu 越大越好）
            warmup_ratio=0.1, # 前 10% 训练 step 学习率线性升到初始设置的 lr
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none", # 控制是否向实验追踪平台报告（例如 wandb、tensorboard、mlflow 等）
            remove_unused_columns=False,
            dataloader_num_workers=4, # num_workers = 0 → 数据在主进程里处理，训练会被阻塞等待数据，GPU 会空闲，显存利用低 num_workers > 0 → 数据并行预处理，主进程只负责调度，GPU 更高效利用
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            label_names=["labels"]
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt"
        )

        trainer = OOMSafeTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator
        )

        # === 开始训练 ===
        if self.accelerator.is_local_main_process:
            print("🚀 Starting LoRA training...")
        trainer.train()

        # === 保存 adapter ===
        if self.accelerator.is_local_main_process:
            # trainer.save_model(os.path.join(self.output_dir, "final"))
            trainer.model.save_pretrained(os.path.join(self.output_dir, "final"))
            self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))
            print("✅ LoRA adapter and tokenizer saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="WarmUp SFT Training with LoRA")

    # === 路径配置 ===
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the training dataset (JSON)")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to the validation dataset (JSON)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--prompt_file_path", type=str, required=True,
                        help="Path to the system prompt file")

    # === LoRA 配置 ===
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")

    # === 训练参数 ===
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Total number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X update steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every X steps")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(f"Current process PID: {os.getpid()}")
    args = parse_args()
    print(f"Current args")
    print(vars(args))

    trainer = WarmUp_Trainer(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        prompt_file_path=args.prompt_file_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    trainer.model_train()
