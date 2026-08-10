import json
import torch
import random
from torch.utils.data import Dataset

# 需要额外添加的 special tokens 
SPECIAL_TOKENS = ["<main-question>", "</main-question>", "<main-answer>", "</main-answer>", 
                  "<think>", "</think>", "<sub-question>", "</sub-question>",
                  "<search>", "</search>", "<doc>", "</doc>",
                  "<doc-type>", "</doc-type>", "<sub-answer>", "</sub-answer>"]


class MultiHopDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 data_path, 
                 system_prompt, 
                 paddind_truncation_length=None, 
                 add_special_tokens=True):
        """
        tokenizer: HF tokenizer
        max_length: SFT 序列最大长度 
        add_special_tokens: 是否将 SPECIAL_TOKENS 加入 tokenizer
        system_prompt: system 提示内容
        """
        self.tokenizer = tokenizer
        self.paddind_truncation_length = paddind_truncation_length
        
        self.system_prompt = system_prompt

        self.add_special_tokens = add_special_tokens
        # 如果某 special token 已经添加 不会重复添加
        if add_special_tokens:
            # 在分词器中添加 special token
            self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        
        # Get special token IDs for <doc> and </doc>
        self.doc_start_id = self.tokenizer.convert_tokens_to_ids("<doc>")
        self.doc_end_id = self.tokenizer.convert_tokens_to_ids("</doc>")

        self.ignore_index = -100

        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

    def __len__(self):
        return len(self.raw_data)

    def _assistant_content_gen(self, sample: dict):
        assistant_content = ''
        random.seed(42)
        # add think chain
        for item in sample["chain_of_thought"]:
            assistant_content += f"<think>{item["think"]}</think>"
            assistant_content += f"<sub-question>{item["sub_question"]}</sub-question>"
            assistant_content += f"<search>{str(item["retrieval"])}</search>"
            if item["retrieval"] == True:
                if item["retrieval_golden"] == False:
                    # add golden doc into item["retrieval_info"]
                    candidates = item["retrieval_info"][:2] + [item["doc"]]
                    random.shuffle(candidates)
                    item["retrieval_info"] = candidates
                for doc in item["retrieval_info"]:
                    assistant_content += f"<doc>{doc}</doc>"
                    if doc == item["doc"]:
                        assistant_content += f"<doc-type>useful</doc-type>"
                    else:
                        assistant_content += f"<doc-type>useless</doc-type>"
            assistant_content += f"<sub-answer>{item["sub_answer"]}</sub-answer>\n"

        # add main answer
        # assistant_content += "<think> Now I can answer the main question. </think>"
        assistant_content += f"<main-answer>{sample["main_answer"]}</main-answer>"
        return assistant_content

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        user_content = f"<main-question>{sample["main_question"]}</main-question>"
        assistant_content = self._assistant_content_gen(sample)

        # 构建 messages 列表（system + user + assistant）
        system_msg = {"role": "system", "content": self.system_prompt}
        user_msg = {"role": "user", "content": user_content}

        # 构造 input 部分 (system + user) 字符串 -- 使用 tokenizer 的 apply_chat_template 构造
        input_text = self.tokenizer.apply_chat_template([system_msg, user_msg], tokenize=False, add_generation_prompt=True)

        # 构造 output 部分 (assistant) 字符串
        output_text = assistant_content + self.tokenizer.eos_token

        # tokenize input 部分 和 output 部分, 得到每一个 token 的 id
        input_encode = self.tokenizer(input_text, return_attention_mask=False, return_tensors=None)
        output_encode = self.tokenizer(output_text, return_attention_mask=False, return_tensors=None)

        # Build labels for output: self.ignore_index for tokens inside <doc>...</doc> (inclusive)
        output_labels = []
        inside_doc = False
        for token_id in output_encode["input_ids"]:
            if token_id == self.doc_start_id:
                inside_doc = True
            if inside_doc:
                output_labels.append(self.ignore_index)
            else:
                output_labels.append(token_id)
            if token_id == self.doc_end_id:
                inside_doc = False

        sequence_ids = input_encode["input_ids"] + output_encode["input_ids"]
        attention_mask = [1] * len(sequence_ids)
        # 创建 labels(是否反向传播 & 反向传播使用的 token id)：input 部分为 self.ignore_index，output 部分非 doc 部分保留真实 id
        sequence_labels = [self.ignore_index] * len(input_encode["input_ids"]) + output_labels

        if self.paddind_truncation_length != None:
            # 将 sequence_ids, attention_mask, sequence_labels 截断或填充到 paddind_truncation_length
            sequence_len = len(sequence_ids)
            if sequence_len > self.paddind_truncation_length:
                sequence_ids = sequence_ids[:self.paddind_truncation_length]
                attention_mask = attention_mask[:self.paddind_truncation_length]
                sequence_labels = sequence_labels[:self.paddind_truncation_length]
            else:
                pad_len = self.paddind_truncation_length - sequence_len
                sequence_ids = sequence_ids + [self.tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                sequence_labels = sequence_labels + [self.ignore_index] * pad_len

        # 将 sequence_ids, attention_mask, sequence_labels 转为 tensor
        # sequence_ids = torch.tensor(sequence_ids, dtype=torch.long)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # sequence_labels = torch.tensor(sequence_labels, dtype=torch.long)

        return {
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "labels": sequence_labels
        }

# usage example
if __name__ == "__main__":
    from modelscope import AutoTokenizer
    model_name = "/datanfs4/wangyan/model_cache/Qwen/Qwen3-0___6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_path = "/datanfs4/wangyan/RL-RAG/WarmUp/warmup_dataset_construction/warmup_dataset/val_warmup_Qwen3-0.6B.json"
    prompt_file = '/datanfs4/wangyan/RL-RAG/WarmUp/prompt.txt'
    with open(prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    dataset_cls = MultiHopDataset(tokenizer, data_path, system_prompt)
    item = dataset_cls.__getitem__(0)

    print(item["input_ids"])
    print(item["attention_mask"])
    print(item["labels"])

    # 逐个 token 显示（保留 <doc> 等 special tokens）
    tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
    print("Tokens (with special tokens):")
    print(tokens)

    # 还原为完整文本（也保留 special tokens）
    decoded_text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
    print("\nDecoded text (with special tokens):")
    print(decoded_text)

