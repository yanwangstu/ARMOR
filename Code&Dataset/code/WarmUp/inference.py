import os
import argparse
# 设置可见 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import chromadb
import random
random.seed(42)
import sys
from retriever import e5_Retriever

def print_cuda_mem(tag):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{tag}] allocated={allocated:.1f}MB, reserved={reserved:.1f}MB, max={max_allocated:.1f}MB")



def interleave_flatten(lst):
    max_len = max(len(sub) for sub in lst)
    flat = []

    for i in range(max_len):
        for sub in lst:
            if i < len(sub):
                flat.append(sub[i])
    return flat


class stop_on_tokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        """
        stop_token_ids: list[int]，表示要匹配的 token 序列，比如 [123, 456, 789]
        """
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        """
        input_ids: 当前批次生成的 token 序列 (batch_size, seq_len)
        scores: 当前 step 的 logits，可忽略
        """
        seq_len = input_ids.shape[1]
        pattern_len = len(self.stop_token_ids)
        if seq_len < pattern_len:
            return False  # 序列太短，不可能匹配

        # 取最后 stop_len 个 token
        last_tokens = input_ids[0, -pattern_len:].tolist()

        # 检查是否完全匹配
        return last_tokens == self.stop_token_ids


class trained_model_inference:
    def __init__(self, 
                 base_model_path: str, 
                 lora_adapter_path: str, 
                 prompt_file_path: str,
                 e5_model_path: str,
                 retriever_topk: int = 3,
                 global_retrieve: bool = False,
                 do_sample: bool = False
                 ) -> None:
        
        # record the model math
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path

        # init the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
        print(f"Tokenizer Loaded")
        
        # Prompt 配置
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

        # set additional stop tokens
        self.stop_tokens_search = ["<search>", "True", "</search>"]
        self.stop_token_ids_search = self.tokenizer.convert_tokens_to_ids(self.stop_tokens_search)
        self.stop_tokens_doctype = ["</doc-type>"]
        self.stop_token_ids_doctype = self.tokenizer.convert_tokens_to_ids(self.stop_tokens_doctype)
        self.stopping_criteria_search=StoppingCriteriaList([stop_on_tokens(self.stop_token_ids_search)])
        self.stopping_criteria_doc_type=StoppingCriteriaList([stop_on_tokens(self.stop_token_ids_doctype)])

        # init the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print_cuda_mem("after base model load")
        self.model.resize_token_embeddings(len(self.tokenizer))


        # add lora para on the base model
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        print_cuda_mem("after lora load")

        # merge lora para into base model
        self.model = self.model.merge_and_unload()
        print_cuda_mem("after merge_and_unload")
        print(f"Model Loaded")
        
        # 初始化 chromadb
        self.wiki_chroma_file_dic = os.path.join('/datanfs4/wangyan/RL-RAG/DatasetConstruction/DocPool', '2WikiMultiHopQA', 'train_set')
        self.wiki_client = chromadb.PersistentClient(path=self.wiki_chroma_file_dic)
        self.wiki_collection = self.wiki_client.get_or_create_collection("doc")

        self.musi_chroma_file_dic = os.path.join('/datanfs4/wangyan/RL-RAG/DatasetConstruction/DocPool', 'MusiQue', 'train_set')
        self.musi_client = chromadb.PersistentClient(path=self.musi_chroma_file_dic)
        self.musi_collection = self.musi_client.get_or_create_collection("doc")

        self.hotpot_chroma_file_dic = os.path.join('/datanfs4/wangyan/RL-RAG/DatasetConstruction/DocPool', 'HotpotQA', 'train_set')
        self.hotpot_client = chromadb.PersistentClient(path=self.hotpot_chroma_file_dic)
        self.hotpot_collection = self.hotpot_client.get_or_create_collection("doc")
        self.collections = {
            "2WikiMultiHopQA": self.wiki_collection,
            "MusiQue": self.musi_collection,
            "HotpotQA": self.hotpot_collection
        }
        print(f"Chroma Database Loaded")

        self.retriever_topk = retriever_topk
        self.global_retrieve = global_retrieve
        self.do_sample = do_sample
        print(f"Global Retrieve: {global_retrieve}")
        print(f"Do Sample: {do_sample}")

        # 初始化编码器
        self.e5 = e5_Retriever(e5_model_path)

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def _retrieve_docs(
            self, 
            sub_question: str,  
            origin_dataset: str, 
            origin_sample_index: int,
            noise_ratio: float|None = None,
            golden_doc: list[list[str]]|None = None
        ) -> list[str]:

        sub_question_text=["query: " + sub_question]
        sub_question_embedding = self.e5.embedding_generation(sub_question_text).tolist()

        collection = self.collections[origin_dataset]

        if noise_ratio == None or golden_doc == None:
            if self.global_retrieve:
                result = collection.query(
                    query_embeddings=sub_question_embedding,
                    n_results=self.retriever_topk,
                )
            else:
                result = collection.query(
                    query_embeddings=sub_question_embedding,
                    n_results=self.retriever_topk,
                    where={"origin_sample_index": origin_sample_index}
                )
                
            documents = result["documents"][0]
            # 过滤掉空字符串或全是空白的条目
            filtered_documents = [doc for doc in documents if doc.strip()]
            if len(filtered_documents)==0:
                    print("⚠️ WARNING: 0 retreival document find!")
            return filtered_documents, ["retrieved"] * len(filtered_documents)
        
        else: 
            if int(noise_ratio*self.retriever_topk)== noise_ratio*self.retriever_topk:
                noise_num = int(noise_ratio*self.retriever_topk)
            else:
                raise ValueError("Noise ratio * retriever_topk must be an integer.")
            golden_num = self.retriever_topk - noise_num

            retrieved_docs = []

            golden_doc = interleave_flatten(golden_doc)

            # construct golden docs
            if len(golden_doc) < golden_num:
                raise ValueError("Provided golden docs are less than required number.")
            
            # construct noisy docs
            if self.global_retrieve:
                result = collection.query(
                    query_embeddings=sub_question_embedding,
                    n_results=noise_num+5,
                )
            else:
                result = collection.query(
                    query_embeddings=sub_question_embedding,
                    n_results=noise_num+5,
                    where={"origin_sample_index": origin_sample_index}
                )
            ret_docs = result["documents"][0]
            
            # 过滤掉与 golden_doc 重复的部分
            print(f"Before Filter {len(ret_docs)} Candidate Noisy Docs: ", ret_docs)
            golden_docs_no_space = [re.sub(r'\s+', '', d) for d in golden_doc]
            candidate_noisy_docs = [
                doc for doc in ret_docs
                if re.sub(r'\s+', '', doc)  # 删除所有空白
                and re.sub(r'\s+', '', doc) not in golden_docs_no_space
            ]
            candidate_golden_docs = [
                doc for doc in ret_docs
                if re.sub(r'\s+', '', doc)  # 删除所有空白
                and re.sub(r'\s+', '', doc) in golden_docs_no_space
            ]
            print(f"After Filter {len(candidate_noisy_docs)} Candidate Noisy Docs: ", candidate_noisy_docs)
            
            selected_noisy_docs = candidate_noisy_docs[:noise_num]
            selected_golden_doc = candidate_golden_docs[:golden_num]
            if len(selected_golden_doc) < golden_num:
                for doc in golden_doc:
                    if re.sub(r'\s+', '', doc) not in [re.sub(r'\s+', '', d) for d in selected_golden_doc]:
                        selected_golden_doc.append(doc)
                    if len(selected_golden_doc) == golden_num:
                        break
                
            retrieved_docs = selected_noisy_docs + selected_golden_doc
            print("Ret Docs Num: ", len(retrieved_docs))
            if len(retrieved_docs) == 0:
                print("⚠️ WARNING: 0 retreival document find!")
                return retrieved_docs, []
            labels = ["noisy"] * len(selected_noisy_docs) + ["golden"] * len(selected_golden_doc)
            combined = list(zip(retrieved_docs, labels))
            random.shuffle(combined)
            shuffled_docs, shuffled_labels = zip(*combined)

            return list(shuffled_docs), list(shuffled_labels)

    
    def _sub_question_extraction(self, output_text: str): 
        """ 
        从输出文本中提取所有符合 <sub-question>...</sub-question> <search>True</search> 的子问题， 
        返回末尾匹配内容（去除首尾空白）；若无匹配，返回 None。 
        """ 
        pattern = r"<sub-question>(.*?)</sub-question>" 
        matches = re.findall(pattern, output_text, re.DOTALL) 
        if matches: 
            last_sub_question = matches[-1]
            full_pattern = re.escape(f"<sub-question>{last_sub_question}</sub-question>") + r"\s*<search>True</search>\Z"
            if re.search(full_pattern, output_text, re.DOTALL):
                return last_sub_question.strip() 
        return None
    
    def _main_answer_extraction(self, output_text: str):
        """
        从输出文本中提取所有符合 <main-answer>...</main-answer> 的子问题，
        返回末尾匹配内容（去除首尾空白）；若无匹配，返回 None。
        """
        pattern = r"<main-answer>(.*?)</main-answer>"
        matches = re.findall(pattern, output_text, re.DOTALL)
        end_token = self.tokenizer.eos_token
        if matches:
            last_main_answer = matches[-1]
            full_pattern = re.escape(f"<main-answer>{last_main_answer}</main-answer>") + rf"\s*{re.escape(end_token)}\Z"
            # full_pattern = re.escape(f"<main-answer>{last_main_answer}</main-answer>") + r"\s*<\|im_end\|>\Z"
            if re.search(full_pattern, output_text, re.DOTALL):
                return last_main_answer.strip()
        return None
    
    def _inference_special_token_ended(self, input: dict[str, torch.Tensor], stopping_criteria):
        """
        1. 当 LLM 输出 <search>True</search> 停止输出 返回对应的 sub-question 与 output token id
        2. 当 LLM 输出 <doc-type>True</doc-type> 停止输出 返回 output token id
        3. 当 LLM 输出 eos.token 停止输出 返回 output token id
        input shape:
        {
            'input_ids': tensor([[...]]),
            'attention_mask': tensor([[...]])
        }
        """
        output_ids = self.model.generate(
            **input,
            max_new_tokens=1024,
            do_sample=self.do_sample,
            stopping_criteria=stopping_criteria
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # stop because generate eos_token
        if output_ids[0][-1].item() == self.tokenizer.eos_token_id:
            return {"end_token": "end"}, output_ids

        # stop because generate <search>True</search> 提取 sub-question
        elif output_ids[0][-1].item() == self.stop_token_ids_search[-1]:
            sub_question = self._sub_question_extraction(output_text)
            return {"end_token": "search-true", "sub_question": sub_question}, output_ids
        
        # stop because generate </doc-type>
        elif output_ids[0][-1].item() == self.stop_token_ids_doctype[-1]:
            return {"end_token": "doc-type"}, output_ids
        
        # exceed max length
        else:
            return {"end_token": None}, output_ids

    def inference_RAG(
            self, 
            main_question: str, 
            origin_dataset: str, 
            origin_sample_index: int,
            noise_ratio: str = None, 
            # golden_doc 为 子问题--golden doc 列表 的字典
            golden_doc: list[str]=None
        ):
        """
        返回 error_occure 原始 output token, output text 以及 main-answer
        """
        user_content = f"<main-question>{main_question}</main-question>"
        message = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 构造 initial input 部分 (system + user) 字符串 -- 使用 tokenizer 的 apply_chat_template 构造
        input_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # print(input_text)
        input = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        # input_ids = input["input_ids"][0]
        # decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        # print(decoded_input)

        format_error = False
        doc_list = []
        doc_history = []
        stopping_criteria = self.stopping_criteria_search

        while True:
            info, output_ids = self._inference_special_token_ended(input, stopping_criteria)
            # stop because max token length
            if info["end_token"] ==  None:
                final_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
                final_token_list = self.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)
                format_error = True
                break
            # stop because generate eos_token
            if info["end_token"] ==  "end":
                final_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
                final_token_list = self.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)
                break
            # stop because generate<search>True</search>
            if info["end_token"] == "search-true":
                # sub_question parse failed
                if info["sub_question"] == None:
                    final_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
                    final_token_list = self.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)
                    format_error = True
                    break
                # sub_question parse successed
                else:
                    # init the doc list of current sub question
                    doc_list, doc_label = self._retrieve_docs(info["sub_question"],  origin_dataset, origin_sample_index, noise_ratio, golden_doc)
                    doc_history.append({
                        "sub_question": info["sub_question"],
                        "retrieved_docs": doc_list.copy(),
                        "doc_label": doc_label
                    })
                    
            # generate <search>True</search> or </doc-type>
            if len(doc_list)<=1:
                stopping_criteria = self.stopping_criteria_search
            else: 
                stopping_criteria = self.stopping_criteria_doc_type  
            
            # add a doc in the end of the output and continue to generate if possible
            doc_str = f'<doc>{doc_list.pop(0)}</doc>' if doc_list else None

            full_input_ids = output_ids
            full_attention_mask = torch.ones_like(output_ids)

            # 计算 input_ids & attention_mask
            if doc_str is not None:
                doc_input_ids = self.tokenizer(doc_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
                full_input_ids = torch.cat([full_input_ids, doc_input_ids], dim=1)
                full_attention_mask = torch.cat([full_attention_mask, torch.ones_like(doc_input_ids)], dim=1)
            input = {
                    "input_ids": full_input_ids,
                    "attention_mask": full_attention_mask,
                }
        

        # break from sub_question parse failed
        if format_error == True:
            if info["end_token"] ==  None:
                error_info = "token length exceed"
            else:
                error_info = "sub_question_parse_failed"
            return {
                "format_error": True,
                "error_info": error_info,
                "final_output_text": final_text, 
                "final_token_list": final_token_list, 
                "main_answer": None,
                "doc_history": doc_history
            }
        # break from info == {"end_token": "end"}:
        else:
            main_answer = self._main_answer_extraction(final_text)
            # format error occures
            if main_answer == None:
                return {
                    "format_error": True,
                    "error_info": "main_answer_parse_failed",
                    "final_output_text": final_text, 
                    "final_token_list": final_token_list, 
                    "main_answer": None,
                    "doc_history": doc_history
                }
            # format correct
            else:
                return {
                    "format_error": False,
                    "error_info": None,
                    "final_output_text": final_text, 
                    "final_token_list": final_token_list, 
                    "main_answer": main_answer,
                    "doc_history": doc_history
                }
            