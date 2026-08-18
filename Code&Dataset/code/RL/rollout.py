import os
import argparse
# 设置可见 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import chromadb
import sys
from retriever import e5_Retriever
import logging



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


class Rollout:
    def __init__(
        self, 
        model: str, 
        tokenizer: str, 
        e5_model_path: str,
        prompt_file_path: str,
        retriever_topk: int = 3,
        do_sample: bool = True,
        max_token_len: int = 2500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        rank: int|None = None
    ) -> None:
        
        # record the model math
        self.model = model
        self.tokenizer = tokenizer

        # logger
        self.logger = logging.getLogger(__name__)

        # sampling setting
        self.do_sample=do_sample
        self.max_token_len = max_token_len
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.rank=rank
        
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

 
        # 初始化 chromadb
        self.wiki_chroma_file_dic = os.path.join('../DatasetConstruction/DocPool', '2WikiMultiHopQA', 'train_set')
        self.wiki_client = chromadb.PersistentClient(path=self.wiki_chroma_file_dic)
        self.wiki_collection = self.wiki_client.get_or_create_collection("doc")

        self.musi_chroma_file_dic = os.path.join('../DatasetConstruction/DocPool', 'MusiQue', 'train_set')
        self.musi_client = chromadb.PersistentClient(path=self.musi_chroma_file_dic)
        self.musi_collection = self.musi_client.get_or_create_collection("doc")
        # self.logger.info(f"Chroma Database Loaded")

        self.retriever_topk = retriever_topk

        # 初始化 e5 编码器
        self.e5 = e5_Retriever(e5_model_path)

    def _retrieve_docs(
        self, 
        sub_question: str,  
        origin_dataset: str, 
        origin_sample_index: int
    ) -> list[str]:

        sub_question_text=["query: " + sub_question]
        sub_question_embedding = self.e5.embedding_generation(sub_question_text).tolist()

        if origin_dataset=="2WikiMultiHopQA": 
            result = self.wiki_collection.query(
                query_embeddings=sub_question_embedding,
                n_results=3,
                where={"origin_sample_index": origin_sample_index}
            )
        if origin_dataset=="MusiQue": 
            result = self.musi_collection.query(
                query_embeddings=sub_question_embedding,
                n_results=3,
                where={"origin_sample_index": origin_sample_index}
            )
        documents = result["documents"][0]
        # 过滤掉空字符串或全是空白的条目
        filtered_documents = [doc for doc in documents if doc.strip()]
        if len(filtered_documents)==0:
            self.logger.warning(f"[rank {self.rank}] 0 retreival document find in rollout search stage -- Sample: {origin_dataset}-{origin_sample_index}")
        return filtered_documents
    
    def _sub_question_extraction(self, output_text: str): 
        """ 
        从输出文本中提取所有符合 <sub-question>...</sub-question> *** <search>True</search>$ 的子问题， 
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
    
    def _extract_token_indices(
        self,
        token_str_list: list[str], 
        extract_range: tuple, 
        start_token_str, 
        end_token_str,
        include_start_end=False
    )->list[tuple]:
        """
        Extract token indices within `extract_range`  that start with `start_token_str` and end with `end_token_str`.
        (Non-greedy, `start_token_str` and end with `end_token_str` are not included in the range)
        `extract_range` = (start_index, end_index)
        
        Returns:
            List of indices that match the criteria
            eg: [(3, 6), (9, 12), ..]
        """
        result_indices = []

        i = extract_range[0]
        while i <= extract_range[1]:
            if token_str_list[i] == start_token_str:
                found_match = False
                for j in range(i + 1, min(extract_range[1]+1, len(token_str_list))):
                    if token_str_list[j] == end_token_str:
                        if include_start_end == False:
                            result_indices.append((i+1, j-1))
                        else:
                            result_indices.append((i, j))
                        i = j + 1
                        found_match = True
                        break
                if not found_match:
                    i += 1
            else:
                i += 1
        return result_indices
    
    def _concatenate_tokens_in_range(
        self,
        token_str_list: list[str], 
        token_range: tuple
    ) -> str:
        """
        Concatenate token strings in token_range [a, b] using tokenizer's proper detokenization.
        token_str_list: List of token strings (e.g., ["he", "##llo", "!"])

        Returns:
            str: Correctly reconstructed text segment
        """

        tokens_slice = token_str_list[token_range[0]:token_range[1]+1]
        # Use tokenizer's built-in detokenizer to properly join subwords
        return self.tokenizer.convert_tokens_to_string(tokens_slice).strip()

    def _result_parse(self, final_token_str_list: list[str], input_token_len: int)->tuple[list[dict] | None, bool | None]:
        """
        解析模型生成的token字符串列表，提取assistant部分并找到思维节点符号的内容索引
        
        Args:
            final_token_str_list: 包含生成token字符串的列表
            
        Returns:
            list: 包含每个样本解析结果的列表，每个结果包含assistant部分和思维节点索引信息
        """
        reject_answer = False

        # 提取完整思维链节点的索引范围
        CoT_range = self._extract_token_indices(final_token_str_list, 
                                                (input_token_len, len(final_token_str_list)-1),
                                                '<think>',
                                                '</sub-answer>',
                                                True)
        
        # format error occures
        if len(CoT_range)==0:
            return None, None
        
        nodes_parse_range = []
        for node_range in CoT_range:
            # 提取 <think> </think>
            think_range = self._extract_token_indices(final_token_str_list, 
                                    node_range,
                                    '<think>',
                                    '</think>')
            if len(think_range)!=1:
                return None, None
            think_range = think_range[0]
            think_str = self._concatenate_tokens_in_range(final_token_str_list, think_range)
            
            # 提取 <sub-question> </sub-question>
            sub_question_range = self._extract_token_indices(final_token_str_list, 
                                    node_range,
                                    '<sub-question>',
                                    '</sub-question>')
            if len(sub_question_range)!=1:
                return None, None
            sub_question_range=sub_question_range[0]
            sub_question_str = self._concatenate_tokens_in_range(final_token_str_list, sub_question_range)

            # 提取 <search> </search> 
            search_range = self._extract_token_indices(final_token_str_list, 
                                    node_range,
                                    '<search>',
                                    '</search>')
            if len(search_range)!=1:
                return None, None
            search_range = search_range[0][0]
            search_str = final_token_str_list[search_range]
            if search_str not in ['True', 'False']:
                return None, None

            # 提取 <doc-type> </doc-type> if <search>True</search>
            if final_token_str_list[search_range] == 'True':
                doc_type_range = self._extract_token_indices(final_token_str_list, 
                                    node_range,
                                    '<doc-type>',
                                    '</doc-type>')
                doc_range = self._extract_token_indices(final_token_str_list, 
                                node_range,
                                '<doc>',
                                '</doc>')
                if len(doc_type_range)!= len(doc_range):
                    return None, None
                doc_info_str_list = []
                for i in range(len(doc_type_range)):
                    doc_type_str = self._concatenate_tokens_in_range(final_token_str_list, doc_type_range[i])
                    doc_str = self._concatenate_tokens_in_range(final_token_str_list, doc_range[i])
                    doc_info_str_list.append((doc_type_str, doc_str))
                # 针对此子问题 全部文档均为 useless 拒绝回答
                if all(x[0] == 'useless' for x in doc_info_str_list):
                    reject_answer = True
                    
            else:
                doc_type_range = None
                doc_info_str_list = None

            # 提取 <sub-answer> </sub-answer>
            sub_answer_range = self._extract_token_indices(final_token_str_list, 
                                    node_range,
                                    '<sub-answer>',
                                    '</sub-answer>')
            if len(sub_answer_range)!=1:
                return None, None
            sub_answer_range=sub_answer_range[0]
            sub_answer_str = self._concatenate_tokens_in_range(final_token_str_list, sub_answer_range)

            # 构建当前节点的解析结果
            parse_result = {
                'node_range': node_range,       # tuple[int, int]
                'think_range': think_range,     # tuple[int, int]
                'think_str': think_str,
                'sub_question_range': sub_question_range,   # tuple[int, int]
                'sub_question_str': sub_question_str,
                'search_range': search_range,               # int
                'search_str': search_str,
                'doc_type_range_list': doc_type_range,      # list[tuple[int, int], tuple[int, int], ...] | None
                'doc_info_str_list': doc_info_str_list,
                'sub_answer_range': sub_answer_range,       # tuple[int, int]
                'sub_answer_str': sub_answer_str
            }
            nodes_parse_range.append(parse_result)

        return nodes_parse_range, reject_answer

    
    
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
        input_len = input["input_ids"].shape[-1]
        max_new_tokens = max(1, self.max_token_len - input_len)

        output_ids = self.model.generate(
            **input,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
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

    def inference_RAG(self, main_question: str, origin_dataset: str, origin_sample_index: int):
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
        input = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        initial_input_len = len(input['input_ids'][0])

        format_error = False
        doc_list = []
        search_doc_history = []
        stopping_criteria = self.stopping_criteria_search

        try:
            while True:
                info, output_ids = self._inference_special_token_ended(input, stopping_criteria)
                # stop because max token length
                if info["end_token"] ==  None:
                    final_text = self.tokenizer.decode(output_ids[0][initial_input_len:], skip_special_tokens=False)
                    final_token_list = self.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)
                    format_error = True
                    break
                # stop because generate eos_token
                if info["end_token"] ==  "end":
                    final_text = self.tokenizer.decode(output_ids[0][initial_input_len:], skip_special_tokens=False)
                    final_token_list = self.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)
                    break
                # stop because generate<search>True</search>
                if info["end_token"] == "search-true":
                    # sub_question parse failed
                    if info["sub_question"] == None:
                        final_text = self.tokenizer.decode(output_ids[0][initial_input_len:], skip_special_tokens=False)
                        final_token_list = self.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)
                        format_error = True
                        break
                    # sub_question parse successed
                    else:
                        # init the doc list of current sub question
                        doc_list = self._retrieve_docs(info["sub_question"],  origin_dataset, origin_sample_index)
                        search_doc_history.append(doc_list.copy())
                        
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

            if format_error == True:
                if info["end_token"] ==  None:
                    error_info = "token length exceed"
                else:
                    error_info = "sub question parse failed"
                self.logger.error(f"[rank {self.rank}] Rollout Error occures: {error_info} -- Sample: {origin_dataset}-{origin_sample_index}")
                return {
                    "format_error": True,
                    "error_info": error_info,
                    "final_output_text": final_text, 
                    "input_token_len": initial_input_len,
                    "final_token_len": len(final_token_list)
                }
            # break from info == {"end_token": "end"}:
            else:
                nodes_parse_range, reject_answer = self._result_parse(final_token_list, initial_input_len)
                main_answer_range = self._extract_token_indices(final_token_list, 
                                                            (initial_input_len, len(final_token_list)-1),
                                                            '<main-answer>',
                                                            '</main-answer>')
                if nodes_parse_range != None and main_answer_range != []:
                    main_answer_range = main_answer_range[0]
                    main_answer_str = self._concatenate_tokens_in_range(final_token_list, main_answer_range)
                    return {
                        "format_error": False,
                        "final_output_text": final_text, 
                        "final_token_str_list": final_token_list,
                        "final_token_id_tensor": output_ids[0], # torch.tensor 1D (sequence_length)
                        "main_question": main_question,
                        "nodes_info": nodes_parse_range,
                        "reject_answer": reject_answer,
                        "main_answer_range": main_answer_range,
                        "main_answer_str": main_answer_str,
                        "search_doc_history": search_doc_history,
                        "input_token_len": initial_input_len,
                        "final_token_len": len(final_token_list)
                    }
                else:
                    # final result parse failed
                    self.logger.error(f"[rank {self.rank}] Rollout Error occures: final output parse failed -- Sample: {origin_dataset}-{origin_sample_index}")
                    return {
                        "format_error": True,
                        "error_info": "final output parse failed",
                        "final_output_text": final_text, 
                        "input_token_len": initial_input_len,
                        "final_token_len": len(final_token_list)
                    }
        
        except Exception as e:
            self.logger.exception(f"[rank {self.rank}] Rollout Error occures: {str(e)} -- Sample: {origin_dataset}-{origin_sample_index}")
            return {
                "format_error": None,
                "error_info": str(e),
                "final_output_text": None, 
                "input_token_len": initial_input_len,
                "final_token_len": None
            }
        '''
        return dict format 
        # 0.
        {
            "format_error": None,
            "error_info": str(e),
            "final_output_text": None, 
            "input_token_len": initial_input_len,
            "final_token_len": None
        }
        # 1.
        {
            "format_error": True,
            "error_info": error_info,
            "final_output_text": final_text, 
            "input_token_len": initial_input_len,
            "final_token_len": None
        }
        # 2.
        {
            "format_error": False,
            "final_output_text": final_text, 
            "final_token_str_list": final_token_list,
            "final_token_id_tensor": output_ids[0], # torch.tensor 1D (sequence_length)
            "main_question": main_question,
            "nodes_info": nodes_parse_range,
            "main_answer_range": main_answer_range,
            "main_answer_str": main_answer_str,
            "search_doc_history": [[hop1_doc1, hop1_doc2, hop1_doc3, ...], [hop2_doc1, ...], ...],
            "input_token_len": initial_input_len,
            "final_token_len": len(final_token_list)
        }
        nodes_info:
        [
            {
                'node_range': node_range,       # tuple[int, int]
                'think_range': think_range,     # tuple[int, int]
                'think_str': think_str,
                'sub_question_range': sub_question_range,   # tuple[int, int]
                'sub_question_str': sub_question_str,
                'search_range': search_range,               # int
                'search_str': search_str,
                'doc_type_range_list': doc_type_range,      # list[tuple[int, int], tuple[int, int], ...] | None
                'doc_info_str_list': doc_info_str_list,
                'sub_answer_range': sub_answer_range,       # tuple[int, int]
                'sub_answer_str': sub_answer_str
            },
            {
                ...
            },
            ...
        ]
        '''



      
