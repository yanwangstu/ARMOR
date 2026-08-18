'''
Calculate the reward of a specific step under a specific task
'''
import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.nn import functional as F
from llm_invoke import _llm_invoke

'''
def _answer_refinement(question, answer, reward_model, reward_tokenizer):
    system_prompt = 'Answer the question directly and concisely. Do not add explanations or extra text.'
    user_prompt = (
        f'Question: When did Aleksei Balabanov die?\n'
        f'Answer: Aleksei Balabanov\'s date of death is 18 May 2013.\n'
        f'Refined: 18 May 2013.\n\n'
        f'Question: {question}\n'
        f'Answer: {answer}\n'
        f'Refined:'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    generated_text = _llm_invoke(messages, reward_model, reward_tokenizer, 50, do_sample=False)
    return generated_text.strip().rstrip('.')
'''

# use llm to generate the answer directly
def _answer_llm_generation(question, model, tokenizer):
    system_prompt = 'Please provide a direct and concise answer to the question. Do not add explanations or extra text.'
    user_prompt = f"Question : {question}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    generated_text = _llm_invoke(messages, model, tokenizer, 50, do_sample=False)
    return generated_text.strip()


def _question_same_compare(question_a: str|None, question_b: str, reward_model, reward_tokenizer):
    if question_a == None:
        return False
    if question_a == question_b:
        return True
    system_prompt = (
        "You are an expert at comparing questions. Determine if Question A and Question B "
        "are semantically equivalent (i.e., they ask for the same information and would have "
        "the same correct answer). Respond with ONLY 'Yes' or 'No'."
    )
    user_prompt = f"Question A: {question_a}\nQuestion B: {question_b}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    generated_text = _llm_invoke(messages, reward_model, reward_tokenizer, 10, do_sample=False)
    print(messages)
    print(question_a, question_b, generated_text)
    generated_text = generated_text.strip().lower()
    first_word = generated_text.split()[0] if generated_text.split() else ""
    if first_word == 'yes':
        return True
    else:
        return False


def _correct_answer_generation(reward_model, reward_tokenizer, question: str, ref_qa_pair: tuple[str, str]):
    """
    使用 reward model（作为LLM）判断问题是否语义一致，决定是否返回参考答案
    """
    ref_question, ref_answer = ref_qa_pair

    
    # 1. 字符串精确匹配（最快路径）
    if question.strip() == ref_question.strip():
        return ref_answer
    
    # 2. 使用 reward model（LLM）进行语义匹配
    compare_same_result = _question_same_compare(question, ref_question, reward_model, reward_tokenizer)
    if compare_same_result == True:
        return ref_answer
    
    # 3. 匹配失败
    return None


def _correct_answer_log_confidence(policy_model, policy_tokenizer, question: str, correct_answer: str):
    system_prompt = 'Please provide a direct and concise answer to the question. Do not add explanations or extra text.'

    question = f"Question: {question}"
    messages = [
        {"role": "system",   "content": system_prompt},
        {"role": "user",     "content": question},
        {"role": "assistant","content": correct_answer},
    ]

    # 返回 token tensor（包含完整上下文 + answer）
    input_ids = policy_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(policy_model.device)

    messages_no_answer = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
        {"role": "assistant", "content": ""},
    ]

    input_ids_no_answer = policy_tokenizer.apply_chat_template(
        messages_no_answer,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(policy_model.device)

    answer_start = input_ids_no_answer.shape[1]
    answer_end = input_ids.shape[1] - 1

    with torch.no_grad():
        attention_mask = torch.ones_like(input_ids)
        outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits   # [1, length, Vocab_Size]

    # shift
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # answer mask
    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    mask[:, answer_start-1:answer_end] = True

    # token-level log prob
    log_probs = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none"
    ).reshape(shift_labels.size())

    selected = log_probs[mask]

    avg_log_prob = selected.mean()
    return avg_log_prob.float().cpu().item()
    '''
    return {
        "avg_log_prob": avg_log_prob,
        "num_tokens": selected.numel(),
    }
    '''


def _question_search_reward_calculate(
        reward_model, 
        reward_tokenizer, 
        policy_model, 
        policy_tokenizer, 
        question: str, 
        search: bool,
        pattern_qa_pairs: list[tuple[str, str]],
        index: int
        )->float:
    # print("function", question)
    # print("function", pattern_qa_pairs)
    if 0 <= index < len(pattern_qa_pairs):
        ordered = [pattern_qa_pairs[index]] + [
            pair for i, pair in enumerate(pattern_qa_pairs) if i != index
        ]
    else:
        ordered = pattern_qa_pairs
    
    for ref_qa_pair in ordered:
        correct_answer: str = _correct_answer_generation(reward_model, reward_tokenizer, question, ref_qa_pair)
        if correct_answer != None:
            break
    # print(correct_answer)
    if correct_answer == None:
        return None, None

    policy_answer = _answer_llm_generation(question, policy_model, policy_tokenizer).strip()

    system_prompt = 'Compare two answer. Do not add explanations or extra text.'
    user_prompt = (
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n" 
        f"Candidate Answer: {policy_answer}\n\n"
        "Are these two answers essentially the same in meaning? Answer ONLY with 'True' or 'False'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    generated_text = _llm_invoke(messages, reward_model, reward_tokenizer, 10, do_sample=False)

    answer_same = generated_text.strip().lower() == 'true'

    if answer_same == True:
        if search == True:
            reward=-0.5
        else:
            reward=0.5
    if answer_same == False:
        if search == True:
            reward=1
        else:
            reward=-1
    return reward, (correct_answer, policy_answer)


def _question_answer_reward_calculate(
        reward_model, 
        reward_tokenizer, 
        question,
        evaluate_answer,
        pattern_qa_pairs,
        index):
    
    if index == None:
        _, correct_answer = pattern_qa_pairs[0]
    
    else:
        if 0 <= index < len(pattern_qa_pairs):
            ordered = [pattern_qa_pairs[index]] + [
                pair for i, pair in enumerate(pattern_qa_pairs) if i != index
            ]
        else:
            ordered = pattern_qa_pairs
        
        for ref_qa_pair in ordered:
            correct_answer: str = _correct_answer_generation(reward_model, reward_tokenizer, question, ref_qa_pair)
            if correct_answer != None:
                break
        
        if correct_answer == None:
            return None, None
    
    score_prompt_file_path='prompt/score_prompt.txt'
    with open(score_prompt_file_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    user_prompt = (
        f"Question: {question}\n"
        f"Candidate Answer:{evaluate_answer}\n"
        f"Ground Truth Answer:{correct_answer}\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    score_text = _llm_invoke(messages, reward_model, reward_tokenizer, 10, do_sample=False)
    
    # match the 0-5 score in the output
    try:
        match = re.search(r'\d', score_text)
        match_score = int(match.group())
        if match_score == 5:
            return 1, correct_answer
        elif 0 <= match_score < 5:
            return 0, correct_answer
        else:
            return 0, correct_answer
    except Exception as e:
        return 0, correct_answer
    

def _doc_useful_compare(question: str, document: str, reward_model, reward_tokenizer):
    """
    Return True if the document is useful for answering the question,
    otherwise return False.
    """

    score_prompt_file_path='prompt/doc_useful_compare_prompt.txt'
    with open(score_prompt_file_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    user_prompt = (
        f"Question: {question}\n"
        f"Document:{document}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    raw_output = _llm_invoke(messages, reward_model, reward_tokenizer, max_new_tokens=10, do_sample=False)

    # 解析 True / False
    text = raw_output.strip().lower()
    if "true" in text:
        return True
    if "false" in text:
        return False
    # fallback: 若模型输出异常，则当作 False
    return False
