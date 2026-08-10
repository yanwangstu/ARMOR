import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _llm_invoke(messages, model, tokenizer, max_new_tokens, do_sample=False):
    if hasattr(model, 'module'):
        model = model.module
    model_name = model.config.name_or_path
    kwargs = {'enable_thinking': False} if 'Qwen3' in model_name else {}
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        **kwargs
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    input_ids_length = inputs['input_ids'].shape[1]
    
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )
    
    generated_text = tokenizer.decode(output[0][input_ids_length:], skip_special_tokens=True)
    return generated_text
