import json
import torch

def dump_rollout_json(data, file_path):
    """
    将 data 写入 file_path，满足：
    - list[int], list[str], list[list[int]] 均为单行；
    - 其他结构缩进 4 空格；
    - 保留 Unicode 字符；
    - 文件为 UTF-8。
    """
    def is_simple_int_list(lst):
        return (
            isinstance(lst, list)
            and len(lst) > 0
            and all(isinstance(x, int) and not isinstance(x, bool) for x in lst)
        )

    def is_simple_str_list(lst):
        return (
            isinstance(lst, list)
            and len(lst) > 0
            and all(isinstance(x, str) for x in lst)
        )
    
    def is_list_of_int_lists(lst):
        return (
            isinstance(lst, list)
            and len(lst) > 0
            and all(is_simple_int_list(x) for x in lst)
        )

    def format_value(obj, indent=0):
        pad = "    " * indent
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = []
            for k, v in obj.items():
                key_str = json.dumps(k, ensure_ascii=False)
                val_str = format_value(v, indent + 1)
                items.append(f"{pad}    {key_str}: {val_str}")
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        
        elif isinstance(obj, list):
            if is_simple_int_list(obj):
                return "[" + ", ".join(str(x) for x in obj) + "]"
            if is_simple_str_list(obj):
                return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in obj) + "]"
            if is_list_of_int_lists(obj):
                inner = ", ".join(
                    "[" + ", ".join(str(x) for x in inner_lst) + "]"
                    for inner_lst in obj
                )
                return "[" + inner + "]"
            # 其他 list：递归带缩进
            if not obj:
                return "[]"
            items = [format_value(item, indent + 1) for item in obj]
            return "[\n" + ",\n".join(f"{pad}    {item}" for item in items) + "\n" + pad + "]"
        
        else:
            return json.dumps(obj, ensure_ascii=False)
        
    data = convert_tensors(data)
    formatted = format_value(data, 0)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(formatted)


def convert_tensors(obj):
    """递归地将所有 Tensor 转换为列表或原生类型"""
    if isinstance(obj, torch.Tensor):
        # 如果 Tensor 是标量
        if obj.numel() == 1:
            return obj.item()
        # 如果 Tensor 是多维的
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors(item) for item in obj)
    else:
        return obj