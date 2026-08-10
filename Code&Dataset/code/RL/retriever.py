import re
import os
import torch.nn.functional as F
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel


# e5 模型较小 可以直接在 CPU 上运行
class e5_Retriever:
    def __init__(self, model_path):
        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
    def _average_pool(self,
                     last_hidden_states: Tensor,
                     attention_mask: Tensor
                     ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embedding_generation(self, texts: list[str]) -> Tensor:
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def sim_calculate_vector(self,
                      querys: list[str],
                      passage_embeddings: Tensor,
                      ) -> list:
        query_embeddings = self.embedding_generation(querys)
        # 计算相似度分数
        scores = (query_embeddings @ passage_embeddings.T) * 100
        return scores.tolist()

    def sim_calculate(self,
                      querys: list[str],
                      passages: list[str],
                      ) -> list:
        query_embeddings = self.embedding_generation(querys)
        passage_embeddings = self.embedding_generation(passages)
        # 计算相似度分数
        scores = (query_embeddings @ passage_embeddings.T) * 100
        return scores.tolist()
    
if __name__ == "__main__":
    from dotenv import load_dotenv

    # load .env file (store the api_key)
    load_dotenv('../.env')
    MODEL_PATH = os.getenv("E5_MODEL_PATH")

    retriever = e5_Retriever(model_path=MODEL_PATH)


    # Each input text should start with "query: " or "passage: ".
    # For tasks other than retrieval, you can simply use the "query: " prefix.
    q_texts = ['query: how much protein should a female eat']
    p_texts = ["passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
                "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]
    
    print(retriever.sim_calculate(q_texts, p_texts))