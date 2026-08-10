from ast import arg
import os
# 设置可见 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import argparse
from inference import trained_model_inference

class RAGTestInferenceRunner:
    def __init__(
            self, 
            base_model_path,
            lora_adapter_path, 
            prompt_file_path, 
            e5_model_path, 
            retriever_topk, 
            global_retrieve,
            do_sample
        ):
        self.rag_instance = trained_model_inference(
            base_model_path=base_model_path,
            lora_adapter_path=lora_adapter_path,
            prompt_file_path=prompt_file_path,
            e5_model_path=e5_model_path,
            retriever_topk=retriever_topk,
            global_retrieve=global_retrieve,
            do_sample=do_sample
        )

    def run_inference(self, test_file_path, output_file_path, noise_ratio: float|None = None, save_every=10):
        """
        Run RAG inference on test data and save results incrementally.
        
        Args:
            test_file_path (str): Path to the test JSON file.
            output_file_path (str): Path to save the output JSON results.
            save_every (int): Save results to file every N samples.
        """
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_data=json.load(f)

        results = []
        total = len(test_data)
        print(f"Loaded {total} test samples from {test_file_path}")

        for idx, item in enumerate(test_data):
            main_question = item["main_question"]
            origin_dataset = item["data_source"][0]
            origin_sample_index = item["data_source"][2]
            if noise_ratio is not None:
                # golden_doc = {node["sub_question"]: [node["doc"]] + node["augmented_docs"]  for node in item["chain_of_thought"]}
                golden_doc = [[node["doc"]] + node["augmented_docs"]  for node in item["chain_of_thought"]]
                print("Golden Docs: ", golden_doc)
            else:
                golden_doc = None

            print(f"\n[{idx + 1}/{total}] \t Processing: {main_question}")

            rag_result = self.rag_instance.inference_RAG(
                main_question, 
                origin_dataset, 
                origin_sample_index,
                noise_ratio,
                golden_doc
            )

            filtered_rag_result = {
                "format_error": rag_result.get("format_error"),
                "error_info": rag_result.get("error_info"),
                "final_output_text": rag_result.get("final_output_text"),
                "main_answer": rag_result.get("main_answer"),
                "doc_history": rag_result.get("doc_history")
            }

            result_entry = {
                "split_dataset_index": idx,
                "data_source": item["data_source"],
                "main_question": main_question,
                "origin_dataset": origin_dataset,
                "origin_sample_index": origin_sample_index,
                "rag_result": filtered_rag_result
            }
            results.append(result_entry)

            # Save incrementally
            if (idx + 1) % save_every == 0:
                self._save_results(results, output_file_path)
                print(f"✅ Saved {len(results)} results so far to {output_file_path}")

        # Final save
        self._save_results(results, output_file_path)
        print(f"🎉 Inference completed. Final results saved to {output_file_path}")

    def _save_results(self, results, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run RAG inference with LoRA-adapted Qwen model.")
    # RAG system setting
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model (e.g., Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--e5_model_path", type=str, required=True,
                        help="Path to the e5 model (e.g., e5-base-v2)")
    parser.add_argument("--prompt_file_path", type=str, required=True,
                        help="Path to the prompt template file")
    parser.add_argument("--retriever_topk", type=int, default=3,
                        help="Number of retrieved documents (default: 3)")
    
    # test data path setting
    parser.add_argument("--test_file_path", type=str, required=True,
                         help="Path to the test JSON file")
    
    # result save setting
    parser.add_argument("--output_file_path", type=str, required=True,
                        help="Path to save the inference results (JSON)")
    
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save results every N samples (default: 10)")
    
    parser.add_argument("--global_retrieve", type=str, default="False")
    
    parser.add_argument("--do_sample", type=str, default="False")
    
    parser.add_argument("--noise_ratio", type=float, default=None,
                        help="The ratio of noise to add to the retrieved documents (default: None)")

    
    args = parser.parse_args()

    noise_ratio = args.noise_ratio

    print(f"Current process PID: {os.getpid()}")
    print(f"Current args: {vars(args)}")

    from datetime import datetime
    now = datetime.now()
    print(f"Start Time: {now}")

    if args.global_retrieve=="True":
        global_retrieve=True
    else:
        global_retrieve=False
    
    if args.do_sample=="True":
        do_sample=True
    else:
        do_sample=False


    runner = RAGTestInferenceRunner(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        prompt_file_path=args.prompt_file_path,
        e5_model_path=args.e5_model_path,
        retriever_topk=args.retriever_topk,
        global_retrieve=global_retrieve,
        do_sample=do_sample
    )

    
    print(f"Test File Path:{args.test_file_path}")
    runner.run_inference(
        test_file_path=args.test_file_path,
        output_file_path=args.output_file_path,
        noise_ratio=noise_ratio,
        save_every=args.save_every
    )

    now = datetime.now()
    print(f"End Time: {now}")


if __name__ == "__main__":
    main()

