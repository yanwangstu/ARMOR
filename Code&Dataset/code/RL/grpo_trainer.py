import os
# 禁用 LangChain 的匿名遥测数据收集。
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import torch
import math
import json
import logging
import concurrent.futures
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from optimizer import GRPOOptimizer
from rollout import Rollout
from task_schedular import TeacherAgent, TrainTaskType
from rl_data_utils import RLDataset
from json_dump import dump_rollout_json
from reward_cal import (
    _question_same_compare, 
    _question_answer_reward_calculate, 
    _question_search_reward_calculate, 
    _doc_useful_compare
)




class GRPOTrainer(GRPOOptimizer):
    def __init__(
        self,
        train_dataset_path: str,
        system_prompt_path: str,
        lora_config: LoraConfig,
        policy_model_path: str,
        reward_model_path: str,
        e5_model_path: str,
        model_save_path: str,
        model_save_steps: int,
        rollout_info_save_dic: str,
        rollout_info_save_step: int,
        batch_size: int,
        rollout_num: int,
        rollout_micro_batch_size_per_gpu: int,
        policy_ref_micro_batch_size_per_gpu: int,
        scale_rewards: bool,
        max_token_len: int,
        temperature: float,
        top_p: float,
        top_k: float
    ):
        
        # init dataset
        self.train_dataset_path = train_dataset_path
        self.system_prompt_path = system_prompt_path
        self.train_dataset = RLDataset(train_dataset_path, system_prompt_path)

        # init training config
        self.batch_size = batch_size
        self.rollout_num = rollout_num
        self.rollout_micro_batch_size_per_gpu = rollout_micro_batch_size_per_gpu
        self.policy_ref_micro_batch_size_per_gpu = policy_ref_micro_batch_size_per_gpu
        self.model_save_steps = model_save_steps
        self.model_save_path = model_save_path
        self.scale_rewards = scale_rewards

        # init logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Current process PID: {os.getpid()}")
        
        # init model
        self.lora_config = lora_config
        self.policy_model_path = policy_model_path
        self.reward_model_path = reward_model_path
        self.e5_model_path = e5_model_path
        self._model_init()
        self.logger.info(f"Current Rank: {self.rank}")
        self.logger.info(f"World Size: {self.world_size}")
        self.logger.info(f"Model initialized successfully.")
        if self.rank==0:
            # log model Info
            print('Policy Model Training Info and LoRA Wrapped Structure')
            self.policy_model.module.print_trainable_parameters()
            print(self.policy_model)
        
        # init optimizer
        super().__init__(self.policy_model)

        # init rollout instance
        try:
            self.rollout_max_token_len = max_token_len
            self.rollout_tem = temperature
            self.rollout_top_p = top_p
            self.rollout_top_k = top_k
            self.rollout_instance = Rollout(
                model=self.policy_model.module,
                tokenizer=self.policy_tokenizer,
                e5_model_path=self.e5_model_path,
                prompt_file_path=self.system_prompt_path,
                # logger=self.logger,
                max_token_len = self.rollout_max_token_len,
                temperature = self.rollout_tem,
                top_p = self.rollout_top_p,
                top_k = self.rollout_top_k,
                rank = self.rank
            )
            os.makedirs(rollout_info_save_dic, exist_ok=True)
            self.rollout_info_save_path = os.path.join(rollout_info_save_dic, f"rank_{self.rank}.json")
            self.rollout_info_save_step = rollout_info_save_step
            self.logger.info(f"Rollout instance initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Rollout instance: {e}")
            raise e
        
        # init schedular in the first process
        if self.rank == 0:
            self.train_steps = [math.ceil(len(self.train_dataset)/batch_size)]
            self.schedular = TeacherAgent(self.train_steps[0])
        else:
            self.schedular = None
            self.train_steps = [None]
        dist.broadcast_object_list(self.train_steps, src=0)
        self.train_steps = self.train_steps[0]
        self.logger.info(f'Task Schedular initialized successfully in MAIN rank, total training steps {self.train_steps}')

        return
    
    def _model_init(self):
        # init distributed environment parameter
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
            if os.environ['RANK'] != os.environ['LOCAL_RANK']:
                raise NotImplementedError("Multi-node training is not supported.")
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE']) # total process num
            self.gpu_id = int(os.environ['LOCAL_RANK'])
            # check available GPU num
            available_gpu_num = torch.cuda.device_count()
            if self.world_size > available_gpu_num:
                raise ValueError(f"Not enough GPUs available. Word Size set to {self.world_size}, Required: {self.world_size} GPUs, Available: {available_gpu_num} GPUs")
            if self.world_size > self.rollout_num:
                raise ValueError(f'world_size ({self.world_size}) > rollout_num ({self.rollout_num}), some process (GPU) face 0 rollout')
        else:
            raise ValueError("Distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not set")
        
        # set GPU for current process, each process own one GPU
        torch.cuda.set_device(self.gpu_id)

        # init distributed process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://', # read distributed config form os.env
            world_size=self.world_size,
            rank=self.rank
        )

        # load policy tokenizer
        self.policy_tokenizer = AutoTokenizer.from_pretrained(self.policy_model_path)
        if self.policy_tokenizer.pad_token is None:
            self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token
        
        # load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.policy_model_path,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": self.gpu_id}
        )
        self.policy_model = get_peft_model(self.policy_model, self.lora_config)
        self.policy_model = DDP(self.policy_model, device_ids=[self.gpu_id])

        # load reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_path)
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        
        # load reward model -- no gradient
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            self.reward_model_path,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": self.gpu_id}
        )
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # load reference tokenizer
        self.ref_tokenizer = self.policy_tokenizer

        # load reference model -- no gradient
        if self.reward_model_path == self.policy_model_path:
            self.ref_model = self.reward_model
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.policy_model_path,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map={"": self.gpu_id}
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # wait to Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
        else:
            raise RuntimeError("distributed env not initialized.")
    

    def _rollout(
            self, 
            batch_sample: dict[dict], 
            rollout_num: int, 
            current_task: TrainTaskType
        )->dict[list]:
        
        rollout_results = {}
        max_concurrent = min(rollout_num, self.rollout_micro_batch_size_per_gpu)
        for id, sample in batch_sample.items():
            question_results = [None] * rollout_num
            # control concurrent through thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # submit all tasks
                # future_to_idx map future obj into index
                future_to_idx = {
                    executor.submit(
                        self.rollout_instance.inference_RAG,
                        sample["main_question"],
                        sample["data_source"][0],
                        sample["data_source"][2],
                    ): local_rollout_index for local_rollout_index in range(rollout_num)
                }
                # 等待所有任务完成并收集结果
                for future in concurrent.futures.as_completed(future_to_idx):
                    local_rollout_index = future_to_idx[future]
                    result = future.result()
                    question_results[local_rollout_index] = result
            rollout_results[id] = question_results
        return rollout_results
    
        '''
        rollout_results format
        {
            "sample_id_1": [result_1, result_2, ..., result_N],
            "sample_id_2": [result_1, result_2, ..., result_N],
            ...
        }
        result format
        # 0.
        {
            "format_error": None,
            "error_info": e,
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
            "final_token_len": len(final_token_list)
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

    def _reward_cal(
        self, 
        batch_responses: dict[str, list[dict]], 
        batch: dict[dict], 
        current_task: TrainTaskType
    ):

        if current_task==TrainTaskType.QuestionSearch:
            return self._question_search_task_reward(batch_responses, batch)
        elif current_task==TrainTaskType.DocType:
            return self._doc_type_task_reward(batch_responses, batch)
        elif current_task==TrainTaskType.QuestionAnswer:
            return self._question_answer_task_reward(batch_responses, batch)
        else:
            raise ValueError(f"Unsupported task type: {current_task}")

    def _question_search_task_reward(self, batch_responses: dict[str, list[dict]], batch: dict[dict]):
        reward = {}
        for id, sample in batch.items():
            # 遍历该样本的标准 CoT QA 对，存储到 pattern_qa_pairs
            pattern_qa_pairs = []
            for node in sample["chain_of_thought"]:
                pattern_qa_pairs.append((node["sub_question"], node["sub_answer"]))
            
            # 遍历该样本的 rollout 结果
            reward[id] = []
            for rollout in batch_responses[id]:
                rollout_reward = []

                if "nodes_info" not in rollout:
                    # format error or other error occures
                    reward[id].append(rollout_reward)
                    continue

                for index, node in enumerate(rollout["nodes_info"]):
                    sub_question = node['sub_question_str']
                    search = True if node['search_str']=='True' else False
                    search_range: int = node['search_range']

                    # correct_noRAG_answer = (correct_answer, policy_answer)
                    search_reward, correct_noRAG_answer = _question_search_reward_calculate(
                        self.reward_model, 
                        self.reward_tokenizer, 
                        self.policy_model, 
                        self.policy_tokenizer, 
                        sub_question, 
                        search,
                        pattern_qa_pairs,
                        index)
                    if search_reward != None:
                        rollout_reward.append((search_reward, search_range, correct_noRAG_answer))

                reward[id].append(rollout_reward)
        return reward
        '''
        question_search_reward format
        {
            "sample_id_1": [[(1, 226, (correct_answer, noRAG_answer)), (...), ...], []=>no reward, ...],
            "sample_id_2": [[(-1, 211, (correct_answer, noRAG_answer)), (...), ...], ...],
            ...
        }
        '''
    
    def _question_answer_task_reward(self, batch_responses: dict[str, list[dict]], batch: dict[dict]):
        reward = {}
        for id, sample in batch.items():
            # 遍历该样本的标准 CoT QA 对，存储到 pattern_qa_pairs
            pattern_qa_pairs = []
            for node in sample["chain_of_thought"]:
                pattern_qa_pairs.append((node["sub_question"], node["sub_answer"]))
            # 遍历该样本的 rollout 结果
            reward[id] = []
            for rollout in batch_responses[id]:
                rollout_reward = []
                sub_answer_failed = False

                if "nodes_info" not in rollout:
                    # format error or other error occures
                    reward[id].append(rollout_reward)
                    continue

                # sub_answer_reward_cal
                for index, node in enumerate(rollout["nodes_info"]):
                    sub_question = node['sub_question_str']
                    sub_answer = node['sub_answer_str']
                    sub_answer_range: tuple[int, int] = node['sub_answer_range']

                    answer_reward, correct_answer = _question_answer_reward_calculate(
                        self.reward_model, 
                        self.reward_tokenizer, 
                        sub_question,
                        sub_answer,
                        pattern_qa_pairs,
                        index)
                    if answer_reward != None:
                        if answer_reward == -1:
                            sub_answer_failed = True
                        rollout_reward.append((answer_reward, sub_answer_range, (correct_answer, sub_answer)))
                    else:
                        sub_answer_failed = True
                    
                if not sub_answer_failed:
                    main_question = sample['main_question']
                    main_answer = rollout['main_answer_str']
                    main_answer_range: tuple[int, int] = rollout['main_answer_range']
            
                    ref_qa_pair = [(sample['main_question'], sample['main_answer'])]
                    answer_reward, correct_answer = _question_answer_reward_calculate(
                        self.reward_model, 
                        self.reward_tokenizer, 
                        main_question,
                        main_answer,
                        ref_qa_pair,
                        None)
                    if answer_reward != None:
                        rollout_reward.append((answer_reward, main_answer_range, (correct_answer, main_answer)))
                    
                # store
                reward[id].append(rollout_reward)
        return reward
        '''
        question_answer_reward format
        {
            "sample_id_1": [[(1, (221, 226), (correct_answer, policy_answer)), (...), ...], []=>no reward, ...],
            "sample_id_2": [[(-1, (122, 129), (correct_answer, policy_answer)), (...), ...], ...],
            ...
        }
        '''
    
    def _doc_type_task_reward(self, batch_responses: dict[str, list[dict]], batch: dict[dict]):
        reward = {}
        for id, sample in batch.items():
            # 遍历该样本的标准 CoT Question-GoldenDoc(useful doc) 对，存储到 pattern_qd_pairs
            pattern_qd_pairs = []
            for node in sample["chain_of_thought"]:
                pattern_qd_pairs.append((node["sub_question"], node["doc"]))
            # 遍历该样本的 rollout 结果
            reward[id] = []
            for rollout in batch_responses[id]:
                rollout_reward = []

                if "nodes_info" not in rollout:
                    # format error or other error occures
                    reward[id].append(rollout_reward)
                    continue
                
                doc_list = rollout['search_doc_history']
                no_search = 0
                for index, node in enumerate(rollout["nodes_info"]):
                    sub_question = node['sub_question_str']
                    doc_type_range_list = node['doc_type_range_list']
                    doc_info_str_list = node['doc_info_str_list']
                    if doc_info_str_list == None:
                        no_search += 1
                        continue
                    try:
                        ref_qd_pair: tuple[str, str] = pattern_qd_pairs[index]
                    # out of range
                    except IndexError:
                        ref_qd_pair = (None, None)
                    if _question_same_compare(ref_qd_pair[0], sub_question, self.reward_model, self.reward_tokenizer):
                        for i in range(len(doc_info_str_list)):
                            doc = doc_list[index-no_search][i]
                            
                            doc_type = doc_info_str_list[i][0]
                            doc_type_range = doc_type_range_list[i]

                            doc_is_useful = (doc == ref_qd_pair[1])
                            doc_is_useful_llm = (doc_type=='useful')

                            doc_reward = 1 if doc_is_useful == doc_is_useful_llm else -1
                            rollout_reward.append((doc_reward, doc_type_range, 'useful' if doc_is_useful else 'useless'))

                    else:
                        for i in range(len(doc_info_str_list)):
                            doc_type = doc_info_str_list[i][0]
                            doc_type_range = doc_type_range_list[i]
                            doc = doc_list[index-no_search][i]
                            # doc = doc_list[index-no_search][i]

                            doc_is_useful = _doc_useful_compare(sub_question, doc, self.reward_model, self.reward_tokenizer)
                            doc_is_useful_llm = (doc_type=='useful')

                            doc_reward = 1 if doc_is_useful == doc_is_useful_llm else -1
                            rollout_reward.append((doc_reward, doc_type_range, 'useful' if doc_is_useful else 'useless'))

                # store
                reward[id].append(rollout_reward)
        return reward
        '''
        question_answer_reward format
        {
            "sample_id_1": [[(1, (221, 222), correct_type), (...), ...], []=>no reward, ...],
            "sample_id_2": [[(-1, (122, 123), correct_type), (...), ...], ...],
            ...
        }
        '''

    def _compute_average_reward(self, all_rewards) -> None|float:
        """
        all_rewards: List[Dict[sample_id -> List[List[(reward, token_range, add_info)]]]]

        返回：
            avg_reward: float
        """
        total = 0.0
        count = 0

        for rank_reward in all_rewards:  # 遍历每个 rank 聚合的 reward dict
            for sample_id, rollout_list in rank_reward.items():  # 每个 sample 的 rollout 列表
                for rollout in rollout_list:  # rollout 是 list[(reward, token_range, info)]
                    # rollout 可能是空列表（格式错误）
                    for item in rollout:
                        reward = item[0]      # 取 (reward, token_range, info) 中的 reward
                        total += float(reward)
                        count += 1

        if count == 0:
            return None

        return total / count
    
    def _reward_exist_cal(self, all_rewards: list[dict[str, list]]):
        reward_exist_count = {}
        for process in all_rewards:
            for sample_id, rollouts in process.items():
                if sample_id not in reward_exist_count:
                    reward_exist_count[sample_id] = []
                count = 0
                for item in rollouts:
                    if item != []:
                        count+=1
                reward_exist_count[sample_id].append(count)
        for sample_id in reward_exist_count:
            reward_exist_count[sample_id] = min(reward_exist_count[sample_id])
        return [reward_exist_count]
    
    def _advantage_cal_global(self, all_rewards: list[dict[str, list]]) -> list[dict[str, list]]:
        '''
        Compute advantages for process-supervised GRPO using **global** (inter-group) normalization.
        All rewards across all sample_ids and rollouts are used to compute a single mean and std.
        
        Invalid rollouts (empty list) are skipped in baseline computation,
        and their advantages are set to empty list.

        :param all_rewards: same structure as before.
        :return: all_advantages with same structure, advantages normalized globally.
        '''
        # Step 1: Collect all valid reward scalars globally
        all_valid_rewards = []
        for process in all_rewards:
            for rollouts in process.values():
                for rollout_rewards in rollouts:
                    if not rollout_rewards:
                        continue
                    for step_reward in rollout_rewards:
                        all_valid_rewards.append(step_reward[0])

        # Step 2: Compute global mean and std
        if not all_valid_rewards:
            global_mean = None
            global_std = None
        else:
            n = len(all_valid_rewards)
            global_mean = sum(all_valid_rewards) / n
            if self.scale_rewards:
                variance = sum((r - global_mean) ** 2 for r in all_valid_rewards) / n
                global_std = math.sqrt(variance) + 1e-4
            else:
                global_std = 1.0

        # Step 3: Construct advantages using global stats
        all_advantages = []
        for process in all_rewards:
            new_process = {}
            for sample_id, rollouts in process.items():
                new_rollouts = []
                for rollout_rewards in rollouts:
                    if not rollout_rewards:
                        new_rollouts.append([])
                    else:
                        new_steps = [
                            ((reward - global_mean) / global_std, token_range, add_info)
                            for (reward, token_range, add_info) in rollout_rewards
                        ]
                        new_rollouts.append(new_steps)
                new_process[sample_id] = new_rollouts
            all_advantages.append(new_process)

        return all_advantages


    def _advantage_cal_local(self, all_rewards: list[dict[str, list]]) -> list[dict[str, list]]:
        '''
        Compute advantages for process-supervised GRPO.
        For each sample_id and each step index
        
        Invalid rollouts (empty list) are skipped in baseline computation,
        and their advantages are set to empty list.

        :param all_rewards: 
        [
            {
                "sample_id_1": [ [ (1 => reward, token_range, add_info), (...), ...], []=>no reward, ...],
                "sample_id_2": [ [ (1 => reward, token_range, add_info), (...), ...], ...],
                ...
            },
            {
                "sample_id_1": ... ,
                "sample_id_2": ... ,
                ...
            },
            ...
        ]

        Returns:
            all_advantages: same structure as all_rewards, but with advantages instead of rewards.
        '''
        # Step 1: Collect all valid rewards per sample_id (across all rollouts and all steps)
        sample_reward_values = {}
        for process in all_rewards:
            for sample_id, rollouts in process.items():
                if sample_id not in sample_reward_values:
                    sample_reward_values[sample_id] = []
                for rollout_rewards in rollouts:
                    # Skip empty rollout_rewards
                    if not rollout_rewards:
                        continue
                    # Each step_reward is (reward, token_range, add_info)
                    for step_reward in rollout_rewards:
                        sample_reward_values[sample_id].append(step_reward[0])
        
        # Step 2: Compute mean reward and std per sample_id
        sample_stats = {}
        for sample_id, rewards in sample_reward_values.items():
            if not rewards:
                sample_stats[sample_id] = {'mean': 0.0, 'std': 1.0}
            else:
                n = len(rewards)
                mean = sum(rewards) / n
                # Compute std (population std, not sample std)
                variance = sum((r - mean) ** 2 for r in rewards) / n
                if self.scale_rewards == True:
                    std = math.sqrt(variance) + 1e-4
                else:
                    std = 1.0
                sample_stats[sample_id] = {'mean': mean, 'std': std}

        # Step 3: Construct advantages with same structure as all_rewards
        all_advantages = []
        for process in all_rewards:
            new_process = {}
            for sample_id, rollouts in process.items():
                stats = sample_stats.get(sample_id, {'mean': 0.0, 'std': 1.0})
                mean = stats['mean']
                std = stats['std']
                new_rollouts = []
                for rollout_rewards in rollouts:
                    if not rollout_rewards:
                        new_rollouts.append([])
                    else:
                        new_steps = [
                            ((reward - mean) / std, token_range, add_info)
                            for (reward, token_range, add_info) in rollout_rewards
                        ]
                        new_rollouts.append(new_steps)
                new_process[sample_id] = new_rollouts
            all_advantages.append(new_process)

        return all_advantages
    

    def _policy_model_update(
        self, 
        batch_responses: dict[str, list[dict]],
        advantage: dict[str, list[list]],
        min_format_correct: dict[str, int],
    ) -> None:
        '''
        对每个 sample_id 的多个 rollouts，分 micro-batch 前向传播，
        _forward_policy_and_reference_micro_batch 返回每个 sample_id 对应的 (policy_logprobs_list, reference_logprobs_list)
        再进行反向传播计算梯度
        
        Returns:
            其中每个 list 的长度等于该 sample 的 rollout 数量
        '''
        micro_batch_size = self.policy_ref_micro_batch_size_per_gpu

        # 分组处理有效的 rollout
        valid_indices = []
        valid_tensors = []
        valid_tensors_input_len = []
        valid_advantages = []
        
        for sample_id, sample_rollouts in batch_responses.items():      
            max_count = min_format_correct[sample_id]
            count = 0
            for idx, item_advantage in enumerate(advantage[sample_id]):
                if count == max_count:
                    break
                if item_advantage != []:
                    valid_indices.append(idx)
                    valid_tensors.append(sample_rollouts[idx]["final_token_id_tensor"])
                    valid_tensors_input_len.append(sample_rollouts[idx]["input_token_len"])
                    valid_advantages.append(item_advantage)
                    count +=1

        for i in range(0, len(valid_tensors), micro_batch_size):
            chunk_tensors = valid_tensors[i:i + micro_batch_size]
            chunk_tensors_input_len = valid_tensors_input_len[i:i + micro_batch_size]
            chunk_indices = valid_indices[i:i + micro_batch_size]
            chunk_advantages = valid_advantages[i:i + micro_batch_size]
            
            policy_selected, ref_selected, advantages_selected = \
                self._forward_policy_and_reference_micro_batch(
                    chunk_tensors, 
                    chunk_tensors_input_len, 
                    chunk_advantages
                )
            loss_dict = self._para_update_step(
                policy_selected=policy_selected,
                ref_selected=ref_selected,
                advantages_selected=advantages_selected
            )
            self.logger.info(f"[rank {self.rank}] loss info: {loss_dict}")

        return
        '''
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'effective_tokens': advantages_mask.sum().item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        '''

    def _policy_model_update_old(
        self, 
        batch_responses: dict[str, list[dict]],
        advantage: dict[str, list[list]],
        min_reward_exist: dict[str, int],
    ) -> dict[str, tuple[list[Tensor], list[Tensor]]]:
        '''
        对每个 sample_id 的多个 rollouts，分 micro-batch 前向传播，
        返回每个 sample_id 对应的 (policy_logprobs_list, reference_logprobs_list)
        再进行反向传播计算梯度
        
        Returns:
            
            其中每个 list 的长度等于该 sample 的 rollout 数量
        '''
        micro_batch_size = self.policy_ref_micro_batch_size_per_gpu
        
        for sample_id, sample_rollouts in batch_responses.items():
            
            if min_reward_exist[sample_id]==0:
                self.logger.warning(f"[rank {self.rank}] sample_id={sample_id} get 0 rollout with reward ")
            
            # 对有效的 rollout 批量操作
            else:
                # 分组处理有效的 rollout
                valid_indices = []
                valid_tensors = []
                valid_tensors_input_len = []
                valid_advantages = []
                
                for idx, rollout in enumerate(sample_rollouts):
                    if "final_token_id_tensor" in rollout:
                        valid_indices.append(idx)
                        valid_tensors.append(rollout["final_token_id_tensor"])
                        valid_tensors_input_len.append(rollout["input_token_len"])
                        valid_advantages.append(advantage[sample_id][idx])
            
                # for i in range(0, len(valid_tensors), micro_batch_size):
                for i in range(0, min_reward_exist[sample_id], micro_batch_size):
                    chunk_tensors = valid_tensors[i:i + micro_batch_size]
                    chunk_tensors_input_len = valid_tensors_input_len[i:i + micro_batch_size]
                    chunk_indices = valid_indices[i:i + micro_batch_size]
                    chunk_advantages = valid_advantages[i:i + micro_batch_size]
                    
                    policy_selected, ref_selected, advantages_selected = \
                        self._forward_policy_and_reference_micro_batch(
                            chunk_tensors, 
                            chunk_tensors_input_len, 
                            chunk_advantages
                        )
                    loss_dict = self._para_update_step(
                        policy_selected=policy_selected,
                        ref_selected=ref_selected,
                        advantages_selected=advantages_selected
                    )
                    self.logger.info(f"[rank {self.rank}] loss info: {loss_dict}")
        return
        '''
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'effective_tokens': advantages_mask.sum().item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        '''


    def _build_advantages_and_mask(
        self,
        batch_size, 
        max_seq_len, 
        advantage_list, 
        token_id_input_len,
        seq_lengths,
        device
    ):
        """
        构建 advantages 和 advantages_mask 张量
        对于 response 中未显式标注的 token，设 advantage=0.2, mask=1
        
        Args:
            batch_size: 批次大小
            max_seq_len: 最大序列长度
            token_lengths: 每个样本的实际token长度列表
            advantage_list: advantage信息列表
            device: 设备
        
        Returns:
            advantages: Tensor, shape [B, L_max-1]
            advantages_mask: Tensor, shape [B, L_max-1]
        """
        advantages = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.float32)
        advantages_mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.int)
        
        
        for batch_idx in range(batch_size):
            prompt_len = token_id_input_len[batch_idx]
            total_len = seq_lengths[batch_idx]  # 实际序列长度（不含 padding）
            resp_start = prompt_len
            resp_end = total_len - 1
            
            # Step 1: 先处理显式标注的 advantages
            for advantage_info in advantage_list[batch_idx]:
                # 解析 advantage_info 格式: (advantage_value, (start_idx, end_idx), correct_type)
                if len(advantage_info) >= 2:
                    advantage_value = advantage_info[0]
                    token_indices = advantage_info[1]
                    
                    # 处理token索引范围（闭区间）
                    if isinstance(token_indices, tuple) and len(token_indices) == 2:
                        start_idx, end_idx = token_indices
                        # 闭区间：包括start_idx和end_idx
                    elif isinstance(token_indices, int):
                        start_idx = token_indices
                        end_idx = token_indices  # 单个token，闭区间就是它自己
                    else:
                        raise ValueError('Advantage range type unsupported.')
                    
                    # 设置advantages和mask（闭区间）
                    if end_idx >= start_idx:  # 确保有至少一个token
                        # 闭区间：包括start_idx和end_idx
                        for idx in range(start_idx, end_idx + 1):
                            advantages[batch_idx, idx] = advantage_value
                            advantages_mask[batch_idx, idx] = 1
                    else:
                        raise ValueError('Advantage range error, end_idx <= start_idx.')
            
            # Step 2: 对 response 区域中未被 mask 的位置，设 advantage=0.2, mask=1
            for idx in range(resp_start, min(resp_end + 1, max_seq_len)):
                if advantages_mask[batch_idx, idx] == 0:
                    advantages[batch_idx, idx] = 0.2
                    advantages_mask[batch_idx, idx] = 1
                    
        return advantages[:, 1:], advantages_mask[:, 1:]
    

    def _forward_policy_and_reference_micro_batch(
        self, 
        token_id_tensor_list: list[torch.Tensor], 
        token_id_input_len: list[int],
        advantage_list: list[list[tuple]]
    ):
        """
        对多个（长度不同的）token ID 序列，分别用 policy_model 和 reference_model 前向传播，
        返回每个序列中从第2个token开始的 log probability

        Args:
            token_id_tensor_list: List[Tensor[int]]. n 个样本，每个是 token ID 序列
            token_id_input_len: list[int]. n 个样本，每个是 token ID 序列中输入 token 的长度
            advantage_list: list[list[tuple]]. n 个样本，每个是 token ID 序列中 token index 对应的 advantage
                eg: [(advantage_value, (start_idx, end_idx), correct_type), ...]
                其中:
                - advantage_value: 浮点数，advantage值
                - (start_idx, end_idx): 元组，token索引范围（左闭右闭）
                - correct_type: 可选的正确类型标识

        Returns:
            policy_logprobs: Tensor, shape [B, L_max-1]
            ref_logprobs: Tensor, shape [B, L_max-1]
            advantages: Tensor, shape [B, L_max-1]
            advantages_mask: Tensor, shape [B, L_max-1]
        """
        batch_size = len(token_id_tensor_list)
        
        # 1. Pad sequences
        input_ids = pad_sequence(
            token_id_tensor_list,
            batch_first=True,
            padding_value=self.policy_tokenizer.pad_token_id
        ) # shape [B, L_max]
        attention_mask = (input_ids != self.policy_tokenizer.pad_token_id).long()  # [B, L_max]

        
        L_max = input_ids.shape[1]
        device = input_ids.device
        
        # 2. 构建 advantages 和 mask
        seq_lengths = [len(item) for item in token_id_tensor_list]
        shifted_advantages, shifted_advantages_mask = self._build_advantages_and_mask(
            batch_size=batch_size,
            max_seq_len=L_max,
            advantage_list=advantage_list,
            token_id_input_len=token_id_input_len,
            seq_lengths=seq_lengths,
            device=device
        )

        # --- Policy Model Forward ---
        policy_outputs = self.policy_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False
        )
        policy_logits = policy_outputs.logits  # [B, L_max, Vocab_size]

        # --- Reference Model Forward (nograd) ---
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False
            )
        ref_logits = ref_outputs.logits  # [B, L_max, Vocab_size]

        # Shift: labels are input_ids[:, 1:], logits are logits[:, :-1]
        shifted_logits_policy = policy_logits[:, :-1, :]  # [B, L_max-1, V]
        shifted_logits_ref = ref_logits[:, :-1, :]        # [B, L_max-1, V]
        shifted_labels = input_ids[:, 1:]                 # [B, L_max-1, V]

        needed_indices = torch.nonzero(shifted_advantages_mask) # shape [N, 2]
        needed_b = needed_indices[:, 0]
        needed_l = needed_indices[:, 1]
        
        needed_policy_logits = shifted_logits_policy[needed_b, needed_l, :] # shape [N, V]
        needed_ref_logits = shifted_logits_ref[needed_b, needed_l, :]       # shape [N, V]
        needed_labels = shifted_labels[needed_b, needed_l]                  # shape [N]
        
        # 计算logprobs
        needed_policy_logprobs = torch.log_softmax(needed_policy_logits, dim=-1) # shape [N, V]
        needed_ref_logprobs = torch.log_softmax(needed_ref_logits, dim=-1)       # shape [N, V]
        
        # 提取目标token的logprobs
        policy_selected = needed_policy_logprobs[torch.arange(len(needed_b)), needed_labels]  # shape [N]
        ref_selected = needed_ref_logprobs[torch.arange(len(needed_b)), needed_labels]        # shape [N]

        advantages_selected = shifted_advantages[needed_b, needed_l]    # shape [N]
        
        return policy_selected, ref_selected, advantages_selected

        
    def _checkpoint_save(
        self, 
        middle_save: bool, 
        step: int, 
        task_info_list: list[tuple]|None = None
    ):
        """
        Save LoRA + tokenizer + scheduler checkpoint in rank-0 only.
        Works for DDP-wrapped PEFT model.

        middle_save: whether this is an intermediate checkpoint
        step: current training step (int)
        """
        if self.rank != 0:
            return  # only rank 0 saves

        # ==== 1. prepare save dir ====
        if middle_save:
            save_dir = os.path.join(self.model_save_path, f"checkpoint_step_{step}")
        else:
            save_dir = os.path.join(self.model_save_path, "checkpoint_final")

        if step == -1:
            save_dir = os.path.join(self.model_save_path, "checkpoint_init")

        os.makedirs(save_dir, exist_ok=True)

        # ==== 2. unwrap DDP & get PEFT model ====
        # self.policy_model is DDP(PEFT(model))
        peft_model = self.policy_model.module  # unwrap DDP

        # ==== 3. save lora adapter ====
        model_dir = os.path.join(save_dir, "model")
        peft_model.save_pretrained(model_dir)
        self.logger.info(f"[rank 0] LoRA adapter saved to {model_dir}")

        # ==== 4. save tokenizer ====
        self.policy_tokenizer.save_pretrained(model_dir)
        self.logger.info(f"[rank 0] tokenizer saved to {model_dir}")

        # ==== 6. also save step, configs etc ====
        meta_path = os.path.join(save_dir, "meta.json")
        json.dump({
            "step": step,
            "middle_save": middle_save,
            "task_info_list": task_info_list,
        }, open(meta_path, "w"), indent=4)

        # self.logger.info(f"[rank 0] checkpoint saved at {save_dir}")

    
    def train(self):
        # construct dataloader
        def identity_collate_fn(batch):
            new_batch = {sample["id"]: sample for sample in batch}
            return new_batch
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=identity_collate_fn
        )

        '''Rollout Prepare Stage'''
        # Calculate rollout parameters for each process
        # Total rollout number divided among processes
        rollout_per_pro = self.rollout_num // self.world_size # rollout_per_pro per sample
        remainder = self.rollout_num % self.world_size
        # Check if divisible; if not, issue a warning (only on rank 0)
        if self.rank == 0 and remainder != 0:
            self.logger.error(f"Rollout number ({self.rollout_num}) not evenly divisible by world size (processes num) ({self.world_size}). "
                        f"First {remainder} processes will handle one extra rollout.")
            raise ValueError("Rollout number ({self.rollout_num}) not evenly divisible by world size.")
        if rollout_per_pro ==0:
            raise ValueError(f"Process ([rank {self.rank}]) face 0 rollout")
    
        rollout_info = []
        task_info_list = []
        

        if self.rank == 0:
            self._checkpoint_save(middle_save=True, step=-1, task_info_list=task_info_list)
        dist.barrier()

        for step, batch in enumerate(dataloader):
            
            '''Training Task Select Stage (in rank 0)'''
            self.policy_model.train()
            # choose task in rank 0 for Step = step
            if self.rank == 0:
                current_task, probs_dict = self.schedular.select_task()
                # current_task = TrainTaskType.DocType
            else:
                current_task = None
                probs_dict = None
            # record select info
            task_info = [current_task]
            if current_task!=None:
                task_info_list.append((current_task.value, probs_dict))
            else:
                task_info_list.append((current_task, probs_dict))
            
            
            '''Training Task Broadcast Stage (from rank 0 to all ranks)'''
            # broadcast current_task from rank 0 to other process
            dist.broadcast_object_list(task_info, src=0)
            current_task = task_info[0]
            self.logger.info(f"[rank MAIN]  [Step {step}] [Current task: {current_task}]")
            step_info = {"step": step, "rank": self.rank, "task": current_task.value}

            
            '''Rollout Stage'''
            # Rollout in each process - each GPU does its portion
            batch_responses = self._rollout(batch, rollout_per_pro, current_task)
            self.logger.info(f"[rank {self.rank}] {rollout_per_pro*len(batch_responses)} rollout finished")
            step_info["batch_responses"] = batch_responses


            '''Reward Calculate Stage'''
            reward: dict[str, list] = self._reward_cal(batch_responses, batch, current_task)
            self.logger.info(f"[rank {self.rank}] reward calculated")
            step_info["reward"] = reward

            
            '''Reward Aggregate Stage (from all rank to rank 0)'''
            if self.rank == 0:
                all_rewards = [None] * self.world_size
                # dist.gather_object(..., dst=0) 会将所有进程（包括 rank 0 自己）的 reward 对象都收集到 all_rewards 列表中
                # in rank 0: all_rewards: list[dict[str, list]]
                dist.gather_object(reward, object_gather_list=all_rewards, dst=0)
            else:
                dist.gather_object(reward, dst=0)
            

            '''Advantage Calculate Stage (in rank 0)'''
            if self.rank == 0:
                all_advantages: list[dict[str, list]] = self._advantage_cal_global(all_rewards)
                # self._reward_exist_cal 计算所有 rank 中 reward_exist 的 rollout 数量 的最小值
                # rollout 会有失败情况
                # 不同 rank 成功数量，不经过裁剪可能会导致 DistributedDataParallel all_reduce 时直接报错（不同进程 all_reduce 次数不同）
                # 因此，需要计算所有 rank 中 reward_exist 的 rollout 数量 的最小值，强制裁剪每个 rank 的 rollout，来保证所有 rank 的 rollout 数量完全一致
                min_reward_exist = self._reward_exist_cal(all_rewards) 
            else:
                all_advantages = [None]*self.world_size
                min_reward_exist = [None]
        

            '''Advantage Broadcast Stage (from rank 0 to all ranks)'''
            dist.broadcast_object_list(min_reward_exist, src=0)
            min_reward_exist = min_reward_exist[0]
            dist.broadcast_object_list(all_advantages, src=0)
            advantage: dict[str, list] = all_advantages[self.rank]
            self.logger.info(f"[rank MAIN] Advantage calculated")

            step_info["advantage"] = advantage
            if self.rank == 0:
                step_info['all_advantages'] = all_advantages
                step_info['min_reward_exist'] = min_reward_exist


            ''' Store Rollout Info Stage'''
            rollout_info.append(step_info)
            if (step+1)%self.rollout_info_save_step==0:
                dump_rollout_json(rollout_info, self.rollout_info_save_path)
            
            
            '''Back Propogation Stage'''
            self._policy_model_update(batch_responses, advantage, min_reward_exist)
            self.logger.info(f"[rank {self.rank}] policy model parameter updated")


            '''Task Weight Update Stage -- for Step = step+1 (in rank 0)'''
            if self.rank == 0:
                ave_reward: float = self._compute_average_reward(all_rewards)
                self.schedular.after_training_update(ave_reward)
                self.logger.info(f"[rank MAIN] ave_reward={ave_reward}, task schedular updated")


            '''Check Point Save (in rank 0)'''
            dist.barrier() 
            if step%self.model_save_steps == 0 and step != 0:
                if self.rank == 0:
                    self._checkpoint_save(middle_save=True, step=step, task_info_list=task_info_list)
                self.logger.info(f'[rank MAIN] Checkpoint Saved')
            dist.barrier() 
        

        '''Final Result Save (in rank 0)'''
        dist.barrier() 
        if self.rank == 0:
            self._checkpoint_save(middle_save=False, step=step, task_info_list=task_info_list)
        self.logger.info(f'[rank MAIN] Final Result Save')
        dist.barrier()
        
        return


# usage test
if __name__ =="__main__":
    
    import argparse

    def parse_training_args():
        parser = argparse.ArgumentParser()

        # —————— 路径配置 ——————
        parser.add_argument("--theme", type=str)
        parser.add_argument("--train_dataset_path", type=str)
        parser.add_argument("--system_prompt_path", type=str)
        parser.add_argument("--policy_model_path", type=str)
        parser.add_argument("--reward_model_path", type=str)
        parser.add_argument("--model_save_path", type=str)
        parser.add_argument("--rollout_info_save_dir", type=str)

        # —————— LoRA 配置（拆开以便传参）——————
        parser.add_argument("--lora_r", type=int)
        parser.add_argument("--lora_alpha", type=int)
        parser.add_argument("--lora_dropout", type=float)
        
        # —————— 训练控制参数 ——————
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--rollout_num", type=int)
        parser.add_argument("--rollout_micro_batch_size_per_gpu", type=int)
        parser.add_argument("--policy_ref_micro_batch_size_per_gpu", type=int)

        # —————— 生成与采样参数 ——————
        parser.add_argument("--max_token_len", type=int)
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--top_p", type=float)
        parser.add_argument("--top_k", type=int)

        # —————— 保存与日志 ——————
        parser.add_argument("--model_save_steps", type=int)
        parser.add_argument("--rollout_info_save_step", type=int)

        return parser.parse_args()


    # 解析参数
    args = parse_training_args()

    # 设置路径配置
    os.makedirs(f'logs/{args.theme}', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'logs/{args.theme}/{str(os.environ['RANK'])}.log',  # 指定日志文件路径
        filemode='w'                 # （默认是 'a'追加模式，也可设为 'w' 覆盖模式）
    )
    
    load_dotenv('../.env')
    e5_model_path = os.getenv("E5_MODEL_PATH")


    # 构建 LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM
    )

    # 赋值回原变量名
    train_dataset_path = args.train_dataset_path
    system_prompt_path = args.system_prompt_path
    policy_model_path = args.policy_model_path
    reward_model_path = args.reward_model_path
    model_save_path = args.model_save_path
    rollout_info_save_dir = args.rollout_info_save_dir

    batch_size = args.batch_size
    rollout_num = args.rollout_num
    rollout_micro_batch_size_per_gpu = args.rollout_micro_batch_size_per_gpu
    policy_ref_micro_batch_size_per_gpu = args.policy_ref_micro_batch_size_per_gpu

    max_token_len = args.max_token_len
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k

    model_save_steps = args.model_save_steps
    rollout_info_save_step = args.rollout_info_save_step

    scale_rewards = True

    

    # 实例化 Trainer
    trainer = GRPOTrainer(
        # 数据与提示
        train_dataset_path=train_dataset_path,
        system_prompt_path=system_prompt_path,

        # 模型路径
        policy_model_path=policy_model_path,
        reward_model_path=reward_model_path,
        e5_model_path=e5_model_path,

        # LoRA 配置
        lora_config=lora_config,

        # 保存路径与频率
        model_save_path=model_save_path,
        model_save_steps=model_save_steps,
        rollout_info_save_dic=rollout_info_save_dir,
        rollout_info_save_step=rollout_info_save_step,

        # 批次与 rollout 控制
        batch_size=batch_size,
        rollout_num=rollout_num,
        rollout_micro_batch_size_per_gpu=rollout_micro_batch_size_per_gpu,
        policy_ref_micro_batch_size_per_gpu=policy_ref_micro_batch_size_per_gpu,

        # 奖励设置
        scale_rewards=scale_rewards,

        # 生成参数
        max_token_len=max_token_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    trainer.train()

    # 释放 GPU 显存或通信资源
    torch.distributed.destroy_process_group()

    print("Training Finished")