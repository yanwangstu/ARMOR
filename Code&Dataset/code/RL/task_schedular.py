import math
import random
from typing import Dict, Tuple
from enum import Enum

class TrainTaskType(Enum):
    QuestionSearch = "QuestionSearchVerdict"
    DocType = "DocTypeClassify"
    QuestionAnswer = "QuestionAnswerGeneration"

class TeacherAgent:
    def __init__(
        self,
        total_steps: int,
        tasks: Tuple[TrainTaskType, ...] = tuple(TrainTaskType),
        exploration_hyperpara: float|None = None,
        forget_hyperpara: float|None = None,
    ):
        """
        初始化自适应任务调度器（Teacher Agent）
        
        Args:
            total_steps: 任务调度轮次数
            tasks: 任务名称元组
        """
        self.tasks = tasks
        self.tasks_num = len(tasks)
        self.total_steps = total_steps

        # 初始化历史奖励和权重
        self.rewards: Dict[TrainTaskType, list] = {task: [] for task in tasks}
        self.weights: Dict[TrainTaskType, list] = {task: [1] for task in tasks}  # t=0 时为均一初始权重
        self.select_tasks = []
        self.current_task = None
        self.current_select_probs = None

        # 当前轮次 t（轮次 t 从 0 开始，-1 表示 __init__ ）
        self.t = -1

        # 确定超参数 如果没有提供，使用理论遗憾界成立的推荐值
        self.exploration_hyperpara = (
            exploration_hyperpara if exploration_hyperpara is not None 
            else self._exploration_hyperpara_cal()
        )
        self.forget_hyperpara = (
            forget_hyperpara if forget_hyperpara is not None 
            else self._forget_hyperpara_cal()
        )

    def _exploration_hyperpara_cal(self) -> float:
        """计算探索系数 gamma"""
        numerator = self.tasks_num * math.log(self.tasks_num * self.total_steps)
        gamma = math.sqrt(numerator / self.total_steps)
        return min(1.0, gamma)
    
    def _forget_hyperpara_cal(self) -> float:
        """计算遗忘系数 alpha"""
        return 1/self.total_steps

    def _get_selection_probabilities(self) -> Dict[str, float]:
        """计算每个任务的选择概率 p_t(τ)"""
        total_weight = sum(self.weights[task][self.t] for task in self.tasks)
        probs = {}
        for task in self.tasks:
            exploit = (1 - self.exploration_hyperpara) * (self.weights[task][self.t] / total_weight)
            explore = self.exploration_hyperpara / self.tasks_num
            probs[task.value] = exploit + explore
        return probs

    def select_task(self) -> Tuple[str, Dict[str, float]]:
        """
        根据当前权重选择一个任务进行训练。
        
        Returns:
            selected_task: 被选中的任务名
            probabilities: 所有任务的选择概率（用于后续重要性加权）
        """
        self.t+=1
        probs_dict = self._get_selection_probabilities()
        probs_list = [probs_dict[task.value] for task in self.tasks]
        tasks_list = list(self.tasks)

        # 按概率分布随机选择
        self.current_task = random.choices(tasks_list, weights=probs_list, k=1)[0]
        self.select_tasks.append(self.current_task)
        self.current_select_probs = probs_dict
        return self.current_task, probs_dict
    
    def select_task_random(self) -> Tuple[str, None]:
        """
        按照相同概率随机选择一个任务进行训练。
        
        Returns:
            selected_task: 被选中的任务名
        """
        self.t+=1
        probs_list = [1/len(self.tasks) for task in self.tasks]
        tasks_list = list(self.tasks)

        # 按概率分布随机选择
        self.current_task = random.choices(tasks_list, weights=probs_list, k=1)[0]
        print(self.current_task)
        self.select_tasks.append(self.current_task)
        return self.current_task, None

    def select_task_sequential(self) -> Tuple[str, None]:
        """
        顺序选择一个任务进行训练。
        
        Returns:
            selected_task: 被选中的任务名
        """
        self.t+=1
        tasks_list = list(self.tasks)

        self.current_task = tasks_list[self.t%len(self.tasks)]
        self.select_tasks.append(self.current_task)
        return self.current_task, None

    def _get_max_old_reward(self, task: str) -> float:
        """获取任务 task 在历史中的最大奖励（r_old）"""
        try:
            max_old_reward = max(x for x in self.rewards[task] if x is not None)
        except ValueError:
            max_old_reward = None
        return max_old_reward

    def after_training_update(self, current_reward: float|None):
        """
        在一轮训练结束后更新内部状态。
        
        Args:
            selected_task: 本轮训练的任务
            current_reward: 该任务本轮获得的奖励 r_t(selected_task)
            selection_probs: 本轮任务选择概率 p_t(τ)
        """
        # 0. find r_old for current training task
        r_old = self._get_max_old_reward(self.current_task)

        # 1. record reward in the current step
        for task in self.tasks:
            if task == self.current_task:
                self.rewards[task].append(current_reward)
            else:
                self.rewards[task].append(None)
        
        if current_reward != None:
            # 2. 计算 self.current_task 的 ALP & \hat{r}
            if r_old == None:
                # 第一次出现，ALP 用绝对奖励
                alp = abs(current_reward)
            else:
                alp = abs(current_reward - r_old)
            hat_r = alp/self.current_select_probs[self.current_task.value]

            # 3. 更新所有任务的权重 w_{t+1}(τ)
            total_weight = sum(self.weights[task][self.t] for task in self.tasks)
            mean_weight = (math.e*self.forget_hyperpara/self.tasks_num)*total_weight
            for task in self.tasks:
                if task == self.current_task:
                    update_weight = self.weights[task][self.t]*math.exp(self.exploration_hyperpara * hat_r / self.tasks_num)
                else:
                    update_weight = self.weights[task][self.t]
                new_weight = update_weight+mean_weight
                self.weights[task].append(new_weight)
        
        else:
            for task in self.tasks:
                new_weight = self.weights[task][self.t]
                self.weights[task].append(new_weight)

        return