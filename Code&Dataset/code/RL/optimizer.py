import torch
import torch.distributed as dist
from typing import Dict, Tuple
import logging


class GRPOOptimizer:
    """
    正确的GRPO优化器，依赖DDP自动梯度同步
    """
    
    def __init__(
        self,
        model,  # 必须是DDP包装的LoRA模型
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        T_max: int = 1000,
        eta_min: float = 1e-5,
        kl_coef: float = 0.0, # trl config 
        clip_epsilon: float = 0.2, # trl config
        clip_epsilon_high: float|None = 0.28 # trl config (DAPO recommend)
    ):
        """
        初始化GRPO优化器
        
        注意：policy model必须是DDP包装的模型
        """
        self.model = model
        
        # 验证输入模型是 DDP
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            raise ValueError("policy model必须是DDP包装的模型！")
        
        # hyperparameter for optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._setup_optimizer(self.learning_rate, self.weight_decay)

        # learning rate schedular
        self.T_max = T_max
        self.eta_min = eta_min
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min
        )
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # hyperparameters for loss calculation
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.clip_epsilon_high = clip_epsilon if clip_epsilon_high == None else clip_epsilon_high
        
        # init logger
        self.logger = logging.getLogger(__name__)
        total_params = 0
        trainable_params = 0
        for p in self.model.module.parameters():
            num = p.numel()
            total_params += num
            if p.requires_grad:
                trainable_params += num
        trainable_ratio = trainable_params / total_params * 100
        self.logger.info(f"GRPO Optimizer Intialized - Trainable Parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")

    def _setup_optimizer(self, learning_rate: float, weight_decay: float):
        """设置优化器"""
        lora_params = [p for p in self.model.parameters() if p.requires_grad]

        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def _compute_grpo_loss(
        self,
        policy_selected: torch.Tensor, # [N]
        ref_selected: torch.Tensor, # [N]
        advantages_selected: torch.Tensor, # [N]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算GRPO损失
        """
        # print("advantages_selected", advantages_selected)
        # print("policy_selected", policy_selected)
        # print("ref_selected", ref_selected)
        # 1. 计算重要性采样比率 element-wise
        log_ratio = policy_selected - ref_selected.detach()
        ratio = torch.exp(log_ratio)
        
        # 5. 计算损失 (DAPO Loss)
        surr1 = ratio * advantages_selected
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon_high) * advantages_selected
        policy_loss = -torch.min(surr1, surr2).sum() / advantages_selected.shape[0]
        # print("policy_loss", policy_loss)
        # if policy_loss == 0:
        #     print(surr1, surr2)


        # 6. 计算KL散度(使用近似方法估计)
        if self.kl_coef!= 0:
            kl_div = torch.exp(-log_ratio) + log_ratio - 1
            kl_div = kl_div.sum() / advantages_selected.shape[0]
        else:
            kl_div = torch.zeros(1, device=ratio.device)

        # 7. 总损失
        total_loss = policy_loss + self.kl_coef * kl_div
        
        # 8. 统计信息
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item() if self.kl_coef!= 0 else None,
            'effective_tokens': advantages_selected.shape[0],
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return total_loss, loss_dict
    
    def _para_update_step(
            self,
            policy_selected: torch.Tensor, # [N]
            ref_selected: torch.Tensor, # [N]
            advantages_selected: torch.Tensor, # [N]
        ) -> Tuple[torch.Tensor, Dict]:
        """
        执行一步参数更新
        
        依赖DDP自动梯度同步，不需要手动同步
        """
        
        # 计算损失
        total_loss, loss_dict = self._compute_grpo_loss(
            policy_selected=policy_selected,
            ref_selected=ref_selected,
            advantages_selected=advantages_selected
        )
        
        # 反向传播 -- DDP会自动同步所有GPU的梯度
        total_loss.backward()
        # 参数更新
        self.optimizer.step()
        # 学习率更新
        self.scheduler.step()
        # 清空梯度
        self.optimizer.zero_grad()

        return loss_dict