
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch



class AdamC(optim.Adam):
    """
    实现了论文 "Why Gradients Rapidly Increase Near the End of Training" 中提出的 AdamC 优化器。
    
    这个优化器对指定参数组的权重衰减进行了修正，使其与学习率解耦。
    修正公式为: effective_weight_decay = base_weight_decay * current_lr / max_lr”
    这玩意没有用，垃圾论文
    """

    def __init__(self, parmas, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, *, gamma_max: float, **kwargs):
        super().__init__(parmas, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
        self.gamma_max = gamma_max

        for group in self.param_groups:
            group['initial_weight_decay'] = group['weight_decay']


    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步优化
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            if group.get('apply_correction', False):
                gamma_t = group['lr']
                lammbda_ = group['initial_weight_decay']

                corrected_decay = lammbda_ * (gamma_t / self.gamma_max)

                group['weight_decay'] = corrected_decay

        super().step(closure)

        for group in self.param_groups:
            if group.get('apply_correction', False):
                group['weight_decay'] = group['initial_weight_decay']

        return loss
    

def get_parameter_groups(model, cfg):
    weight_decay = cfg.weight_decay

    decay_dan_decay, decay_stat_decay, no_decay = [], [], []
    name_dan_decay, name_stat_decay, name_no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # bias、LayerNorm、embedding 不加 decay
        if (
            name.endswith(".bias")
            or "LayerNorm.weight" in name
            or "layernorm" in name.lower()
            or "embedding" in name.lower()
            or "emb" in name.lower()
        ):
            no_decay.append(param)
            name_no_decay.append(name)
        elif(
            "forward_layers" in name
            or "attention" in name
        ):
            decay_dan_decay.append(param)
            name_dan_decay.append(name)
        else:
            decay_stat_decay.append(param)
            name_stat_decay.append(name)


    group = [
        {"params": decay_dan_decay, "weight_decay": weight_decay, "apply_correction": True},
        {"params": decay_stat_decay, "weight_decay": weight_decay, "apply_correction": False},
        {"params": no_decay, "weight_decay": 0.0, "apply_correction": False},
    ]
    print(f"Name with decay: {name_dan_decay}")
    print(f"Name with stat decay: {name_stat_decay}")
    print(f"Name without decay: {name_no_decay}")

    return group

def get_optim(cfg, model):
    """
    Get optimizer based on the configuration.
    """
    args = cfg.optimizer
    if args.type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.betas[0], args.betas[1]),
        )
    elif args.type == 'adamw':
        optimizer = optim.AdamW(
            get_parameter_groups(model, args),
            lr=args.lr,
            betas=(args.betas[0], args.betas[1])
        )
    elif args.type == 'adamc':
        optimizer = AdamC(
            get_parameter_groups(model, args),
            lr=args.lr,
            betas=(args.betas[0], args.betas[1]),
            gamma_max=args.lr,
            weight_decay=args.weight_decay
        )
    else: 
        raise ValueError(f"Unsupported optimizer type: {args.type}")

    return optimizer

def get_cosine_schedule_with_warmup(optimizer, cfg, num_training_steps):
    """
    带warmup的余弦退火学习率调度器
    """
    num_warmup_steps = cfg.warmup_steps 
    min_lr_ratio = cfg.min_lr_ratio

    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
