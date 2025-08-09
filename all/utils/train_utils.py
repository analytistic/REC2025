import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    带warmup的余弦退火学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup_restarts(optimizer, num_warmup_steps, T_0, T_mult=1, eta_min_ratio=0.0, restart_ratio=0.6):
    """
    带warmup和重启的余弦退火学习率调度器 (SGDR)，支持restart_ratio
    
    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        T_0: 第一个重启周期的步数
        T_mult: 每次重启后周期长度的倍数
        eta_min_ratio: 最小学习率比例
        restart_ratio: 每次重启最大lr的缩放因子
    """
    def lr_lambda(current_step):
        # Warmup阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦退火重启阶段
        step_after_warmup = current_step - num_warmup_steps
        
        # 计算当前在第几个周期以及周期内的位置
        T_i = T_0
        T_cur = step_after_warmup
        epoch_i = 0
        
        while T_cur >= T_i:
            T_cur -= T_i
            epoch_i += 1
            T_i *= T_mult
        
        # restart_ratio控制每次重启最大lr递减
        max_lr_scale = restart_ratio ** epoch_i
        progress = T_cur / T_i
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max_lr_scale * (eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor)
    
    return LambdaLR(optimizer, lr_lambda)


def compute_gradient_stats(model):
    """
    计算模型梯度统计信息
    """
    total_norm = 0.0
    param_count = 0
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            grad_norms.append(param_norm.item())
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    max_grad_norm = np.max(grad_norms) if grad_norms else 0.0
    min_grad_norm = np.min(grad_norms) if grad_norms else 0.0
    
    return {
        'total_norm': total_norm,
        'avg_grad_norm': avg_grad_norm,
        'max_grad_norm': max_grad_norm,
        'min_grad_norm': min_grad_norm,
        'param_count': param_count
    }


def create_optimizer(model, config):
    """
    根据配置创建优化器
    """
    optimizer_config = config['optimizer']
    optimizer_type = optimizer_config['type']
    
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            betas=tuple(optimizer_config['betas']),
            weight_decay=optimizer_config['weight_decay'],
            eps=optimizer_config['eps']
        )
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config, total_steps):
    """
    根据配置创建学习率调度器
    """
    scheduler_config = config['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_config['warmup_steps'],
            num_training_steps=total_steps,
            min_lr_ratio=scheduler_config['min_lr_ratio']
        )
    elif scheduler_type == "cosine_with_warmup_restarts" or scheduler_type == "cosine_restarts":
        scheduler = get_cosine_schedule_with_warmup_restarts(
            optimizer,
            num_warmup_steps=scheduler_config['warmup_steps'],
            T_0=scheduler_config['T_0'],
            T_mult=scheduler_config.get('T_mult', 1),
            eta_min_ratio=scheduler_config.get('eta_min_ratio', 0.0),
            restart_ratio=scheduler_config.get('restart_ratio', 1.0)
        )
    elif scheduler_type == "constant":
        scheduler = LambdaLR(optimizer, lambda step: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def log_gradient_stats(model, writer, global_step, log_freq=100):
    """
    记录梯度统计信息到TensorBoard
    """
    if global_step % log_freq == 0:
        grad_stats = compute_gradient_stats(model)
        writer.add_scalar('Gradient/total_norm', grad_stats['total_norm'], global_step)
        writer.add_scalar('Gradient/avg_norm', grad_stats['avg_grad_norm'], global_step)
        writer.add_scalar('Gradient/max_norm', grad_stats['max_grad_norm'], global_step)
        writer.add_scalar('Gradient/min_norm', grad_stats['min_grad_norm'], global_step)
        return grad_stats
    return None


def clip_gradients(model, max_norm):
    """
    梯度裁剪
    """
    if max_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return grad_norm.item()
    return 0.0
