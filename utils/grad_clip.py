
import numpy as np
import torch

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

    return None


def clip_gradients(model, max_norm):
    """
    梯度裁剪
    """
    if max_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return None