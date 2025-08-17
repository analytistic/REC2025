"""
混合精度训练辅助工具
"""
import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any


class AMPManager:
    """混合精度训练管理器"""
    
    def __init__(self, 
                 enabled: bool = True,
                 init_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        初始化AMP管理器
        
        Args:
            enabled: 是否启用混合精度训练
            init_scale: 初始缩放因子
            growth_factor: 缩放因子增长倍数
            backoff_factor: 缩放因子回退倍数
            growth_interval: 缩放因子更新间隔
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
        else:
            self.scaler = None
            
        print(f"Mixed precision training: {'enabled' if self.enabled else 'disabled'}")
        if self.enabled:
            print(f"Initial scale: {init_scale}")
    
    def get_context(self):
        """获取自动混合精度上下文管理器"""
        if self.enabled:
            return autocast()
        else:
            return torch.no_grad() if not torch.is_grad_enabled() else NullContext()
    
    def scale_loss(self, loss):
        """缩放损失"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss):
        """执行反向传播"""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def unscale_gradients(self, optimizer):
        """取消梯度缩放"""
        if self.enabled:
            self.scaler.unscale_(optimizer)
    
    def step(self, optimizer):
        """执行优化器步骤"""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        if self.enabled:
            return {'scaler': self.scaler.state_dict()}
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        if self.enabled and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])


class NullContext:
    """空上下文管理器"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def check_amp_compatibility():
    """检查混合精度训练兼容性"""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, mixed precision training will be disabled")
        return False
    
    # 检查GPU计算能力
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 7:  # Tensor Core需要计算能力7.0+
        print(f"Warning: GPU compute capability {device_capability[0]}.{device_capability[1]} < 7.0, "
              "mixed precision may not provide significant speedup")
    
    # 检查PyTorch版本
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (1, 6):
        print("Warning: PyTorch version < 1.6, automatic mixed precision not supported")
        return False
    
    print("Mixed precision training is compatible with your system")
    return True


def profile_memory_usage():
    """分析显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    return 0, 0
