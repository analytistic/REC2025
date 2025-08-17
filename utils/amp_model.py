"""
支持混合精度训练的模型包装器
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, Tuple, List, Any


class AMPModelWrapper(nn.Module):
    """
    混合精度训练模型包装器
    自动处理前向传播中的混合精度
    """
    
    def __init__(self, model: nn.Module, use_amp: bool = True):
        super().__init__()
        self.model = model
        self.use_amp = use_amp and torch.cuda.is_available()
        
    def forward(self, *args, **kwargs):
        """前向传播，自动应用混合精度"""
        if self.use_amp:
            with autocast():
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """代理属性访问到包装的模型"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class AMPLossWrapper:
    """
    混合精度训练损失包装器
    """
    
    def __init__(self, loss_fn, use_amp: bool = True):
        self.loss_fn = loss_fn
        self.use_amp = use_amp and torch.cuda.is_available()
    
    def __call__(self, *args, **kwargs):
        """计算损失，自动应用混合精度"""
        if self.use_amp:
            with autocast():
                return self.loss_fn(*args, **kwargs)
        else:
            return self.loss_fn(*args, **kwargs)


def setup_amp_model(model: nn.Module, use_amp: bool = True) -> Tuple[nn.Module, bool]:
    """
    设置模型以支持混合精度训练
    
    Args:
        model: 要包装的模型
        use_amp: 是否使用混合精度
        
    Returns:
        包装后的模型和是否启用AMP的标志
    """
    amp_enabled = use_amp and torch.cuda.is_available()
    
    if amp_enabled:
        # 检查模型中是否有不兼容的层
        incompatible_layers = []
        for name, module in model.named_modules():
            # 某些层可能不兼容float16
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                 nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                                 nn.InstanceNorm2d, nn.InstanceNorm3d)):
                incompatible_layers.append(name)
        
        if incompatible_layers:
            print(f"Found normalization layers that will remain in float32: {incompatible_layers[:5]}...")
        
        wrapped_model = AMPModelWrapper(model, use_amp=True)
        print("Model wrapped for mixed precision training")
    else:
        wrapped_model = model
        print("Mixed precision training disabled")
    
    return wrapped_model, amp_enabled


def optimize_model_for_amp(model: nn.Module) -> nn.Module:
    """
    优化模型以更好地支持混合精度训练
    
    Args:
        model: 要优化的模型
        
    Returns:
        优化后的模型
    """
    # 确保所有权重初始化合理，避免梯度溢出
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # 使用Xavier初始化，有助于混合精度训练的稳定性
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用正态分布初始化
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].fill_(0)
    
    return model


class AMPTrainingManager:
    """
    混合精度训练管理器
    整合模型、优化器和损失函数的混合精度训练
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 use_amp: bool = True,
                 **scaler_kwargs):
        self.model, self.amp_enabled = setup_amp_model(model, use_amp)
        self.optimizer = optimizer
        
        if self.amp_enabled:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(**scaler_kwargs)
        else:
            self.scaler = None
            
    def training_step(self, 
                     loss_fn,
                     inputs: Any,
                     clip_norm: Optional[float] = None) -> torch.Tensor:
        """
        执行一步训练
        
        Args:
            loss_fn: 损失函数
            inputs: 模型输入
            clip_norm: 梯度裁剪范数
            
        Returns:
            损失值
        """
        self.optimizer.zero_grad()
        
        if self.amp_enabled:
            with autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, inputs)
            
            self.scaler.scale(loss).backward()
            
            if clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
            self.optimizer.step()
        
        return loss
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
