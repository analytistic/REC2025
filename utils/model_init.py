import torch
def init_model_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            # Embedding层：正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].fill_(0)
                
        elif isinstance(module, torch.nn.Linear):
            # Linear层：根据后续激活函数选择
            if 'gate' in name.lower() or 'gater' in name.lower():
                # 门控层：Xavier + 偏置初始化为小正值（倾向开启）
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.1)
            elif any(act in name.lower() for act in ['silu', 'relu', 'gelu']):
                # 激活函数层：Kaiming初始化
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            else:
                # 普通Linear层：Xavier初始化
                torch.nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None and 'gate' not in name.lower():
                torch.nn.init.zeros_(module.bias)
                
