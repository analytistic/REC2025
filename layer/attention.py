import torch
import torch.nn.functional as F


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)
        output = attn_output

        return output, None
    
class TimeIntervalAwareSelfAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, embeddings={}):
        super(TimeIntervalAwareSelfAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        self.E_PK = embeddings["E_PK"]
        self.E_PV = embeddings["E_PV"]
        self.E_RK = embeddings["E_RK"]
        self.E_RV = embeddings["E_RV"]

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, poss_seq, interval_seq, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query) 
        K = self.k_linear(key) + self.E_PK(poss_seq)
        V = self.v_linear(value) + self.E_PV(poss_seq)

        rel_emb_k = self.E_RK(interval_seq) # [B, L, L, H]
        rel_emb_v = self.E_RV(interval_seq)

        rel_emb_k = rel_emb_k.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4) # [B, H, L, L, D]
        rel_emb_v = rel_emb_v.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算位置注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        # 计算时间间隔注意力
        q_expanded = Q.unsqueeze(3) # [B, H, L, 1, D]
        rel_attn = torch.sum(q_expanded * rel_emb_k, dim=-1)  # [B, H, L, L]

        attn_scores = (attn_scores + rel_attn) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1).masked_fill(~(attn_mask.unsqueeze(1)), 0.0)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)

        # 计算V
        attn_output = torch.matmul(attn_weights, V)
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, H, L, L, 1]
        rel_attn_output = torch.sum(attn_weights_expanded * rel_emb_v, dim=3) # [B, H, L, D]

        attn_output = (attn_output + rel_attn_output).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        attn_output = self.out_linear(attn_output)
        attn_output = torch.where(attn_output.isnan(), 0, attn_output)  # 处理NaN
        return attn_output, None
  











# class HierarchicalSequentialTransductionUnit(torch.nn.Module):
#     def __init__(self, hidden_units, cfg):
#         super(HierarchicalSequentialTransductionUnit, self).__init__()
#         self.num_heads = cfg.num_heads
#         self.qk_dim = cfg.qk_dim
#         self.v_dim = cfg.v_dim
#         self.dropout_rate = cfg.dropout_rate
#         self.hidden_units = hidden_units

#         self.f1 = torch.nn.Linear(hidden_units, 2*self.qk_dim * self.num_heads + 2 * self.v_dim * self.num_heads)

    
#     def forward(self, x, mask=None):
#         x = self.f1(x)
#         U, V, Q, K = torch.split(x, [self.v_dim * self.num_heads, self.v_dim * self.num_heads, self.qk_dim * self.num_heads, self.qk_dim * self.num_heads], dim=-1)
#         import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class HSTU(nn.Module):
#     """
#     实现文档中定义的Hierarchical Sequential Transduction Unit (HSTU)
#     包含点投影、逐点聚合注意力、点变换三层结构，融合相对注意力偏差(rab^{p,t})
#     对应文档中式(1)(2)(3)及相关架构设计
#     """
#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         d_qk: int,
#         d_v: int,
#         max_pos_dist: int = 100,
#         max_time_buckets: int = 20,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.d_model = d_model  # 嵌入维度
#         self.num_heads = num_heads  # 注意力头数
#         self.d_qk = d_qk  # 查询/键维度
#         self.d_v = d_v    # 值/门控维度
        
#         # 式(1)：点投影线性层 f1(X)，生成Q、K、V、U的合并张量
#         self.f1 = nn.Linear(
#             d_model,
#             2 * num_heads * d_qk + 2 * num_heads * d_v  # Q+K+V+U总维度
#         )
        
#         # 式(3)：点变换线性层 f2(X)
#         self.f2 = nn.Linear(num_heads * d_v, d_model)
        
#         # 层归一化（对应Norm操作）
#         self.norm = nn.LayerNorm(d_model)
        
#         # 相对注意力偏差 rab^{p,t}（融合位置和时间信息）
#         self.pos_bias = nn.Embedding(2 * max_pos_dist + 1, num_heads)  # 位置偏差
#         self.time_bias = nn.Embedding(max_time_buckets + 1, num_heads)  # 时间偏差
#         self.max_pos_dist = max_pos_dist
#         self.max_time_buckets = max_time_buckets
        
#         self.dropout = nn.Dropout(dropout)

#     def _compute_rab(self, seq_len: int, timestamps: torch.Tensor) -> torch.Tensor:
#         """计算相对注意力偏差rab^{p,t}，对应文档中融入位置(p)和时间(t)的偏差"""
#         # 1. 位置偏差 (rab^p)
#         pos_indices = torch.arange(seq_len, device=timestamps.device)
#         pos_dist = pos_indices.view(-1, 1) - pos_indices.view(1, -1)  # i-j的相对距离
#         pos_dist = torch.clamp(pos_dist, -self.max_pos_dist, self.max_pos_dist)
#         pos_dist += self.max_pos_dist  # 转为非负索引
#         pos_bias = self.pos_bias(pos_dist).permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
        
#         # 2. 时间偏差 (rab^t)
#         time_delta = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [batch, seq_len, seq_len]
#         time_delta = torch.clamp(time_delta, min=1e-6)  # 避免非正值
#         time_buckets = torch.log(time_delta).int()  # 对数分桶（适配时间非线性分布）
#         time_buckets = torch.clamp(time_buckets, 0, self.max_time_buckets)
#         time_bias = self.time_bias(time_buckets).mean(dim=0).permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
        
#         return pos_bias + time_bias  # 融合位置和时间偏差

#     def forward(
#         self,
#         x: torch.Tensor,
#         seq_lengths: torch.Tensor,
#         timestamps: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         x: 输入序列，形状 [batch_size, seq_len, d_model]
#         seq_lengths: 每个序列的实际长度，形状 [batch_size]
#         timestamps: 时间戳序列，形状 [batch_size, seq_len]
#         """
#         batch_size, seq_len, _ = x.shape
        
#         # 1. 式(1)：点投影与Split操作
#         projected = self.f1(x)  # [batch, seq_len, 2h*d_qk + 2h*d_v]
#         projected = F.silu(projected)  # ϕ1激活函数（SiLU）
        
#         # 分割为Q、K、V、U
#         qk_dim = 2 * self.num_heads * self.d_qk
#         qk, vu = projected.split([qk_dim, 2 * self.num_heads * self.d_v], dim=-1)
        
#         # 拆分多头：[batch, num_heads, seq_len, d_qk/d_v]
#         Q = qk[:, :, :self.num_heads * self.d_qk].view(batch_size, self.num_heads, seq_len, self.d_qk)
#         K = qk[:, :, self.num_heads * self.d_qk:].view(batch_size, self.num_heads, seq_len, self.d_qk)
#         V = vu[:, :, :self.num_heads * self.d_v].view(batch_size, self.num_heads, seq_len, self.d_v)
#         U = vu[:, :, self.num_heads * self.d_v:].view(batch_size, self.num_heads, seq_len, self.d_v)
        
#         # 2. 式(2)：逐点聚合注意力
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Q*K^T
#         rab = self._compute_rab(seq_len, timestamps)  # 计算rab^{p,t}
#         attn_scores += rab.unsqueeze(0)  # 融入相对偏差
        
#         attn = F.silu(attn_scores)  # ϕ2激活（替代Softmax的逐点聚合）
        
#         # 稀疏掩码：过滤无效序列位置（适配推荐数据稀疏性）
#         mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= seq_lengths.unsqueeze(1)
#         mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
#         attn = attn.masked_fill(mask, 0.0)
        
#         attn_output = torch.matmul(attn, V)  # A(X)*V(X)
        
#         # 3. 式(3)：点变换
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # 合并多头
#         norm_attn = self.norm(attn_output)  # Norm操作
        
#         U_flat = U.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # 门控权重展平
#         gated = norm_attn * U_flat  # 逐点乘门控
        
#         output = self.f2(gated)  # f2投影
#         output = self.dropout(output)
#         output = output + x  # 残差连接
        
#         return output
    

#     import torch
# import torch.nn as nn

# def generate_pos_time_embeddings(
#     batch_size: int,
#     seq_len: int,
#     dim: int,
#     timestamps: torch.Tensor,  # 形状 [batch_size, seq_len]，每个元素为时间戳（秒）
#     max_pos: int = 1000,       # 最大位置索引范围
#     max_time_buckets: int = 100  # 时间分桶数量
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     生成固定序列长度的位置嵌入和时间嵌入
    
#     返回:
#         pos_emb: 位置嵌入，形状 [batch_size, seq_len, dim]
#         time_emb: 时间嵌入，形状 [batch_size, seq_len, dim]
#     """
#     # 1. 位置嵌入（相对位置编码）
#     # 生成位置索引 [0, 1, ..., seq_len-1]
#     pos_indices = torch.arange(seq_len, device=timestamps.device)  # [seq_len]
#     # 扩展到 batch 维度
#     pos_indices = pos_indices.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
#     # 截断位置索引（避免超出预定义范围）
#     pos_indices = torch.clamp(pos_indices, 0, max_pos - 1)
    
#     # 初始化位置嵌入表并获取嵌入
#     pos_emb_table = nn.Embedding(max_pos, dim, device=timestamps.device)
#     pos_emb = pos_emb_table(pos_indices)  # [batch_size, seq_len, dim]
    
#     # 2. 时间嵌入（基于时间差分桶）
#     # 计算每个序列的最后一个时间戳作为基准时间
#     base_time = timestamps[:, -1].unsqueeze(1)  # [batch_size, 1]
#     # 计算相对时间差（基准时间 - 每个位置的时间戳）
#     time_diffs = base_time - timestamps  # [batch_size, seq_len]
#     # 确保时间差非负，转换为分钟
#     time_diffs = torch.clamp(time_diffs, min=1e-6) / 60.0  # 转为分钟
    
#     # 时间差分桶（对数变换压缩大范围时间）
#     time_buckets = torch.log(time_diffs).int()
#     # 截断分桶范围
#     time_buckets = torch.clamp(time_buckets, 0, max_time_buckets - 1)
    
#     # 初始化时间嵌入表并获取嵌入
#     time_emb_table = nn.Embedding(max_time_buckets, dim, device=timestamps.device)
#     time_emb = time_emb_table(time_buckets)  # [batch_size, seq_len, dim]
    
#     return pos_emb, time_emb