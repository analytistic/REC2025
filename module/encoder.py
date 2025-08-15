import torch
import torch.nn.functional as F
from .gater import Gatelayer




class GLUFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, out_units, droupout_rate):
        super(GLUFeedForward, self).__init__()
        self.W_u = torch.nn.Linear(hidden_units, hidden_units)
        self.W_v = torch.nn.Linear(hidden_units, hidden_units)
        self.W_o = torch.nn.Linear(hidden_units, out_units)
        self.act_u = torch.nn.SiLU()
        self.act_v = torch.nn.SiLU()
        self.dropout1 = torch.nn.Dropout(p=droupout_rate)
        self.dropout2 = torch.nn.Dropout(p=droupout_rate)
        self.dropout = torch.nn.Dropout(p=droupout_rate)

    def forward(self, inputs):
        u = self.act_u(self.dropout1(self.W_u(inputs)))
        v = self.act_v(self.dropout2(self.W_v(inputs)))
        outputs = self.W_o(u * v)
        outputs = self.dropout(outputs)
        return outputs
        


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


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, out_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.act = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(hidden_units, out_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.act(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs
    


class MoeFFN(torch.nn.Module):
    def __init__(self, hidden_units, out_units, dropout_rate, num_experts=4):
        super(MoeFFN, self).__init__()
        self.gater = Gatelayer(hidden_units, num_experts)
        self.experts = torch.nn.ModuleList([
            GLUFeedForward(hidden_units, out_units, dropout_rate) for _ in range(num_experts)
        ])


    def forward(self, inputs):
        gates = self.gater(inputs)  # (batch_size, seq_len, num_experts)
        
        distribute_answer = torch.cat([expert(inputs).unsqueeze(-1) for expert in self.experts], -1)
        combine_answer = (distribute_answer * gates.unsqueeze(-2)).sum(dim=-1)  # (batch_size, seq_len, out_units )
        return combine_answer




class LogEncoder(torch.nn.Module):
    """
    [user, user] [item1, item1*user] [item2, item2*user] -> [item1, item2, item3]
    """
    def __init__(self, args):
        super(LogEncoder, self).__init__()
        self.args = args


        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks-1):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = MoeFFN(args.hidden_units, args.hidden_units, args.dropout_rate, num_experts=args.num_experts)
            self.forward_layers.append(new_fwd_layer)
        
        last_attn_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
        self.attention_layernorms.append(last_attn_layernorm)

        last_attn_layer = FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
        self.attention_layers.append(last_attn_layer)

        last_fwd_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
        self.forward_layernorms.append(last_fwd_layernorm)

        last_fwd_layer = MoeFFN(args.hidden_units, args.hidden_units, args.dropout_rate, num_experts=args.num_experts)
        self.forward_layers.append(last_fwd_layer)

        self.last_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
        self.output_linear = torch.nn.Linear(2 * args.hidden_units, args.hidden_units, bias=False)

    def forward(self, seqs, mask, scale):
        """
        Args:
            seqs: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
            mask: 序列的mask，形状为 [batch_size, maxlen]


        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size, maxlen, _ = seqs.shape
        seqs *= scale
        poss = torch.arange(1, maxlen + 1, device=seqs.device).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= torch.tensor(mask != 0, device=seqs.device).long() 
        # user_index = torch.clamp((mask == 1).float().argmax(dim=1)-1, min=0)
        # user_feat = seqs[torch.arange(batch_size, device=seqs.device), user_index, :].unsqueeze(1)  # [bs, 1, hidden_units]
        # cross_seqs = seqs * user_feat
        # cross_seqs *= scale
        # seqs = seqs + cross_seqs
        seqs += self.pos_emb(poss) 
        seqs = self.emb_dropout(seqs)

        ones_matrix = torch.ones((maxlen, maxlen), device=seqs.device, dtype=torch.bool)
        attention_mask_tril = torch.tril(ones_matrix)

        attention_mask_pad = (mask != 0).to(seqs.device)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.args.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        log_feats = self.last_layernorm(seqs)
        log_feats = seqs
        return log_feats


