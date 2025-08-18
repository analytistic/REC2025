import torch
import torch.nn.functional as F
from .gater import Gatelayer
from .emb_fusion import EmbeddingFusionGate
from layer.attention import FlashMultiHeadAttention
from layer.ffn import GLUFeedForward


class CrossFeatFusion(torch.nn.Module):
    def __init__(self, cat_dim, hidden_units):
        super(CrossFeatFusion, self).__init__()
        self.gate = EmbeddingFusionGate(cat_emb_dim=cat_dim, fusion_dim=hidden_units)

    def forward(self, user_emb, item_emb):
        user_emb = user_emb.expand(-1, item_emb.shape[1], -1) 
        fusion_emb = self.gate(user_emb, item_emb)
        return fusion_emb


    


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




class TransformerEncoder(torch.nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args

        self.attention_layernorms = torch.nn.ModuleList()  
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks-1):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
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
    
    def forward(self, q, k, v, mask=None):
        ones_matrix = torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool)
        attention_mask_tril = torch.tril(ones_matrix)

        attention_mask_pad = (mask != 0).to(q.device)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.args.norm_first:
                x = self.attention_layernorms[i](q)
                mha_outputs, _ = self.attention_layers[i](q, k, v, attn_mask=attention_mask)
                q = q + mha_outputs
                q = q + self.forward_layers[i](self.forward_layernorms[i](q))
            else:
                mha_outputs, _ = self.attention_layers[i](q, k, v, attn_mask=attention_mask)
                q = self.attention_layernorms[i](q + mha_outputs)
                q = self.forward_layernorms[i](q + self.forward_layers[i](q))
        log_feats = self.last_layernorm(q)
        return log_feats

class LogEncoder(torch.nn.Module):
    """
   
    """
    def __init__(self, args, fusion_module):
        super(LogEncoder, self).__init__()
        self.args = args

        self.act_emb = torch.nn.Embedding(2, args.hidden_units, padding_idx=0)
        self.time_stamp_emb = torch.nn.ModuleDict(
            {
                "hour": torch.nn.Embedding(25, args.hidden_units, padding_idx=0),
                "day": torch.nn.Embedding(32, args.hidden_units, padding_idx=0),
                "month": torch.nn.Embedding(13, args.hidden_units, padding_idx=0),
                "minute": torch.nn.Embedding(61, args.hidden_units, padding_idx=0),
            }
        )


        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # self.cross_fusion1 = CrossFeatFusion(cat_dim=2*args.hidden_units, hidden_units=args.hidden_units)
        # self.cross_fusion2 = CrossFeatFusion(cat_dim=2*args.hidden_units, hidden_units=args.hidden_units)
        self.id_encoder = TransformerEncoder(args.id_encoder)
        self.feat_encoder = TransformerEncoder(args.feat_encoder)
        self.fusion_module = fusion_module


    def forward(self, id_seqs, feat_seqs, mask, seq_time, seq_action_type, scale):
        """
        Args:
            id_seqs: 序列ID
            feat_seqs: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
            seq_time: 序列时间特征
            seq_action_type: 序列动作类型
            scale: 缩放因子，用于缩放输入的ID和特征序列


        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size, maxlen, _ = id_seqs.shape


        id_seqs *= scale
        feat_seqs *= scale
        poss = torch.arange(1, maxlen + 1, device=id_seqs.device).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= (mask != 0).long().to(id_seqs.device)


        # hour = (seq_time // 3600) % 24 + 1
        # day = (seq_time // 86400) % 31 + 1
        # month = (seq_time // (86400 * 30)) % 12 + 1
        # minute = (seq_time // 60) % 60

        pad_mask = (seq_time == 0)
        hour = ((seq_time // 3600) % 24 + 1).masked_fill(pad_mask, 0).to(id_seqs.device)
        day = ((seq_time // 86400) % 31 + 1).masked_fill(pad_mask, 0).to(id_seqs.device)
        month = ((seq_time // (86400 * 30)) % 12 + 1).masked_fill(pad_mask, 0).to(id_seqs.device)
        minute = ((seq_time // 60) % 60 + 1).masked_fill(pad_mask, 0).to(id_seqs.device)

        hour_emb = self.time_stamp_emb["hour"](hour)
        day_emb = self.time_stamp_emb["day"](day)
        month_emb = self.time_stamp_emb["month"](month)
        minute_emb = self.time_stamp_emb["minute"](minute)

        act_emb = self.act_emb(seq_action_type.to(id_seqs.device))

        poss = self.pos_emb(poss)
        poss = hour_emb + day_emb + month_emb + minute_emb  + poss + act_emb

  
        # user_index = torch.clamp((mask == 1).float().argmax(dim=1)-1, min=0)
        # user_feat = feat_seqs[torch.arange(batch_size, device=id_seqs.device), user_index, :].unsqueeze(1)  # [bs, 1, hidden_units]
        # user_id = id_seqs[torch.arange(batch_size, device=id_seqs.device), user_index, :].unsqueeze(1)  # [bs, 1, hidden_units]
        # user_feat = self.emb_dropout(user_feat)
        # feat_seqs = self.emb_dropout(feat_seqs)


        # id_seqs = self.cross_fusion1(user_id, id_seqs) + poss
        # feat_seqs = self.cross_fusion2(user_feat, feat_seqs) + poss
        feat_seqs = self.emb_dropout(feat_seqs+poss) 
        id_seqs = self.emb_dropout(id_seqs+poss) 

        id_log = self.id_encoder(id_seqs, id_seqs, id_seqs, mask)
        feat_log = self.feat_encoder(feat_seqs, feat_seqs, feat_seqs, mask)
        log = self.fusion_module(id_log, feat_log)




        return log


