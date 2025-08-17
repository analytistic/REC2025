import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import NegSample

class RecommendLoss(nn.Module):
    """
    Supported loss types:
    - bce_loss
    - triplet_loss
    - cosine_triplet_loss
    - infonce_loss
    Args:
        cfg: configuration object containing loss parameters
        loss_type: type of loss to compute, default is "bce"
    forward:
        log_feats: log features from the model
        pos_embs: positive embeddings
        neg_embs: negative embeddings
        mask: mask to indicate valid positions
    Returns:
        loss: computed loss value
        pos_score: average score for positive samples
        neg_score: average score for negative samples
        neg_var: variance of negative scores
        neg_max: maximum of negative scores
    """
    def __init__(self, cfg, loss_type: str="bce"):
        super(RecommendLoss, self).__init__()
        self.cfg = cfg
        self.loss_type = loss_type
        self.global_step = 0
        self.loss_map = {
            "bce" : self.bce_loss,
            "triplet": self.triplet_loss,
            "cosine_triplet": self.cosine_triplet_loss,
            "infonce": self.infonce,
            "inbatch_infonce": self.inbatch_infonce_loss,
            "ado_infonce": self.ado_infonce
        }

    def forward(self, log_feats, pos_embs, neg_embs, mask, act_0_mask=None, act_1_mask=None):
        loss, pos_score, neg_score, neg_var, neg_max = self.loss_map[self.loss_type](log_feats, pos_embs, neg_embs, mask, self.cfg, act_0_mask=act_0_mask, act_1_mask=act_1_mask)
        return loss, pos_score, neg_score, neg_var, neg_max


    @staticmethod
    def bce_loss(log_feats, pos_embs, neg_embs, mask, cfg, act_1_mask=None, act_0_mask=None):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        pos_logits = torch.sum(log_feats * pos_embs, dim=-1)
        neg_logits = torch.sum(log_feats.unsqueeze(1) * neg_embs, dim=-1)
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        loss_func = nn.BCEWithLogitsLoss(reduction=cfg.bce.reduction)
        pos_loss = loss_func(pos_logits[pos_indices], pos_labels[pos_indices])
        neg_loss = loss_func(neg_logits[neg_indices], neg_labels[neg_indices])
        return cfg.bce.alpha * pos_loss + cfg.bce.beta * neg_loss, torch.mean(pos_logits[pos_indices]).item(), torch.mean(neg_logits[neg_indices]).item(), torch.var(neg_logits[neg_indices]).item(), torch.max(neg_logits[neg_indices]).item()

    @staticmethod
    def triplet_loss(log_feats, pos_embs, neg_embs, mask, cfg):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4

        if cfg.triplet.neg_embs_sample:
            neg_embs = NegSample.batch_neg_sample(cfg.triplet.use_neg_embs, pos_embs, neg_embs, mask, sample_num=cfg.triplet.use_neg_embs.sample_num)
        
        
        pos_l2 = torch.norm(log_feats - pos_embs, dim=-1)
        neg_l2 = torch.norm(log_feats.unsqueeze(1) - neg_embs, dim=-1)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        triplet_loss = F.relu(pos_l2[pos_indices] - neg_l2[neg_indices] + cfg.triplet.margin)
        return torch.mean(triplet_loss), torch.mean(pos_l2[pos_indices]).item(), torch.mean(neg_l2[neg_indices]).item(), torch.var(neg_l2[neg_indices]).item(), torch.max(neg_l2[neg_indices]).item()


    @staticmethod
    def cosine_triplet_loss(log_feats, pos_embs, neg_embs, mask, cfg, act_1_mask=None, act_0_mask=None):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        if cfg.cosine_triplet.neg_embs_sample:
            neg_embs = NegSample.batch_neg_sample(cfg.cosine_triplet.use_neg_embs, pos_embs, neg_embs, mask, sample_num=cfg.cosine_triplet.use_neg_embs.sample_num)
            
        pos_cos = F.cosine_similarity(log_feats, pos_embs, dim=-1)
        neg_cos = F.cosine_similarity(log_feats.unsqueeze(1), neg_embs, dim=-1)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        triplet_loss = F.relu(neg_cos[neg_indices] - pos_cos[pos_indices] + cfg.cosine_triplet.margin)
        return torch.mean(triplet_loss), torch.mean(pos_cos[pos_indices]).item(), torch.mean(neg_cos[neg_indices]).item(), torch.var(neg_cos[neg_indices]).item(), torch.max(neg_cos[neg_indices]).item()


    @staticmethod
    def infonce(log_feats, pos_embs, neg_embs, mask, cfg, act_1_mask=None, act_0_mask=None):
        # 需要对负样本扩采样
        neg_embs = torch.cat((NegSample.batch_neg_sample(cfg.in_train, pos_embs, neg_embs, mask, sample_num=cfg.in_train.sample_num), neg_embs), dim=1)

        
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        if cfg.infonce.sim == "dot":
            pos_sim = torch.sum(log_feats * pos_embs, dim=-1)
            neg_sim = torch.sum(log_feats.unsqueeze(1) * neg_embs, dim=-1)
        elif cfg.infonce.sim == "cosine":
            pos_sim = F.cosine_similarity(log_feats, pos_embs, dim=-1)
            neg_sim = F.cosine_similarity(log_feats.unsqueeze(1), neg_embs, dim=-1)
        else:
            raise ValueError("Unsupported sim type: {}".format(cfg.infonce.sim))
        

        pos_logits = pos_sim / cfg.infonce.temperature  
        neg_logits = neg_sim / cfg.infonce.temperature  
        
        
        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1) 
        all_loss = torch.logsumexp(all_logits, dim=1) - pos_logits  # (batch, seq_len)

        if cfg.weight_loss.act == True:
            all_loss[act_0_mask==1] *= cfg.weight_loss.weight[0]
            all_loss[act_1_mask==1] *= cfg.weight_loss.weight[1]

        infonce_loss = all_loss[mask == 1]
        pos_indices = torch.where(mask == 1)
        neg_mask = mask.unsqueeze(1).expand_as(neg_sim)  # [batch, seq_len, neg_num]
        neg_indices = torch.where(neg_mask == 1)

        



        return torch.mean(infonce_loss), torch.mean(pos_sim[pos_indices]).item(), torch.mean(neg_sim[neg_indices]).item(), torch.var(neg_sim[neg_indices]).item(), torch.max(neg_sim[neg_indices]).item()

    @staticmethod
    def inbatch_infonce_loss(log_feats, pos_embs, neg_embs, mask, cfg):
        """
        In-batch negative for InfoNCE loss without sampling.
        """
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        bs, seq_len, dim = log_feats.shape

        
        if cfg.inbatch_infonce_loss.sim == "dot":
            logits = torch.matmul(log_feats.reshape(-1, log_feats.shape[-1]), pos_embs.reshape(-1, pos_embs.shape[-1]).T) / cfg.inbatch_infonce_loss.temperature
            if cfg.inbatch_infonce_loss.neg_embs_sample:
                neg_sample = NegSample.batch_neg_sample(cfg.inbatch_infonce_loss.use_neg_embs, pos_embs, neg_embs, mask, sample_num=cfg.inbatch_infonce_loss.use_neg_embs.sample_num) # bs, n, seq_len, dim
                logits_to_neg = torch.sum(log_feats.unsqueeze(1) * neg_sample, dim=-1) / cfg.inbatch_infonce_loss.temperature # bs, n, seq_len
                logits = torch.cat([logits, logits_to_neg.permute(0, 2, 1).reshape(-1, cfg.inbatch_infonce_loss.use_neg_embs.sample_num)], dim=-1) # bs*seq_len, bs*seq_len+n
        elif cfg.inbatch_infonce_loss.sim == "cosine":
            q = F.normalize(log_feats.reshape(-1, log_feats.shape[-1]), dim=-1)  # [N, D]
            k = F.normalize(pos_embs.reshape(-1, pos_embs.shape[-1]), dim=-1)    # [N, D]
            logits = torch.matmul(q, k.T) / cfg.inbatch_infonce_loss.temperature  # [N, N]
            if cfg.inbatch_infonce_loss.neg_embs_sample:
                neg_sample = NegSample.batch_neg_sample(cfg.inbatch_infonce_loss.use_neg_embs, pos_embs, neg_embs, mask, sample_num=cfg.inbatch_infonce_loss.use_neg_embs.sample_num)
                neg_sample = F.normalize(neg_sample, p=2, dim=-1)
                log_feats = F.normalize(log_feats, p=2, dim=-1)
                logits_to_neg = torch.sum(log_feats * neg_sample, dim=-1) / cfg.inbatch_infonce_loss.temperature  # bs, n, seq_len
                logits = torch.cat([logits, logits_to_neg.permute(0, 2, 1).reshape(-1, cfg.inbatch_infonce_loss.use_neg_embs.sample_num)], dim=-1)
        else:
            raise ValueError("Unsupported sim type: {}".format(cfg.inbatch_infonce_loss.sim))
        
        # 根据是否有额外负样本调整mask维度
        if cfg.inbatch_infonce_loss.neg_embs_sample:
            # logits: (bs*seq_len, bs*seq_len+n)
            extra_neg_size = cfg.inbatch_infonce_loss.use_neg_embs.sample_num
            padmask = mask.reshape(-1, 1) * mask.reshape(-1, 1).T  # (bs*seq_len, bs*seq_len)
            # 为额外的负样本位置扩展mask
            extra_mask = mask.reshape(-1, 1).expand(-1, extra_neg_size)  # (bs*seq_len, n)
            padmask = torch.cat([padmask, extra_mask], dim=1)  # (bs*seq_len, bs*seq_len+n)
        else:
            # logits: (bs*seq_len, bs*seq_len)
            padmask = mask.reshape(-1, 1) * mask.reshape(-1, 1).T  # (bs*seq_len, bs*seq_len)
            
        logits = logits.masked_fill(~padmask, float('-inf'))  # Mask out invalid positions
        diagonal_mask = torch.eye(bs * seq_len, device=logits.device, dtype=torch.bool)
        
        if cfg.inbatch_infonce_loss.rowmask:
            row_indices = torch.arange(bs * seq_len, device=logits.device) // seq_len
            rowmask = row_indices.unsqueeze(1) == row_indices.unsqueeze(0)  # (bs*seq_len, bs*seq_len)
            if cfg.inbatch_infonce_loss.neg_embs_sample:
                # 为额外负样本扩展rowmask，额外负样本不受rowmask限制
                extra_rowmask = torch.zeros(bs * seq_len, cfg.inbatch_infonce_loss.use_neg_embs.sample_num, 
                                          device=logits.device, dtype=torch.bool)
                rowmask = torch.cat([rowmask, extra_rowmask], dim=1)
        else:
            print("Warning: rowmask is set to False, this may lead to incorrect loss calculation.")
            rowmask = torch.zeros_like(logits, dtype=torch.bool)
            
        # 只对in-batch部分应用diagonal_mask，额外负样本不受影响
        if cfg.inbatch_infonce_loss.neg_embs_sample:
            diagonal_mask_extended = torch.zeros_like(logits, dtype=torch.bool)
            diagonal_mask_extended[:, :bs*seq_len] = diagonal_mask
            logits = logits.masked_fill(rowmask & (~diagonal_mask_extended), float('-inf'))
        else:
            logits = logits.masked_fill(rowmask & (~diagonal_mask), float('-inf'))
        labels = torch.arange(bs * seq_len, device=logits.device)  

        logits = logits[mask.reshape(-1),:]
        labels = labels[mask.reshape(-1)]

        loss = F.cross_entropy(logits, labels, reduction='mean')
        pos_score = logits[torch.arange(logits.size(0)), labels].mean().item()

        # 计算负样本得分时需要考虑维度变化
        if cfg.inbatch_infonce_loss.neg_embs_sample:
            diagonal_mask_extended = torch.zeros_like(logits, dtype=torch.bool)
            diagonal_mask_extended[:, :bs*seq_len] = diagonal_mask[mask.reshape(-1), :]
            rowmask_valid = rowmask[mask.reshape(-1), :]
            padmask_valid = padmask[mask.reshape(-1), :]
            neg_scores = logits[(~diagonal_mask_extended) & (~rowmask_valid) & padmask_valid]
        else:
            diagonal_mask_valid = diagonal_mask[mask.reshape(-1), :]
            rowmask_valid = rowmask[mask.reshape(-1), :]
            padmask_valid = padmask[mask.reshape(-1), :]
            neg_scores = logits[(~diagonal_mask_valid) & (~rowmask_valid) & padmask_valid]
            
        neg_score = neg_scores.mean().item()
        neg_var = neg_scores.var().item()
        neg_max = neg_scores.max().item()
        return loss, pos_score, neg_score, neg_var, neg_max
    

    @staticmethod
    def ado_infonce(log_feats, pos_embs, neg_embs, mask, cfg, act_1_mask=None, act_0_mask=None):
        """
        ADO InfoNCE loss with in-batch negatives.
        """
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4

        bs, seq_len, dim = log_feats.shape

        neg_embs = neg_embs.squeeze(1)  # (bs, seq_len, dim)
        log_feats = F.normalize(log_feats, p=2, dim=-1)  # Normalize sequence embeddings
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)

        pos_logits = F.cosine_similarity(log_feats, pos_embs, dim=-1)  # (bs, seq_len)
        pos_scores = pos_logits[mask == 1].mean().item()  # Average positive score

        neg_embs = F.normalize(neg_embs, p=2, dim=-1)  # Normalize negative embeddings
        neg_embs_all = neg_embs.reshape(-1, dim)
        neg_logitys = torch.matmul(log_feats, neg_embs_all.transpose(-1, -2))
        neg_scores = neg_logitys.mean().item()
        logtis = torch.cat([pos_logits.unsqueeze(-1), neg_logitys], dim=-1)  # bs, seq_len, neg_num+1
        

        if cfg.weight_loss.act == True:
            logtis_0 = logtis[(mask == 1) & (act_0_mask == 1)] / cfg.ado_infonce.temperature  
            logtis_1 = logtis[(mask == 1) & (act_1_mask == 1)] / cfg.ado_infonce.temperature
            labels_0 = torch.zeros(logtis_0.shape[0], dtype=torch.long, device=logtis.device)
            labels_1 = torch.zeros(logtis_1.shape[0], dtype=torch.long, device=logtis.device)
            loss_0 = F.cross_entropy(logtis_0, labels_0)
            loss_1 = F.cross_entropy(logtis_1, labels_1)
            loss = loss_0 * cfg.weight_loss.weight[0] + loss_1 * cfg.weight_loss.weight[1]
        else:
            logtis = logtis[mask == 1] / cfg.ado_infonce.temperature
            labels = torch.zeros(logtis.shape[0], dtype=torch.long, device=logtis.device)
            loss = F.cross_entropy(logtis, labels)

        return loss, pos_scores, neg_scores, neg_logitys.var().item(), neg_logitys.max().item()