import torch.nn as nn
import torch.nn.functional as F
import torch

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
            "inbatch_infonce": self.inbatch_infonce_loss
        }

    def forward(self, log_feats, pos_embs, neg_embs, mask):
        loss, pos_score, neg_score, neg_var, neg_max = self.loss_map[self.loss_type](log_feats, pos_embs, neg_embs, mask, self.cfg)
        return loss, pos_score, neg_score, neg_var, neg_max


    @staticmethod
    def bce_loss(log_feats, pos_embs, neg_embs, mask, cfg):
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
        pos_l2 = torch.norm(log_feats - pos_embs, dim=-1)
        neg_l2 = torch.norm(log_feats.unsqueeze(1) - neg_embs, dim=-1)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        triplet_loss = F.relu(pos_l2[pos_indices] - neg_l2[neg_indices] + cfg.triplet.margin)
        return torch.mean(triplet_loss), torch.mean(pos_l2[pos_indices]).item(), torch.mean(neg_l2[neg_indices]).item(), torch.var(neg_l2[neg_indices]).item(), torch.max(neg_l2[neg_indices]).item()


    @staticmethod
    def cosine_triplet_loss(log_feats, pos_embs, neg_embs, mask, cfg):
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        pos_cos = F.cosine_similarity(log_feats, pos_embs, dim=-1)
        neg_cos = F.cosine_similarity(log_feats.unsqueeze(1), neg_embs, dim=-1)
        pos_indices = torch.where(mask == 1)
        neg_indices = torch.where(mask.unsqueeze(1) == 1)
        triplet_loss = F.relu(neg_cos[neg_indices] - pos_cos[pos_indices] + cfg.cosine_triplet.margin)
        return torch.mean(triplet_loss), torch.mean(pos_cos[pos_indices]).item(), torch.mean(neg_cos[neg_indices]).item(), torch.var(neg_cos[neg_indices]).item(), torch.max(neg_cos[neg_indices]).item()


    @staticmethod
    def infonce(log_feats, pos_embs, neg_embs, mask, cfg):
        
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
        
        
        all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1) 
        all_loss = torch.logsumexp(all_logits, dim=-1) - pos_logits.unsqueeze(-1)  # (batch, seq_len)

        infonce_loss = all_loss[mask == 1]

        return torch.mean(infonce_loss), torch.mean(pos_sim[mask == 1]).item(), torch.mean(neg_sim[mask.unsqueeze(1) == 1]).item(), torch.var(neg_sim[mask.unsqueeze(1) == 1]).item(), torch.max(neg_sim[mask.unsqueeze(1) == 1]).item()
    
    @staticmethod
    def inbatch_infonce_loss(log_feats, pos_embs, neg_embs, mask, cfg):
        """
        In-batch negative for InfoNCE loss without sampling.
        """
        assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4
        bs, seq_len, dim = log_feats.shape

        
        if cfg.inbatch_infonce_loss.sim == "dot":
            logits = torch.matmul(log_feats.reshape(-1, log_feats.shape[-1]), pos_embs.reshape(-1, pos_embs.shape[-1]).T) / cfg.inbatch_infonce_loss.temperature
        elif cfg.inbatch_infonce_loss.sim == "cosine":
            logits = F.cosine_similarity(log_feats.reshape(-1, log_feats.shape[-1]), pos_embs.reshape(-1, pos_embs.shape[-1]), dim=-1) / cfg.inbatch_infonce_loss.temperature
        else:
            raise ValueError("Unsupported sim type: {}".format(cfg.inbatch_infonce_loss.sim))
        
        padmask = mask.reshape(-1, 1) * mask.reshape(-1, 1).T  # (bs*seq_len, bs*seq_len)
        logits = logits.masked_fill(~padmask, float('-inf'))  # Mask out invalid positions
        diagonal_mask = torch.eye(bs * seq_len, device=logits.device, dtype=torch.bool)
        if cfg.inbatch_infonce_loss.rowmask:
            row_indices = torch.arange(bs * seq_len, device=logits.device) // seq_len
            rowmask = row_indices.unsqueeze(1) == row_indices.unsqueeze(0) 
        else:
            rowmask = torch.zeros_like(logits, dtype=torch.bool)        
        logits = logits.masked_fill(rowmask & (~diagonal_mask), float('-inf'))
        labels = torch.arange(bs * seq_len, device=logits.device)  

        loss = F.cross_entropy(logits, labels, reduction='mean')
        pos_score = logits[labels, labels].mean().item()
        neg_scores = logits[(~diagonal_mask) & (~rowmask) & padmask]
        neg_score = neg_scores.mean().item()
        neg_var = neg_scores.var().item()
        neg_max = neg_scores.max().item()
        return loss, pos_score, neg_score, neg_var, neg_max