import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import toml



class RecommendationLoss:
    """
    推荐系统损失函数类，支持多种损失函数类型
    """
    
    def __init__(self, loss_type, device):
        """
        初始化损失函数
        
        Args:
            loss_type: 损失函数类型，支持 'bce', 'bpr', 'triplet', 'cosine_triplet', 'listwise_contrastive', 'focal'
            device: 计算设备
        """
        self.loss_type = loss_type
        self.device = device
        self.cfg = toml.load('./utils/loss_config.toml')

        # BCE损失函数
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
    def __call__(self, log_feats, pos_embs, neg_embs, loss_mask):
        """
        计算损失函数
        
        Args:
            log_feats: 序列特征表示 [batch_size, seq_len, hidden_dim]
            pos_embs: 正样本embedding [batch_size, seq_len, hidden_dim]
            neg_embs: 负样本embedding [batch_size, seq_len, hidden_dim]
            loss_mask: 损失计算掩码 [batch_size, seq_len]
            
        Returns:
            loss: 计算得到的损失值
        """
        if self.loss_type == 'bce':
            return self._bce_loss(log_feats, pos_embs, neg_embs, loss_mask)
        elif self.loss_type == 'bpr':
            return self._bpr_loss(log_feats, pos_embs, neg_embs, loss_mask)
        elif self.loss_type == 'triplet':
            return self._triplet_loss(log_feats, pos_embs, neg_embs, loss_mask)
        elif self.loss_type == 'cosine_triplet':
            return self._cosine_triplet_loss(log_feats, pos_embs, neg_embs, loss_mask)
        elif self.loss_type == 'listwise_contrastive':
            return self._listwise_contrastive_loss(log_feats, pos_embs, neg_embs, loss_mask)
        elif self.loss_type == 'focal':
            return self._focal_loss(log_feats, pos_embs, neg_embs, loss_mask)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _bce_loss(self, log_feats, pos_embs, neg_embs, loss_mask):
        """
        Binary Cross Entropy Loss
        
        计算正样本和负样本的BCE损失
        """
        # 计算正样本和负样本的logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [batch_size, seq_len]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # [batch_size, seq_len]
        
        # 创建标签
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        indices = np.where(loss_mask == 1)

        
        # 计算BCE损失
        pos_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        neg_loss = self.bce_criterion(neg_logits[indices], neg_labels[indices])

        return pos_loss + neg_loss
    
    def _bpr_loss(self, log_feats, pos_embs, neg_embs, loss_mask):
        """
        Bayesian Personalized Ranking Loss
        
        BPR损失：max(0, -log(σ(pos_score - neg_score)))
        """
        # 计算正样本和负样本的得分
        pos_scores = (log_feats * pos_embs).sum(dim=-1)  # [batch_size, seq_len]
        neg_scores = (log_feats * neg_embs).sum(dim=-1)  # [batch_size, seq_len]
        
        # 计算得分差
        score_diff = pos_scores - neg_scores  # [batch_size, seq_len]
        
        indices = np.where(loss_mask == 1)
        
        # BPR损失：-log(sigmoid(pos_score - neg_score))
        loss = -torch.log(torch.sigmoid(score_diff[indices]) + 1e-8).mean()

        return loss
    
    def _triplet_loss(self, log_feats, pos_embs, neg_embs, loss_mask):
        """
        Triplet Loss
        
        三元组损失：max(0, margin + neg_distance - pos_distance)
        """
        # 计算L2距离
        pos_distances = torch.norm(log_feats - pos_embs, p=2, dim=-1)  # [batch_size, seq_len]
        neg_distances = torch.norm(log_feats - neg_embs, p=2, dim=-1)  # [batch_size, seq_len]
        
        indices = np.where(loss_mask == 1)
        
        # 三元组损失
        margin = self.cfg.get('margin', 1.0)
        loss = F.relu(margin + pos_distances[indices] - neg_distances[indices]).mean()

        return loss
    
    def _cosine_triplet_loss(self, log_feats, pos_embs, neg_embs, loss_mask):
        """
        Cosine Triplet Loss
        
        基于余弦相似度的三元组损失：max(0, margin + neg_similarity - pos_similarity)
        """
        # 归一化向量
        log_feats_norm = F.normalize(log_feats, p=2, dim=-1)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)
        
        # 计算余弦相似度
        pos_similarities = (log_feats_norm * pos_embs_norm).sum(dim=-1)  # [batch_size, seq_len]
        neg_similarities = (log_feats_norm * neg_embs_norm).sum(dim=-1)  # [batch_size, seq_len]
        
        indices = np.where(loss_mask == 1)
        
        # 余弦三元组损失：max(0, margin - pos_similarity + neg_similarity)
        margin = self.cfg.get('margin', 1.0)
        loss = F.relu(margin + neg_similarities[indices] - pos_similarities[indices]).mean()
        
        return loss
    
    def _listwise_contrastive_loss(self, log_feats, pos_embs, neg_embs, loss_mask, temperature=0.1):
        """
        Listwise Contrastive Loss
        
        列表级对比损失，类似于InfoNCE损失
        """
        # 归一化特征
        log_feats_norm = F.normalize(log_feats, p=2, dim=-1)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)
        
        # 计算相似度得分
        pos_scores = (log_feats_norm * pos_embs_norm).sum(dim=-1) / temperature  # [batch_size, seq_len]
        neg_scores = (log_feats_norm * neg_embs_norm).sum(dim=-1) / temperature  # [batch_size, seq_len]
        
        # 应用掩码
        pos_scores_masked = pos_scores[loss_mask]
        neg_scores_masked = neg_scores[loss_mask]
        
        if pos_scores_masked.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 对于每个正样本，计算与所有负样本的对比损失
        # 这里简化处理，将正样本和负样本拼接起来计算softmax
        all_scores = torch.stack([pos_scores_masked, neg_scores_masked], dim=1)  # [N, 2]
        
        # 正样本的标签为0（第一个位置）
        targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=self.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(all_scores, targets)
        
        return loss
    
    def _focal_loss(self, log_feats, pos_embs, neg_embs, loss_mask, alpha=0.25, gamma=2.0):
        """
        Focal Loss
        
        用于处理样本不平衡问题的损失函数
        FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
        """
        # 计算正样本和负样本的logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [batch_size, seq_len]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # [batch_size, seq_len]
        
        # 计算概率
        pos_probs = torch.sigmoid(pos_logits)
        neg_probs = torch.sigmoid(neg_logits)
        
        # 应用掩码
        pos_probs_masked = pos_probs[loss_mask]
        neg_probs_masked = neg_probs[loss_mask]
        
        if pos_probs_masked.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算focal loss
        # 对于正样本：FL = -α * (1-p)^γ * log(p)
        pos_focal_weight = alpha * torch.pow(1 - pos_probs_masked, gamma)
        pos_loss = -pos_focal_weight * torch.log(pos_probs_masked + 1e-8)
        
        # 对于负样本：FL = -(1-α) * p^γ * log(1-p)
        neg_focal_weight = (1 - alpha) * torch.pow(neg_probs_masked, gamma)
        neg_loss = -neg_focal_weight * torch.log(1 - neg_probs_masked + 1e-8)
        
        # 平均损失
        total_loss = (pos_loss.mean() + neg_loss.mean()) / 2
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    对比损失函数（可选的额外实现）
    """
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        计算对比损失
        
        Args:
            features: 特征表示 [batch_size, feature_dim]
            labels: 标签 [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线
        mask = mask.fill_diagonal_(0)
        
        # 计算正样本和负样本的logits
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算exp值
        exp_logits = torch.exp(logits)
        
        # 计算正样本的概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数（对比学习中常用）
    """
    
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, query, positive, negatives):
        """
        计算InfoNCE损失
        
        Args:
            query: 查询向量 [batch_size, dim]
            positive: 正样本向量 [batch_size, dim]
            negatives: 负样本向量 [batch_size, num_negatives, dim]
        """
        # 归一化
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # 计算正样本得分
        pos_score = torch.sum(query * positive, dim=-1) / self.temperature  # [batch_size]
        
        # 计算负样本得分
        neg_scores = torch.sum(query.unsqueeze(1) * negatives, dim=-1) / self.temperature  # [batch_size, num_negatives]
        
        # 合并得分
        all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # [batch_size, 1 + num_negatives]
        
        # 正样本标签为0
        targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(all_scores, targets)
        
        return loss
