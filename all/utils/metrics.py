import torch.nn.functional as F
import torch





def evaluate_metrics(log_feats, pos_embs, neg_embs, loss_mask, k_list=[1, 3], loss_type='infonce'):
    """
    验证集metrics
    - HitRate@k, 计算正样本在前k个预测中的准确率
    """
    distance = {    
        'infonce': 'dot',
        'bce': 'dot',
        'bpr': 'dot',
        'cosine_triplet': 'cosine',
        'triplet': 'euclidean'
    }

    if distance[loss_type] == 'cosine':
        F_log = F.normalize(log_feats, p=2, dim=-1)
        F_pos = F.normalize(pos_embs, p=2, dim=-1)
        F_neg = F.normalize(neg_embs, p=2, dim=-1)

   
        pos_sim = (F_log * F_pos).sum(dim=-1).unsqueeze(1)
        neg_sim = (F_log.unsqueeze(1) * F_neg).sum(dim=-1)
    elif distance[loss_type] == 'dot':
        pos_sim = (log_feats * pos_embs).sum(dim=-1).unsqueeze(1)
        neg_sim = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)
    elif distance[loss_type] == 'euclidean':
        pos_sim = -((log_feats - pos_embs) ** 2).sum(dim=-1).unsqueeze(1)
        neg_sim = -((log_feats.unsqueeze(1) - neg_embs) ** 2).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported distance type: {distance[loss_type]}")

    val_indices = (loss_mask == 1)
    hr_dict = {}
    for k in k_list:
        topk_candidates = torch.topk(torch.cat((pos_sim, neg_sim), dim=1), k=k, dim=1).indices
        hr_k = (topk_candidates == 0).any(dim=1)[val_indices].float().mean().item()
        hr_dict[f"HitRate@{k}"] = hr_k

    return hr_dict