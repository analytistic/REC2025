import torch.nn.functional as F
import torch





def evaluate_metrics(log_feats, pos_embs, neg_embs, loss_mask, k_list=[1, 3], distance='cosine'):
    """
    验证集metrics
    - acc@k, 计算正样本在前k个预测中的准确率
    """
    assert len(log_feats.shape) == 3 and len(pos_embs.shape) == 3 and len(neg_embs.shape) == 4

    if distance == 'cosine':
        F_log = F.normalize(log_feats, p=2, dim=-1)
        F_pos = F.normalize(pos_embs, p=2, dim=-1)
        F_neg = F.normalize(neg_embs, p=2, dim=-1)
        pos_sim = (F_log * F_pos).sum(dim=-1).unsqueeze(1)
        neg_sim = (F_log.unsqueeze(1) * F_neg).sum(dim=-1)
    elif distance == 'dot':
        pos_sim = (log_feats * pos_embs).sum(dim=-1).unsqueeze(1)
        neg_sim = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)
    elif distance == 'euclidean':
        pos_sim = -((log_feats - pos_embs) ** 2).sum(dim=-1).unsqueeze(1)
        neg_sim = -((log_feats.unsqueeze(1) - neg_embs) ** 2).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported distance type: {distance}")

    val_indices = (loss_mask == 1)
    acc_dict = {}
    for k in k_list:
        topk_candidates = torch.topk(torch.cat((pos_sim, neg_sim), dim=1), k=k, dim=1).indices
        acc_k = (topk_candidates == 0).any(dim=1)[val_indices].float().mean().item()
        acc_dict[f"acc@{k}"] = acc_k

    return acc_dict