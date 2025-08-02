import numpy as np
import torch
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader

def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """
    计算 DCG@K (Discounted Cumulative Gain)
    
    Args:
        scores: 相关性分数数组 (1表示相关，0表示不相关)
        k: top-k
    
    Returns:
        DCG@K 值
    """
    scores = scores[:k]
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        y_true: 真实标签 (1表示相关，0表示不相关)
        y_score: 预测分数
        k: top-k
    
    Returns:
        NDCG@K 值
    """
    # 按预测分数降序排序
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    
    # 计算DCG@K
    dcg = dcg_at_k(y_true_sorted, k)
    
    # 计算IDCG@K (理想情况下的DCG)
    ideal_order = np.argsort(y_true)[::-1]
    y_true_ideal = y_true[ideal_order]
    idcg = dcg_at_k(y_true_ideal, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    计算 HR@K (Hit Rate)
    
    Args:
        y_true: 真实标签 (1表示相关，0表示不相关)
        y_score: 预测分数
        k: top-k
    
    Returns:
        HR@K 值 (0或1)
    """
    # 按预测分数降序排序，取top-k
    order = np.argsort(y_score)[::-1][:k]
    y_true_topk = y_true[order]
    
    # 如果top-k中有任何相关物品，则命中
    return 1.0 if np.sum(y_true_topk) > 0 else 0.0


def evaluate_model_on_validation(model, valid_loader, device, args, 
                                 negative_sampling_size: int = 100) -> Tuple[float, float]:
    """
    在验证集上评估模型的 NDCG@10 和 HR@10
    
    Args:
        model: 训练好的推荐模型
        valid_loader: 验证集数据加载器
        device: 设备 (cuda/cpu)
        args: 参数配置
        negative_sampling_size: 负采样大小
    
    Returns:
        (ndcg10, hr10): NDCG@10 和 HR@10 的平均值
    """
    model.eval()
    
    ndcg10_scores = []
    hr10_scores = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating"):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 移至设备
            seq = seq.to(device)
            
            batch_size = seq.shape[0]
            
            for i in range(batch_size):
                # 获取单个用户的序列和特征
                user_seq = seq[i:i+1]  # [1, seq_len]
                user_seq_feat = [seq_feat[i]]
                user_token_type = token_type[i:i+1]
                
                # 获取正样本和负样本
                pos_item = pos[i].item()
                pos_feat_dict = pos_feat[i]
                
                # 构建候选集：正样本 + 负样本
                candidates = [pos_item]
                candidate_feats = [pos_feat_dict]
                
                # 添加负样本
                neg_items = neg[i][:negative_sampling_size]  # 限制负样本数量
                for j, neg_item in enumerate(neg_items):
                    if j < len(neg_feat[i]):
                        candidates.append(neg_item.item())
                        candidate_feats.append(neg_feat[i][j])
                
                # 构建标签：正样本为1，负样本为0
                labels = np.zeros(len(candidates))
                labels[0] = 1  # 第一个是正样本
                
                # 计算候选物品的得分
                scores = []
                for cand_item, cand_feat in zip(candidates, candidate_feats):
                    # 为每个候选物品计算得分
                    cand_item_tensor = torch.tensor([cand_item], device=device).unsqueeze(0)  # [1, 1]
                    cand_feat_list = [cand_feat]
                    
                    # 使用模型预测
                    logits = model.predict(user_seq, user_seq_feat, user_token_type,
                                         target_items=cand_item_tensor, 
                                         target_feats=cand_feat_list)
                    scores.append(logits.cpu().numpy()[0])
                
                scores = np.array(scores)
                
                # 计算指标
                ndcg10 = ndcg_at_k(labels, scores, k=10)
                hr10 = hit_rate_at_k(labels, scores, k=10)
                
                ndcg10_scores.append(ndcg10)
                hr10_scores.append(hr10)
    
    # 计算平均值
    avg_ndcg10 = float(np.mean(ndcg10_scores))
    avg_hr10 = float(np.mean(hr10_scores))
    
    return avg_ndcg10, avg_hr10


def evaluate_with_infer_pipeline(model, test_dataset, args, k: int = 10) -> Tuple[float, float]:
    """
    使用类似 infer.py 的流程进行评估
    
    Args:
        model: 训练好的推荐模型
        test_dataset: 测试数据集
        args: 参数配置
        k: top-k，默认为10
    
    Returns:
        (ndcg_k, hr_k): NDCG@K 和 HR@K 的平均值
    """
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, collate_fn=test_dataset.collate_fn
    )
    
    model.eval()
    all_user_embeddings = []
    user_ids = []
    
    # 1. 获取所有用户的embedding
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Generating user embeddings"):
            seq, token_type, seq_feat, user_id = batch
            seq = seq.to(args.device)
            
            # 生成用户表示
            logits = model.predict(seq, seq_feat, token_type)
            
            for i in range(logits.shape[0]):
                emb = logits[i].detach().cpu().numpy()
                all_user_embeddings.append(emb)
            
            user_ids.extend(user_id)
    
    # 2. 获取候选item embedding (这里简化处理，实际应该从候选库获取)
    # 注：这是一个简化版本，实际使用时需要根据具体的候选库来实现
    print("Warning: This is a simplified evaluation. For complete evaluation, please implement candidate item embedding generation.")
    
    # 暂时返回示例值
    return 0.0, 0.0


def simple_validation_evaluation(model, valid_loader, device, k: int = 10) -> Tuple[float, float]:
    """
    简化版本的验证集评估
    
    Args:
        model: 训练好的推荐模型
        valid_loader: 验证集数据加载器
        device: 设备
        k: top-k，默认为10
    
    Returns:
        (ndcg_k, hr_k): NDCG@K 和 HR@K 的平均值
    """
    model.eval()
    
    ndcg_scores = []
    hr_scores = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Evaluating NDCG@{k} and HR@{k}"):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 移至设备
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            # 获取正样本和负样本的预测分数
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat
            )
            
            batch_size = pos_logits.shape[0]
            
            for i in range(batch_size):
                # 只考虑有效的预测（next_token_type == 1）
                if next_token_type[i] == 1:
                    # 构建候选分数：正样本分数 + 负样本分数
                    pos_score = pos_logits[i].cpu().numpy()
                    neg_scores = neg_logits[i].cpu().numpy()
                    
                    # 合并分数
                    all_scores = np.concatenate([[pos_score], neg_scores])
                    # 标签：正样本为1，负样本为0
                    labels = np.concatenate([[1], np.zeros(len(neg_scores))])
                    
                    # 计算指标
                    ndcg_k = ndcg_at_k(labels, all_scores, k)
                    hr_k = hit_rate_at_k(labels, all_scores, k)
                    
                    ndcg_scores.append(ndcg_k)
                    hr_scores.append(hr_k)
    
    # 计算平均值
    avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    avg_hr = float(np.mean(hr_scores)) if hr_scores else 0.0
    
    return avg_ndcg, avg_hr


def evaluate_ndcg10_hr10(model, valid_loader, device) -> Tuple[float, float]:
    """
    在验证集上计算 NDCG@10 和 HR@10
    
    Args:
        model: 训练好的推荐模型
        valid_loader: 验证集数据加载器  
        device: 设备
    
    Returns:
        (ndcg10, hr10): NDCG@10 和 HR@10 的平均值
    """
    return simple_validation_evaluation(model, valid_loader, device, k=10)


# 使用示例：
"""
# 在训练循环中使用评估函数
from utils.metrics import evaluate_ndcg10_hr10

# 在每个epoch结束后评估
model.eval()
ndcg10, hr10 = evaluate_ndcg10_hr10(model, valid_loader, args.device)
print(f"Validation NDCG@10: {ndcg10:.4f}, HR@10: {hr10:.4f}")

# 可以用于早停或模型选择
if ndcg10 > best_ndcg10:
    best_ndcg10 = ndcg10
    # 保存最佳模型
    torch.save(model.state_dict(), 'best_model.pt')
"""
