#!/usr/bin/env python3
"""
混合精度训练示例脚本
演示如何在推荐系统中使用混合精度训练
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel
from config import BaseConfig
from utils import evaluate_metrics, log_gradient_stats, clip_gradients, get_optim, get_cosine_schedule_with_warmup


def setup_mixed_precision(config):
    """
    设置混合精度训练
    
    Args:
        config: 训练配置
        
    Returns:
        GradScaler对象或None
    """
    if not config.get('use_amp', False) or not torch.cuda.is_available():
        print("Mixed precision training disabled")
        return None
    
    # 检查GPU兼容性
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 7:
        print(f"Warning: GPU compute capability {device_capability[0]}.{device_capability[1]} < 7.0")
        print("Mixed precision may not provide significant speedup")
    
    scaler = GradScaler(
        init_scale=config.get('amp_init_scale', 65536.0),
        growth_factor=config.get('amp_growth_factor', 2.0),
        backoff_factor=config.get('amp_backoff_factor', 0.5),
        growth_interval=config.get('amp_growth_interval', 2000)
    )
    
    print(f"Mixed precision training enabled with initial scale: {scaler.get_scale()}")
    return scaler


def train_step_amp(model, batch, optimizer, scaler, args, device):
    """
    使用混合精度的训练步骤
    
    Args:
        model: 模型
        batch: 批次数据
        optimizer: 优化器
        scaler: 梯度缩放器
        args: 训练参数
        device: 设备
        
    Returns:
        损失值和其他指标
    """
    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type = batch
    seq = seq.to(device)
    pos = pos.to(device)
    neg = neg.to(device)
    
    optimizer.zero_grad()
    
    if scaler is not None:
        # 混合精度训练
        with autocast():
            loss_list, pos_score, neg_score, neg_var, neg_max = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
            )
            loss = torch.stack(loss_list).sum()
            
            # L2正则化
            if args.l2_emb > 0.0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        
        # 取消缩放以进行梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        scaler.step(optimizer)
        scaler.update()
        
    else:
        # 标准精度训练
        loss_list, pos_score, neg_score, neg_var, neg_max = model(
            seq, pos, neg, token_type, next_token_type, next_action_type, 
            seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
        )
        loss = torch.stack(loss_list).sum()
        
        # L2正则化
        if args.l2_emb > 0.0:
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return loss, loss_list, pos_score, neg_score, neg_var, neg_max


def validate_step_amp(model, batch, scaler, device):
    """
    使用混合精度的验证步骤
    
    Args:
        model: 模型
        batch: 批次数据
        scaler: 梯度缩放器
        device: 设备
        
    Returns:
        损失值和预测结果
    """
    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type = batch
    seq = seq.to(device)
    pos = pos.to(device)
    neg = neg.to(device)
    
    if scaler is not None:
        # 验证时也使用混合精度以保持一致性
        with autocast():
            main_loss, bce_loss, log_feats, pos_embs, candidate_item, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
            )
    else:
        main_loss, bce_loss, log_feats, pos_embs, candidate_item, loss_mask = model(
            seq, pos, neg, token_type, next_token_type, next_action_type, 
            seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
        )
    
    return main_loss, bce_loss, log_feats, pos_embs, candidate_item, loss_mask


def optimize_model_for_amp(model):
    """
    优化模型以支持混合精度训练
    
    Args:
        model: 要优化的模型
    """
    # 初始化权重以避免梯度溢出
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.normal_(param, 0, 0.02)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    
    # 特殊处理嵌入层的padding_idx
    if hasattr(model, 'item_emb') and hasattr(model.item_emb, 'padding_idx'):
        model.item_emb.weight.data[0, :] = 0
    if hasattr(model, 'user_emb') and hasattr(model.user_emb, 'padding_idx'):
        model.user_emb.weight.data[0, :] = 0
    
    # 处理LogEncoder中的嵌入层
    if hasattr(model, 'logencoder'):
        if hasattr(model.logencoder, 'pos_emb'):
            model.logencoder.pos_emb.weight.data[0, :] = 0
        if hasattr(model.logencoder, 'act_emb'):
            model.logencoder.act_emb.weight.data[0, :] = 0
        if hasattr(model.logencoder, 'time_stamp_emb'):
            for key in model.logencoder.time_stamp_emb:
                model.logencoder.time_stamp_emb[key].weight.data[0, :] = 0


def profile_memory_usage():
    """分析显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    return 0, 0


def main():
    """主训练函数"""
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--amp_init_scale', default=65536.0, type=float, help='Initial scale for GradScaler')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    
    print(f"Training configuration:")
    print(f"  Mixed precision: {args.use_amp}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    # 设置混合精度训练
    config = vars(args)
    scaler = setup_mixed_precision(config)
    
    # 初始化模型和数据
    # 这里需要根据实际情况初始化模型和数据集
    # model = BaselineModel(...)
    # dataset = MyDataset(...)
    
    print("Mixed precision training setup complete!")
    print("Memory usage before training:")
    profile_memory_usage()


if __name__ == '__main__':
    main()
