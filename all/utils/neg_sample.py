import toml
import random
import numpy as np
import torch
"""
实现各个阶段负采样策略
"""

def random_negative_sampling(cfg, l, r, seq: set, sample_pool: set):
    """
    生成不在序列s中的随机整数, 用于训练时的负采样

    """
  

    candidates = list(sample_pool - seq)

    return random.choices(candidates, k=int(cfg["neg_num"] * cfg["sample"]["random"]))

def shift_change(pos_np, pos_feat, mask, n):

    pos = np.array(pos_np)
    pos_feat = np.array(pos_feat)
    mask = np.array(mask)
    neg_np = []
    neg_feat = []
    for i in range(len(pos_np)):
        sub_mask = mask[i]
        indices = np.where(sub_mask == 1)[0]

        shifted_indices = np.roll(indices, n)
        indices = np.concatenate((np.arange(len(sub_mask)-len(indices)), indices))
        new_neg_np = pos_np[i, indices]
        new_neg_feat = pos_feat[i, indices]
        neg_np.append(new_neg_np)
        neg_feat.append(new_neg_feat)
     
    neg_np = np.expand_dims(np.array(neg_np), axis=1)
    neg_feat = np.expand_dims(np.array(neg_feat), axis=1)
    return neg_np, neg_feat

def batch_shuffling(pos_np, pos_feat, mask):
    
    neg_np = []
    neg_feat = []
    mask = np.array(mask)
    length = mask.shape[1]
    indices = np.where(mask == 1)
    coords = np.stack(indices, axis=1)
    np.random.shuffle(coords)
    start_index = 0
    for i in range(len(pos_np)):
        sub_len = int(np.sum(mask[i]==1))
        end_index = start_index + sub_len
        cols_pad = np.arange(length - sub_len)
        rows_pad = np.full_like(cols_pad, i)
        rows = np.concatenate((rows_pad, coords[start_index: end_index, 0]))
        cols = np.concatenate((cols_pad, coords[start_index: end_index, 1]))

        new_neg_np = pos_np[rows, cols]
        new_neg_feat = pos_feat[rows, cols]
        neg_np.append(new_neg_np)
        neg_feat.append(new_neg_feat)
        start_index = end_index

    neg_np = np.expand_dims(np.array(neg_np), axis=1)
    neg_feat = np.expand_dims(np.array(neg_feat), axis=1)
    return neg_np, neg_feat

 

    

def seq_shuffling(pos_np, pos_feat, mask):
    
    
    neg_np = []
    neg_feat = []
    for i in range(len(pos_np)):
        sub_mask = mask[i]
        indices = np.where(sub_mask == 1)[0]
        np.random.shuffle(indices)
        indices = np.concatenate((np.arange(len(sub_mask)-len(indices)), indices))
        new_neg_np = pos_np[i, indices]
        new_neg_feat = pos_feat[i, indices]
        neg_np.append(new_neg_np)
        neg_feat.append(new_neg_feat)
        
    neg_np = np.expand_dims(np.array(neg_np), axis=1)
    neg_feat = np.expand_dims(np.array(neg_feat), axis=1)

    
    return neg_np, neg_feat

def batch_neg_sample(pos_np, pos_feat, neg_np, neg_feat, mask, item_feat_dict):
    """
    批次内
    负采样
    
    """
    cfg = toml.load('./utils/negsample_config.toml')
    neg_feat = np.array(neg_feat)
    pos_feat = np.array(pos_feat)
    for i in range(cfg['in_batch']['seq_shuffling']):
        new_neg_np, new_neg_feat = seq_shuffling(pos_np, pos_feat, mask)
        neg_np = np.concatenate((neg_np, new_neg_np), axis=1)
        neg_feat = np.concatenate((neg_feat, new_neg_feat), axis=1)

    for i in range(cfg['in_batch']['batch_shuffling']):
        new_neg_np, new_neg_feat = batch_shuffling(pos_np, pos_feat, mask)
        neg_np = np.concatenate((neg_np, new_neg_np), axis=1)
        neg_feat = np.concatenate((neg_feat, new_neg_feat), axis=1)

    for i in range(cfg['in_batch']['shift_change'][0]):
        new_neg_np, new_neg_feat = shift_change(pos_np, pos_feat, mask, cfg['in_batch']['shift_change'][1])
        neg_np = np.concatenate((neg_np, new_neg_np), axis=1)
        neg_feat = np.concatenate((neg_feat, new_neg_feat), axis=1)



    return neg_np, neg_feat


def train_seq_shuffling(emb, mask):
    """
    训练内seq_shuffling
    """
    if len(emb.shape) == 3:
        num = emb.shape[0]
        batch_size = num
    else:
        num = emb.shape[0] * emb.shape[1]
        batch_size = emb.shape[0]
        emb = emb.view(num, emb.shape[-2], emb.shape[-1])
    neg_emb = torch.zeros((num, emb.shape[-2], emb.shape[-1]), dtype=emb.dtype, device=emb.device)
    batch_index = 0
    for i in range(num):
        sub_mask = mask[batch_index]
        indices = torch.where(sub_mask == 1)[0]
        shuffle_indices = indices[torch.randperm(len(indices))]
        indices = torch.cat((torch.arange(len(sub_mask) - len(indices)), shuffle_indices))
        neg_emb[i, :, :] = emb[i, indices, :]
        if (i + 1) % batch_size == 0: batch_size += 1


    return neg_emb

def train_batch_shuffling(emb, mask):
    assert len(emb.shape) == 4
    batch_size = emb.shape[0]
    num = emb.shape[1]
    seq_len = emb.shape[2]
    neg_emb = torch.zeros((batch_size, num, emb.shape[-2], emb.shape[-1]), dtype=emb.dtype, device=emb.device)
    mask_expand = mask.unsqueeze(1).expand(batch_size, num, mask.shape[-1])
    indices = torch.where(mask_expand == 1)
    coords = torch.stack(indices, dim=1)
    shuffle_coords = coords[torch.randperm(len(coords))]
    start_index = 0
    for i in range(batch_size):
        sub_len = int(torch.sum(mask[i] == 1))
        end_index = start_index + num * sub_len
        sub_indices = shuffle_coords[start_index:end_index, :]
        pad = emb[i, :, :(seq_len - sub_len), :].unsqueeze(0)
        shuffle_emb = emb[sub_indices[:,0], sub_indices[:,1], sub_indices[:,2], :].reshape(1, num, sub_len, -1)
        neg_emb[i, :,:,:] = torch.cat((pad, shuffle_emb), dim=2)
        start_index = end_index
    return neg_emb

    




def train_neg_sample(pos_emb, neg_emb, mask):
    """
    训练阶段批次内负采样
    """
    assert type(mask) == torch.Tensor
    cfg = toml.load('./utils/negsample_config.toml')
    from_neg_seq = cfg['in_train']['neg_seq']
    from_pos_seq = cfg['in_train']['pos_seq']
    from_pos_batch = cfg['in_train']['pos_batch']
    from_neg_batch = cfg['in_train']['neg_batch']

    neg_up = []
    neg_up.append(neg_emb)
    for i in range(from_pos_seq):
        new_neg_emb = train_seq_shuffling(pos_emb, mask)
        neg_up.append(new_neg_emb.view(neg_emb.shape[0], -1, neg_emb.shape[-2], neg_emb.shape[-1]))

    for i in range(from_neg_seq):
        new_neg_emb = train_seq_shuffling(neg_emb, mask)
        neg_up.append(new_neg_emb.view(neg_emb.shape[0], -1, neg_emb.shape[-2], neg_emb.shape[-1]))

    for i in range(from_pos_batch):
        new_neg_emb = train_batch_shuffling(pos_emb.unsqueeze(1), mask)
        neg_up.append(new_neg_emb)
    for i in range(from_neg_batch):
        new_neg_emb = train_batch_shuffling(neg_emb, mask)
        neg_up.append(new_neg_emb)
    
    neg_emb = torch.cat(neg_up, dim=1)


    return neg_emb

def train_batch_shuffling_all(pos_emb, neg_emb, mix_ratio=[0,1], sample_num=100, mask=None, hot_emb=None):
    """
    直接负采样,不管是不是padding位置
    Args:
        pos_emb: (bs, 102, 32) 正样本embedding
        neg_emb: (bs, num_neg, 102, 32) 负样本embedding
        mix_ratio: 负样本倍数
        sample_num: 最终采样数量，如果为None则使用原来的num_neg
        hot_emb: 热门物品embedding, 以后实现
    Returns:
        new_neg_emb: (bs, m, 102, 32) 重新采样的负样本
    """
    bs, seq_len, emb_dim = pos_emb.shape
    _, num_neg, _, _ = neg_emb.shape
    
    if sample_num == 0:
        return torch.empty((bs, 0, seq_len, emb_dim), dtype=neg_emb.dtype, device=neg_emb.device)
    
    if mask is None:
        pos_expanded = pos_emb.unsqueeze(1).repeat(1, int(mix_ratio[0] * 1), 1, 1)
        neg_emb = neg_emb.repeat(1, int(mix_ratio[1] * num_neg), 1, 1)
        candidates = torch.cat([pos_expanded, neg_emb], dim=1)
        num_neg = candidates.shape[1]

        batch_indices = torch.randint(0, bs, (bs, sample_num, seq_len), device=pos_emb.device)
        neg_indices = torch.randint(0, num_neg, (bs, sample_num, seq_len), device=pos_emb.device)
        seq_indices = torch.randint(0, seq_len, (bs, sample_num, seq_len), device=pos_emb.device)
        
        new_neg_emb = candidates[batch_indices, neg_indices, seq_indices, :] 

        return new_neg_emb
    else:
        pos_expanded = pos_emb.unsqueeze(1).repeat(1, int(mix_ratio[0] * 1), 1, 1)
        neg_emb = neg_emb.repeat(1, int(mix_ratio[1] * num_neg), 1, 1)
        candidates = torch.cat([pos_expanded, neg_emb], dim=1)  


        valid_indices = torch.where(mask.unsqueeze(1).expand(-1, candidates.shape[1], -1) == 1)  # (batch_idx, neg_idx, seq_idx)
        num_valid = valid_indices[0].shape[0]


        rand_idx = torch.randint(0, num_valid, (bs, sample_num, seq_len), device=mask.device)

        batch_idx = valid_indices[0][rand_idx]
        neg_idx = valid_indices[1][rand_idx]
        seq_idx = valid_indices[2][rand_idx]

        # 生成与正样本冲突位置
        pos_emb = pos_emb.unsqueeze(1).expand(-1, sample_num, -1, -1)
        new_neg_emb = candidates[batch_idx, neg_idx, seq_idx, :]  # shape: (bs, sample_num, seq_len, emb_dim)

        index_conflict = torch.where(torch.sum(torch.abs(new_neg_emb - pos_emb), dim=-1) == 0)

        if len(index_conflict[0]) > 0:
            conflict_mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=mask.device)
            conflict_mask[index_conflict[0], index_conflict[2]] = True
            final_mask = mask & (~conflict_mask)
            final_indices = torch.where(final_mask.unsqueeze(1).expand(-1, candidates.shape[1], -1) == 1)
            rand_final_idx = torch.randint(0, final_indices[0].shape[0], index_conflict[0].shape, device=mask.device)
            new_idx = final_indices[0][rand_final_idx], final_indices[1][rand_final_idx], final_indices[2][rand_final_idx]
            new_neg_emb[index_conflict[0], index_conflict[1], index_conflict[2], :] = candidates[new_idx[0], new_idx[1], new_idx[2], :]



        return new_neg_emb


