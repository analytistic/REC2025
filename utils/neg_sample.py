import torch


class NegSample(object):
    """
    Negative Sampling for Training and Evaluation
    Args:
        cfg: Configuration object containing parameters for negative sampling
    Methods:
        train_neg_sample: Perform negative sampling during training
        eval_neg_sample: Perform negative sampling during evaluation
    我去, 这玩意太费脑子了简直
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = 0

    def train_neg_sample(self, pos_emb, neg_emb, mask):
        cfg = self.cfg.in_train
        sample = cfg.sample_num
        return self.batch_neg_sample(cfg, pos_emb, neg_emb, mask, sample_num=sample)

    def eval_neg_sample(self, pos_emb, neg_emb, mask):
        cfg = self.cfg.in_eval
        sample = cfg.sample_num
        return self.batch_neg_sample(cfg, pos_emb, neg_emb, mask, sample_num=sample)

    @staticmethod
    def batch_neg_sample(cfg, pos_emb, neg_emb, mask, sample_num=100, hot_emb=None):
        assert len(neg_emb.shape) == 4 and len(pos_emb.shape) == 3
        bs, _, seq_len, dim = neg_emb.shape

        from_pos = int(cfg.from_pos * sample_num)
        from_neg = int(cfg.from_neg * sample_num)

        if from_pos + from_neg == 0:
            return neg_emb
        
        neg_emb_neg = torch.zeros((bs, from_neg, seq_len, dim), device=neg_emb.device) 
        neg_emb_pos = torch.zeros((bs, from_pos, seq_len, dim), device=pos_emb.device)

        
        if from_neg > 0:
            valid_indices = torch.where(mask.unsqueeze(1).expand(-1, neg_emb.shape[1], -1) == 1) # bs, _, seq_len
            num_valid = valid_indices[0].shape[0]

            rand_idx = torch.randint(0, num_valid, (bs, from_neg, seq_len), device=neg_emb.device)
            neg_emb_neg = neg_emb[valid_indices[0][rand_idx], valid_indices[1][rand_idx], valid_indices[2][rand_idx], :]

            index_conflict = torch.where(torch.sum(torch.abs(neg_emb_neg - pos_emb.unsqueeze(1)), dim=-1) == 0)
            if len(index_conflict[0]) > 0:
                print(f"Warning: found {len(index_conflict[0])} conflicts in negative sampling, resample")
                conflict_mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=neg_emb.device)
                conflict_mask[index_conflict[0], index_conflict[2]] = True
                conflict_mask = mask & (~conflict_mask)
                valid_indices = torch.where(conflict_mask.unsqueeze(1).expand(-1, neg_emb.shape[1], -1) == 1)
                rand_idx = torch.randint(0, valid_indices[0].shape[0], index_conflict[0].shape, device=neg_emb.device)
                neg_emb_neg[index_conflict[0], index_conflict[1], index_conflict[2]] = neg_emb[valid_indices[0][rand_idx], valid_indices[1][rand_idx], valid_indices[2][rand_idx], :]
            index_conflict = torch.where(torch.sum(torch.abs(neg_emb_neg - pos_emb.unsqueeze(1)), dim=-1) == 0)
            if len(index_conflict[0]) > 0:
                print(f"Warning: Resample found {len(index_conflict[0])} conflicts in negative sampling")

        if from_pos > 0:
            flip_pos_emb = torch.flip(pos_emb, dims=[1])  
            valid_pos_emb = torch.where(mask.unsqueeze(-1), pos_emb, flip_pos_emb)

            current_bs = torch.arange(bs, device=pos_emb.device).view(bs, 1, 1).expand(bs, from_pos, seq_len)
            rand_bs = torch.randint(0, bs - 1, (bs, from_pos, seq_len), device=pos_emb.device)
            rand_bs = rand_bs + (rand_bs >= current_bs).long()
            rand_seq = torch.randint(0, seq_len, (bs, from_pos, seq_len), device=pos_emb.device)
            neg_emb_pos = valid_pos_emb[rand_bs, rand_seq, :]  # (bs, from_pos, seq_len, dim)
   
            index_conflict = torch.where(torch.sum(torch.abs(neg_emb_pos - pos_emb.unsqueeze(1)), dim=-1) == 0)          
            if len(index_conflict[0]) > 0:
                print(f"Warning: found {len(index_conflict[0])} conflicts in positive sampling, resample")
                num_conflicts = len(index_conflict[0])
                new_rand_bs = torch.randint(0, bs - 1, (num_conflicts,), device=pos_emb.device)
                conflict_current_bs = current_bs[index_conflict[0], index_conflict[1], index_conflict[2]]
                new_rand_bs = new_rand_bs + (new_rand_bs >= conflict_current_bs).long()        
                new_rand_seq = torch.randint(0, seq_len, (num_conflicts,), device=pos_emb.device)
                neg_emb_pos[index_conflict[0], index_conflict[1], index_conflict[2]] = valid_pos_emb[new_rand_bs, new_rand_seq, :]
            index_conflict = torch.where(torch.sum(torch.abs(neg_emb_pos - pos_emb.unsqueeze(1)), dim=-1) == 0)
            if len(index_conflict[0]) > 0:
                print(f"Warning: Resample found {len(index_conflict[0])} conflicts in positive sampling")

        return torch.cat([neg_emb_pos, neg_emb_neg], dim=1)
                            
            
