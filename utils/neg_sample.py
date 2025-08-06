import toml
import random
"""
实现各个阶段负采样策略
"""

def random_negative_sampling(cfg, l, r, seq: set, sample_pool: set):
    """
    生成不在序列s中的随机整数, 用于训练时的负采样

    """
  

    candidates = list(sample_pool - seq)

    return random.choices(candidates, k=int(cfg["neg_num"] * cfg["sample"]["random"]))
