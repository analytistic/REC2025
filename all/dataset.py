import json
import pickle
import struct
from pathlib import Path
import os
import numpy as np
import torch
from tqdm import tqdm
from time import time
import toml
from utils.neg_sample import batch_neg_sample

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        config_path = os.path.join(os.path.dirname(__file__), 'utils', 'negsample_config.toml')
        self.negsample_cfg = toml.load(config_path)
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        time_start = time()
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据
        
        ext_user_sequence = []  
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.negsample_cfg['random'], self.maxlen + 1], dtype=np.int32)  
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.negsample_cfg['random'], self.maxlen + 1], dtype=object)  

        nxt = ext_user_sequence[-1]
        idx = self.maxlen
        
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                for i in range(self.negsample_cfg['random']):
                    neg_id = self._random_neq(1, self.itemnum + 1, ts)
                    neg[i, idx] = neg_id  
                    neg_feat[i, idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], int(neg_id))
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)
        time_generate = time() - time_start

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch, feature_default_value=None, down_sample_window=None, item_feat_dict=None):
        """
        Args:
            batch: 多个__getitem__返回的数据
            feature_default_value: 特征默认值
            down_sample_window: 滑动窗口大小列表，例如[10, 15, 20]表示对每个序列生成这三种窗口大小的子序列
            neg_num: 负样本数量，从配置文件中读取
            item_feat_dict: 物品特征字典，用于获取真实的负样本特征

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式（增加了批内负采样）
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            next_action_type: 下一个动作类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        time_generate_start = time()
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        
        if down_sample_window is None or len(down_sample_window) == 0:
            # 不进行滑动窗口采样，进行批内负采样处理
            seq_np = np.array(seq)
            pos_np = np.array(pos)
            neg_np = np.array(neg)  # shape: [batch_size, 1, seq_len]
            token_type_np = np.array(token_type)
            next_token_type_np = np.array(next_token_type)
            next_action_type_np = np.array(next_action_type)
            
                       


            neg_np, neg_feat = batch_neg_sample(pos_np, pos_feat, neg_np, neg_feat, next_token_type, item_feat_dict)


                
                
            
            # 转换为torch tensor
            seq = torch.from_numpy(seq_np)
            pos = torch.from_numpy(pos_np)
            neg = torch.from_numpy(neg_np)
            token_type = torch.from_numpy(token_type_np)
            next_token_type = torch.from_numpy(next_token_type_np)
            next_action_type = torch.from_numpy(next_action_type_np)
            seq_feat = list(seq_feat)
            pos_feat = list(pos_feat)
            neg_feat = list(neg_feat)
        else:
            # 使用NumPy矩阵机制进行多窗口大小的滑动窗口处理
            seq_tmp = np.array(seq)
            pos_tmp = np.array(pos)
            neg_tmp = np.array(neg)
            token_type_tmp = np.array(token_type)
            next_token_type_tmp = np.array(next_token_type)
            next_action_type_tmp = np.array(next_action_type)
            
            batch_size = len(seq)
            seq_len = seq_tmp.shape[1]
            
            # 1. 批量计算所有序列的有效长度
            # 找到每个序列的item [batch_size, seq_len]
            valid_mask = (token_type_tmp == 1)
            
            # 计算每个序列的有效长度 [batch_size]
            valid_lengths = np.sum(valid_mask, axis=1)
            
            # 过滤掉太短的序列（有效长度需要大于最小窗口大小）
            min_window_size = min(down_sample_window)
            valid_seq_mask = valid_lengths >= min_window_size
            
            if not np.any(valid_seq_mask):
                # 如果没有有效序列，返回原始数据
                seq = torch.from_numpy(seq_tmp)
                pos = torch.from_numpy(pos_tmp) 
                neg = torch.from_numpy(neg_tmp)
                token_type = torch.from_numpy(token_type_tmp)
                next_token_type = torch.from_numpy(next_token_type_tmp)
                next_action_type = torch.from_numpy(next_action_type_tmp)
                seq_feat = list(seq_feat)
                pos_feat = list(pos_feat)
                neg_feat = list(neg_feat)
                time_generate_end = time()
                time_generate = time_generate_end - time_generate_start
                return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_generate
            
            # 2. 为每个窗口大小预计算所有序列的可生成子序列数量
            valid_indices = np.where(valid_seq_mask)[0]
            window_sizes = np.array(down_sample_window)  # [num_windows]
            
            # 收集所有生成的子序列
            batch_seq = []
            batch_pos = []
            batch_neg = []
            batch_token_type = []
            batch_next_token_type = []
            batch_next_action_type = []
            batch_seq_feat = []
            batch_pos_feat = []
            batch_neg_feat = []

            # 3. 为每个有效序列和每个窗口大小生成滑动窗口子序列
            for batch_idx in valid_indices:
                batch_seq.append(seq_tmp[batch_idx])
                batch_pos.append(pos_tmp[batch_idx])
                batch_neg.append(neg_tmp[batch_idx])
                batch_token_type.append(token_type_tmp[batch_idx])
                batch_next_token_type.append(next_token_type_tmp[batch_idx])
                batch_next_action_type.append(next_action_type_tmp[batch_idx])
                batch_seq_feat.append(seq_feat[batch_idx])
                batch_pos_feat.append(pos_feat[batch_idx])
                batch_neg_feat.append(neg_feat[batch_idx])

                # 获取当前序列的有效位置
                valid_positions = np.where(valid_mask[batch_idx])[0]
                seq_valid_length = len(valid_positions)
                
                # 对每个窗口大小进行处理
                for window_size in window_sizes:
                    if window_size > seq_valid_length:
                        continue  # 窗口大小超过序列长度，跳过
                    
                    # 计算当前窗口大小的步长和可生成的子序列数量
                    step_size = max(1, window_size // 2)
                    max_start_idx = seq_valid_length - window_size
                    
                    # 生成起点索引数组：[0, step_size, 2*step_size, ...]
                    num_subseqs = (max_start_idx // step_size) + 1
                    start_indices = np.arange(num_subseqs) * step_size
                    # 确保不会超出有效范围
                    start_indices = start_indices[start_indices <= max_start_idx]
                    
                    # 使用矩阵操作批量生成当前窗口大小的所有子序列
                    for start_idx in start_indices:
                        end_idx = start_idx + window_size
                        
                        # 创建新的子序列（初始化为零）
                        new_seq = np.zeros_like(seq_tmp[batch_idx])
                        new_pos = np.zeros_like(pos_tmp[batch_idx])
                        new_neg = np.zeros_like(neg_tmp[batch_idx])
                        new_token_type = np.zeros_like(token_type_tmp[batch_idx])
                        new_next_token_type = np.zeros_like(next_token_type_tmp[batch_idx])
                        new_next_action_type = np.zeros_like(next_action_type_tmp[batch_idx])
                        new_seq_feat = np.full_like(seq_feat[batch_idx], None, dtype=object)
                        new_pos_feat = np.full_like(pos_feat[batch_idx], None, dtype=object)
                        new_neg_feat = np.full_like(neg_feat[batch_idx], None, dtype=object)
                        
                        # 使用left-padding方式正确填充子序列（参考__getitem__的做法）
                        window_positions = valid_positions[start_idx:end_idx]
                        
                        # 从窗口的最后一个元素开始，向前填充
                        new_seq[-(len(window_positions)):] = seq_tmp[batch_idx][window_positions]
                        new_pos[-(len(window_positions)):] = pos_tmp[batch_idx][window_positions]
                        new_neg[-(len(window_positions)):] = neg_tmp[batch_idx][window_positions]
                        new_token_type[-(len(window_positions)):] = token_type_tmp[batch_idx][window_positions]
                        new_next_token_type[-(len(window_positions)):] = next_token_type_tmp[batch_idx][window_positions]
                        new_next_action_type[-(len(window_positions)):] = next_action_type_tmp[batch_idx][window_positions]
                        new_seq_feat[-(len(window_positions)):] = seq_feat[batch_idx][window_positions]
                        new_pos_feat[-(len(window_positions)):] = pos_feat[batch_idx][window_positions]
                        new_neg_feat[-(len(window_positions)):] = neg_feat[batch_idx][window_positions]
                        
                        new_seq[-(len(window_positions)+1)] = seq_tmp[batch_idx][valid_positions[0]-1]
                        new_pos[-(len(window_positions)+1)] = new_seq[-(len(window_positions))]
                        new_neg[-(len(window_positions)+1)] = neg_tmp[batch_idx][valid_positions[0]-1]
                        new_token_type[-(len(window_positions)+1)] = token_type_tmp[batch_idx][valid_positions[0]-1]
                        new_next_token_type[-(len(window_positions)+1)] = new_token_type[-(len(window_positions))]
                        new_next_action_type[-(len(window_positions)+1)] = next_action_type_tmp[batch_idx][window_positions[0]-1]
                        new_seq_feat[-(len(window_positions)+1)] = seq_feat[batch_idx][valid_positions[0]-1]
                        new_pos_feat[-(len(window_positions)+1)] = new_seq_feat[-(len(window_positions))]
                        new_neg_feat[-(len(window_positions)+1)] = neg_feat[batch_idx][valid_positions[0]-1]
                        new_seq_feat[:-(len(window_positions)+1)] = feature_default_value
                        new_pos_feat[:-(len(window_positions)+1)] = feature_default_value
                        new_neg_feat[:-(len(window_positions)+1)] = feature_default_value

                        # 添加到新batch
                        batch_seq.append(new_seq)
                        batch_pos.append(new_pos)
                        batch_neg.append(new_neg)
                        batch_token_type.append(new_token_type)
                        batch_next_token_type.append(new_next_token_type)
                        batch_next_action_type.append(new_next_action_type)
                        batch_seq_feat.append(new_seq_feat)
                        batch_pos_feat.append(new_pos_feat)
                        batch_neg_feat.append(new_neg_feat)
            
            # 转换为torch tensor
            seq = torch.from_numpy(np.array(batch_seq))
            pos = torch.from_numpy(np.array(batch_pos))
            neg = torch.from_numpy(np.array(batch_neg))
            token_type = torch.from_numpy(np.array(batch_token_type))
            next_token_type = torch.from_numpy(np.array(batch_next_token_type))
            next_action_type = torch.from_numpy(np.array(batch_next_action_type))
            seq_feat = batch_seq_feat
            pos_feat = batch_pos_feat
            neg_feat = batch_neg_feat
        
        time_generate_end = time()
        time_generate = time_generate_end - time_generate_start
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_generate


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        # try:
        #     base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
        #     for part_file in base_path.glob('part-*'):
        #         with open(part_file, 'r', encoding='utf-8') as file:
        #             for line in file:
        #                 if line:
        #                     data_dict_origin = json.loads(line.strip())
        #                     if 'emb' not in data_dict_origin.keys():
        #                         insert_emb = np.zeros(shape, dtype=np.float32)
        #                     else:
        #                         insert_emb = data_dict_origin['emb']
        #                     if isinstance(insert_emb, list):
        #                         insert_emb = np.array(insert_emb, dtype=np.float32)
        #                     data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
        #                     emb_dict.update(data_dict)
        # except Exception as e:
        #     print(f"transfer error: {e}")

        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


