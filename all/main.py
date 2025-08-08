import argparse
import json
import os
import time
from pathlib import Path
import pprint
import numpy as np
import torch
import toml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.metrics import evaluate_metrics

from dataset import MyDataset
from model import BaselineModel

from utils.train_utils import (
    create_optimizer, 
    create_scheduler,
    log_gradient_stats,
    clip_gradients
)
# from dotenv import load_dotenv
import random

from functools import partial

# load_dotenv(dotenv_path="/Users/alex/project/Rec/rec_2025/base.env")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果用多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_args_from_config(args, config):
    """
    从配置文件创建完整的args对象
    """
    # 直接使用原始args对象，添加配置文件中的属性
    model_config = config.get('model', {})
    args.hidden_units = model_config.get('hidden_units', 32)
    args.num_blocks = model_config.get('num_blocks', 1)
    args.num_heads = model_config.get('num_heads', 1)
    args.dropout_rate = model_config.get('dropout_rate', 0.2)
    args.maxlen = model_config.get('maxlen', 101)
    args.norm_first = model_config.get('norm_first', False)
    args.dff = model_config.get('dff', 32)
    
    training_config = config.get('training', {})
    if not hasattr(args, 'batch_size') or args.batch_size is None:
        args.batch_size = training_config.get('batch_size', 128)
    args.num_epochs = training_config.get('num_epochs', 3)
    args.grad_clip_norm = training_config.get('grad_clip_norm', 1.0)
    args.grad_accumulation_steps = training_config.get('grad_accumulation_steps', 1)
    args.l2_emb = training_config.get('l2_emb', 0.0)
    
    scheduler_config = config.get('scheduler', {})
    args.warmup_steps = scheduler_config.get('warmup_steps', 1000)
    args.min_lr_ratio = scheduler_config.get('min_lr_ratio', 0.1)
    
    optimizer_config = config.get('optimizer', {})
    if not hasattr(args, 'lr') or args.lr is None:
        args.lr = optimizer_config.get('lr', 0.001)
    args.weight_decay = optimizer_config.get('weight_decay', 0.01)
    
    logging_config = config.get('logging', {})
    args.log_grad_freq = logging_config.get('log_grad_freq', 100)

    eval_config = config.get('eval', {})
    args.eval_hr_k = eval_config.get('hr_k', [1, 3])

    return args



def get_args():
    parser = argparse.ArgumentParser()

    # 配置文件路径
    parser.add_argument('--config', default='./utils/train_config.toml', type=str, help='Training configuration file path')
    
    # 可以被命令行覆盖的基本参数
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--num_epochs', default=None, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str, choices=[str(s) for s in range(81, 87)])
    
    # Loss function type
    parser.add_argument('--loss_type', default='triplet', type=str, 
                       choices=['bce', 'bpr', 'triplet', 'cosine_triplet', 'listwise_contrastive', 'focal', 'infonce'],
                       help='Loss function type to use for training')

    args = parser.parse_args()
    
    # 加载配置文件
    if os.path.exists(args.config):
        config = toml.load(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using default values")
        config = {}
    
    # 用命令行参数覆盖配置文件中的值
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        config.setdefault('optimizer', {})['lr'] = args.lr
    if args.num_epochs is not None:
        config.setdefault('training', {})['num_epochs'] = args.num_epochs
    
    # 将配置添加到args中
    args.config_dict = config

    return args


if __name__ == '__main__':

    set_seed(42)
    train_log_path = os.environ.get('TRAIN_LOG_PATH')
    train_tf_events_path = os.environ.get('TRAIN_TF_EVENTS_PATH')
    train_data_path = os.environ.get('TRAIN_DATA_PATH')
    
    if train_log_path:
        Path(train_log_path).mkdir(parents=True, exist_ok=True)
    if train_tf_events_path:
        Path(train_tf_events_path).mkdir(parents=True, exist_ok=True)
    
    log_file = open(Path(train_log_path or '.', 'train.log'), 'w')
    writer = SummaryWriter(train_tf_events_path or './tb_output')
    # global dataset
    data_path = train_data_path or './TencentGR_1k'

    args = get_args()
    config = args.config_dict
    args = create_args_from_config(args, config)
    pprint.pprint(vars(args))
    
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    # train_dataset = valid_dataset = dataset
    
    # 读取负样本配置
    negsample_cfg = toml.load('utils/negsample_config.toml')

  
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=partial(dataset.collate_fn, feature_default_value=dataset.feature_default_value,
                          down_sample_window=[], item_feat_dict=dataset.item_feat_dict)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(dataset.collate_fn, feature_default_value=None,
                          down_sample_window=[], item_feat_dict=dataset.item_feat_dict)
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0


    # 初始化sparse embedding的padding位置为0
    if hasattr(model, 'sparse_emb'):
        for name, module in model.sparse_emb.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')


    optimizer = create_optimizer(model, config)
    
    # 计算总训练步数
    total_steps = len(train_loader) * args.num_epochs
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config, total_steps)

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    print(f"Using loss function: {args.loss_type}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Gradient clipping norm: {args.grad_clip_norm}")
    print(f"Weight decay: {args.weight_decay}")
    
    # 梯度累积相关变量
    accumulated_loss = 0.0
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        
        if args.inference_only:
            break
            
        epoch_loss = 0.0
        num_batches = 0
        
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_generate = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            print(f"Generating batchsize{len(seq)} from {args.batch_size}, time_generate={time_generate:.2f}s")
     
            # 前向传播
            loss = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, 
                seq_feat, pos_feat, neg_feat, loss_type=args.loss_type
            )

            # 添加L2正则化
            if args.l2_emb > 0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
            
            # 梯度累积
            loss = loss / args.grad_accumulation_steps
            accumulated_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 每grad_accumulation_steps步或最后一步进行参数更新
            if (step + 1) % args.grad_accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 计算梯度统计信息
                log_gradient_stats(model, writer, global_step, args.log_grad_freq)
                
                # 梯度裁剪
                if args.grad_clip_norm > 0:
                    grad_norm = clip_gradients(model, args.grad_clip_norm)
                    if global_step % args.log_grad_freq == 0 and grad_norm > 0:
                        writer.add_scalar('Gradient/clipped_norm', grad_norm, global_step)
                
                # 参数更新
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 记录学习率
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Learning_Rate/lr', current_lr, global_step)
                
                # 记录训练日志
                if global_step % 1 == 0:
                    avg_accumulated_loss = accumulated_loss * args.grad_accumulation_steps
                    log_json = json.dumps({
                        'global_step': global_step, 
                        'loss': avg_accumulated_loss, 
                        'epoch': epoch, 
                        'lr': current_lr,
                        'time': time.time()
                    })
                    log_file.write(log_json + '\n')
                    log_file.flush()
                    print(log_json)
                    print(f"Generate batchsize{len(seq)} from {args.batch_size}, time_generate={time_generate:.2f}s")
                    writer.add_scalar('Loss/train', avg_accumulated_loss, global_step)
                
                # 重置累积损失
                epoch_loss += accumulated_loss * args.grad_accumulation_steps
                accumulated_loss = 0.0
                global_step += 1
                num_batches += 1

            if global_step % 3000 == 0:
                model.eval()
                valid_loss_sum = 0
                valid_bce_loss_sum = 0
                batch_hr_k = {}
                for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, _ = batch
                    seq = seq.to(args.device)
                    pos = pos.to(args.device)
                    neg = neg.to(args.device)
                    
                    # 使用指定的损失函数进行验证
                    with torch.no_grad():
                        loss, loss_bce, log_feats, pos_embs, neg_embs, loss_mask = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, 
                            seq_feat, pos_feat, neg_feat, loss_type=args.loss_type
                        )
                    
                    hr_k = evaluate_metrics(log_feats, pos_embs, neg_embs, loss_mask, k_list=args.eval_hr_k, loss_type=args.loss_type)
                    valid_loss_sum += loss.item()
                    valid_bce_loss_sum += loss_bce.item()
                    for k, v in hr_k.items():
                        batch_hr_k[k] = batch_hr_k.get(k, 0.0) + v
                        
                # 计算平均HR
                hr_k = {k: v / len(valid_loader) for k, v in batch_hr_k.items()}
                valid_loss_sum /= len(valid_loader)
                valid_bce_loss_sum /= len(valid_loader)
                writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
                writer.add_scalar('Loss/valid_bce', valid_bce_loss_sum, global_step)
                for k, v in hr_k.items():
                    writer.add_scalar(f'HitRate/valid_{k}', v, global_step)
                save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "model.pt")
            



    print("Done")
    
    writer.close()
    log_file.close()
