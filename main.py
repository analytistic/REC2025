import argparse
import json
import os
import time
from pathlib import Path
import shutil

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

import random
import numpy as np
import torch

def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407) 

from dotenv import load_dotenv
load_dotenv('/Users/alex/project/Rec/rec_2025_rebuild/base.env') 


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--loss_type', default='bce', type=str, choices=['infonce', 'bce', 'bpr', 'cosine_triplet', 'triplet', 'inbatch_infonce', 'ado_infonce'])

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('USER_CACHE_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    config_path = os.path.join(os.path.dirname(__file__), 'config/')
    cfg = BaseConfig(config_path, vars(args))

    dataset = MyDataset(data_path, args)
    # train_dataset = valid_dataset = dataset  
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.99, 0.01])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, cfg).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.logencoder.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    model.logencoder.time_stamp_emb['hour'].weight.data[0, :] = 0
    model.logencoder.time_stamp_emb['day'].weight.data[0, :] = 0
    model.logencoder.time_stamp_emb['month'].weight.data[0, :] = 0
    model.logencoder.time_stamp_emb['minute'].weight.data[0, :] = 0
    model.logencoder.act_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        
        try:
            model.load_state_dict(torch.load(Path(os.environ.get('USER_CACHE_PATH'), f"temp", 'model.pt'), map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')


    optimizer = get_optim(cfg, model)
    
    # 初始化混合精度训练
    scaler = None
    if cfg.use_amp and torch.cuda.is_available():
        scaler = GradScaler(
            init_scale=cfg.amp_init_scale,
            growth_factor=cfg.amp_growth_factor,
            backoff_factor=cfg.amp_backoff_factor,
            growth_interval=cfg.amp_growth_interval
        )
        print(f"Mixed precision training enabled with initial scale: {cfg.amp_init_scale}")
    else:
        print("Mixed precision training disabled")
    
    if cfg.scheduler.act:
        scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.scheduler, len(train_loader) * args.num_epochs)


    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    print(model)
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if scaler is not None:
                with autocast():
                    loss_list, pos_score, neg_score, neg_var, neg_max = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
                    )
                    loss = torch.stack(loss_list).sum()
                    
                    if args.l2_emb > 0.0:
                        for param in model.item_emb.parameters():
                            loss += args.l2_emb * torch.norm(param)
                
                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                log_gradient_stats(model, writer, global_step, log_freq=cfg.logging.log_grad_freq)
                clip_gradients(model, cfg.grad_clip_norm)
                
                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准精度训练
                loss_list, pos_score, neg_score, neg_var, neg_max = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
                )
                loss = torch.stack(loss_list).sum()
                
                if args.l2_emb > 0.0:
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)
                
                loss.backward()
                log_gradient_stats(model, writer, global_step, log_freq=cfg.logging.log_grad_freq)
                clip_gradients(model, cfg.grad_clip_norm)
                optimizer.step()

            if cfg.scheduler.act:
                scheduler.step()
                writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)

            # 记录训练指标
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            for i, loss_i in enumerate(loss_list):
                if i == 0: writer.add_scalar(f'Loss/train_{cfg.loss_type}', loss_i.item(), global_step)
                else: 
                    writer.add_scalar(f'Loss/train_{cfg.sub_loss[i-1]}', loss_i.item(), global_step)

            writer.add_scalar('Loss/pos_score', pos_score, global_step)
            writer.add_scalar('Loss/neg_score', neg_score, global_step)
            writer.add_scalar('Loss/neg_var', neg_var, global_step)
            writer.add_scalar('Loss/neg_max', neg_max, global_step)
            
            # 记录混合精度相关指标
            if scaler is not None:
                writer.add_scalar('AMP/scale', scaler.get_scale(), global_step)
            
            global_step += 1
            del loss, pos_score, neg_score, neg_var, neg_max


        temp_save_dir = Path(os.environ.get('USER_CACHE_PATH'), f"temp")
        temp_save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), temp_save_dir / "model.pt")
                    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  

        model.load_state_dict(torch.load(temp_save_dir / "model.pt", map_location=torch.device(args.device))) 
        model.eval()
        valid_loss_sum = 0
        valid_bce_sum = 0
        batch_acc_k = {}
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                
                # 验证时也使用混合精度以保持一致性
                if scaler is not None:
                    with autocast():
                        main_loss, bce_loss, log_feats, pos_embs, candidate_item, loss_mask = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
                        )
                else:
                    main_loss, bce_loss, log_feats, pos_embs, candidate_item, loss_mask = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_time, seq_action_type
                    )
                
                acc_k = evaluate_metrics(
                    log_feats, pos_embs, candidate_item, loss_mask, cfg.eval.acc_k, distance=cfg.eval.distance
                )
                for k, v in acc_k.items():
                    batch_acc_k[k] = batch_acc_k.get(k, 0.0) + v

                valid_loss_sum += main_loss.item()
                valid_bce_sum += bce_loss.item()

        valid_loss_sum /= len(valid_loader)
        valid_bce_sum /= len(valid_loader)
        for k, v in batch_acc_k.items():
            batch_acc_k[k] = v / len(valid_loader)
            writer.add_scalar(f'Acc/valid_{k}', batch_acc_k[k], global_step)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        writer.add_scalar('Loss/valid_bce', valid_bce_sum, global_step)
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

        temp_save_dir = Path(os.environ.get('USER_CACHE_PATH'), f"temp")
        if temp_save_dir.exists():
            shutil.rmtree(temp_save_dir)





    print("Done")
    writer.close()
    log_file.close()
