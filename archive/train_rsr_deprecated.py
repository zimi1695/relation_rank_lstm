"""
train_rsr.py

恢复 +20% 配置：
  - 无 L2 归一化
  - LeakyReLU 保留
  - ALPHA=0.1
  - MSE(pred, gt) + rank_loss
  - mask = 目标日有效即可
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from time import time

ROOT_DIR     = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR     = os.path.join(ROOT_DIR, 'data')
PRETRAIN_DIR = os.path.join(ROOT_DIR, 'pretrain')
MARKET       = 'CSI300'

SEQ         = 16
UNIT        = 64
LR          = 0.001
ALPHA       = 0.1
EPOCHS      = 50
STEPS       = 5

VALID_INDEX = 1242
TEST_INDEX  = 1368

SEED       = 123456789
RET_DIM    = 3
RSR_IN_DIM = UNIT + RET_DIM
RET_COLS   = [5, 6, 7]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print(f'relation encoding shape: {relation_encoding.shape}')
    rel_shape  = relation_encoding.shape[:2]
    mask_flags = np.equal(
        np.zeros(rel_shape, dtype=int),
        np.sum(relation_encoding, axis=2)
    )
    mask = np.where(mask_flags,
                    np.ones(rel_shape) * -1e9,
                    np.zeros(rel_shape))
    return relation_encoding, mask


class RSRModel(nn.Module):
    def __init__(self, in_dim, rel_encoding, rel_mask):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.register_buffer('relation',
                             torch.tensor(rel_encoding, dtype=torch.float32))
        self.register_buffer('rel_mask',
                             torch.tensor(rel_mask,     dtype=torch.float32))
        K = rel_encoding.shape[2]
        self.rel_fc  = nn.Linear(K,          1, bias=False)
        self.head_fc = nn.Linear(in_dim,     1, bias=False)
        self.tail_fc = nn.Linear(in_dim,     1, bias=False)
        self.pred_fc = nn.Linear(in_dim * 2, 1, bias=False)

    def forward(self, feature):
        N       = feature.shape[0]
        all_one = torch.ones(N, 1, device=feature.device)
        rel_w  = self.lrelu(self.rel_fc(self.relation)).squeeze(-1)
        head_w = self.lrelu(self.head_fc(feature))
        tail_w = self.lrelu(self.tail_fc(feature))
        weight = head_w @ all_one.T + all_one @ tail_w.T + rel_w
        w_mask = torch.softmax(self.rel_mask + weight, dim=0)
        proped = w_mask @ feature
        # ← 恢复：无 L2 归一化，直接 LeakyReLU
        pred   = self.lrelu(self.pred_fc(torch.cat([feature, proped], dim=1)))
        return pred


def rank_loss_fn(pred, gt, mask):
    all_one    = torch.ones_like(pred)
    pre_pw_dif = pred @ all_one.T - all_one @ pred.T
    gt_pw_dif  = all_one @ gt.T   - gt @ all_one.T
    mask_pw    = mask @ mask.T
    return torch.mean(torch.relu(pre_pw_dif * gt_pw_dif * mask_pw))


def main():
    set_seed(SEED)

    ticker_file = os.path.join(DATA_DIR, f'{MARKET}_tickers.txt')
    tickers     = np.genfromtxt(ticker_file, dtype=str)
    N           = len(tickers)

    emb_fname = f'{MARKET}_rank_lstm_seq-{SEQ}_unit-{UNIT}.npy'
    embedding = np.load(os.path.join(PRETRAIN_DIR, emb_fname))
    print(f'Embedding shape: {embedding.shape}')

    rel_file = os.path.join(DATA_DIR, f'{MARKET}_wiki_relation.npy')
    rel_encoding, rel_mask = load_relation_data(rel_file)

    from pretrain import load_eod_data
    eod_data, mask_data, gt_data, _ = load_eod_data(
        DATA_DIR, MARKET, tickers, STEPS
    )
    T = mask_data.shape[1]
    gt_valid = gt_data[np.abs(gt_data) > 1e-8]
    print(f'N={N}  T={T}  STEPS={STEPS}  ALPHA={ALPHA}')
    print(f'GT: mean={gt_valid.mean():.4f}  max={np.abs(gt_valid).max():.4f}')

    mask_t = torch.tensor(mask_data, dtype=torch.float32)
    gt_t   = torch.tensor(gt_data,   dtype=torch.float32)
    mse_fn = nn.MSELoss(reduction='none')

    model     = RSRModel(RSR_IN_DIM, rel_encoding, rel_mask)
    optimizer = Adam(model.parameters(), lr=LR)

    best_valid_loss = np.inf
    best_state      = None
    n_train         = VALID_INDEX - SEQ - STEPS + 1
    batch_offsets   = np.arange(VALID_INDEX)

    for epoch in range(EPOCHS):
        t0 = time()
        model.train()
        np.random.shuffle(batch_offsets)

        tra_loss = tra_reg = tra_rank = 0.0
        valid_batches = 0

        for j in range(n_train):
            offset = int(batch_offsets[j])
            if offset + SEQ + STEPS > T:
                continue

            emb_b     = torch.tensor(
                embedding[:, offset, :], dtype=torch.float32
            )
            ret_raw   = eod_data[:, offset + SEQ - 1, :][:, RET_COLS]
            ret_clean = np.where(ret_raw < -1000, 0.0, ret_raw).astype(np.float32)
            ret_t     = torch.tensor(ret_clean, dtype=torch.float32)
            feature   = torch.cat([emb_b, ret_t], dim=1)

            mask_b = mask_t[:, offset + SEQ + STEPS - 1].unsqueeze(1)
            gt_b   = gt_t[:,   offset + SEQ + STEPS - 1].unsqueeze(1)

            if mask_b.sum() < 2:
                continue

            pred = model(feature)
            reg  = ((mse_fn(pred, gt_b) * mask_b).sum()
                    / mask_b.sum().clamp(min=1e-8))
            rk   = rank_loss_fn(pred, gt_b, mask_b)
            loss = reg + ALPHA * rk

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tra_loss  += loss.item()
            tra_reg   += reg.item()
            tra_rank  += rk.item()
            valid_batches += 1

        model.eval()
        val_loss = val_reg = val_rank = 0.0
        valid_val_cnt = 0
        with torch.no_grad():
            for cur_offset in range(
                VALID_INDEX - SEQ - STEPS + 1,
                TEST_INDEX  - SEQ - STEPS + 1
            ):
                if cur_offset + SEQ + STEPS > T:
                    continue
                emb_b     = torch.tensor(
                    embedding[:, cur_offset, :], dtype=torch.float32
                )
                ret_raw   = eod_data[:, cur_offset + SEQ - 1, :][:, RET_COLS]
                ret_clean = np.where(ret_raw < -1000, 0.0, ret_raw).astype(np.float32)
                ret_t     = torch.tensor(ret_clean, dtype=torch.float32)
                feature   = torch.cat([emb_b, ret_t], dim=1)

                mask_b = mask_t[:, cur_offset + SEQ + STEPS - 1].unsqueeze(1)
                gt_b   = gt_t[:,   cur_offset + SEQ + STEPS - 1].unsqueeze(1)

                if mask_b.sum() < 2:
                    continue

                pred = model(feature)
                reg  = ((mse_fn(pred, gt_b) * mask_b).sum()
                        / mask_b.sum().clamp(min=1e-8))
                rk   = rank_loss_fn(pred, gt_b, mask_b)
                val_loss  += (reg + ALPHA * rk).item()
                val_reg   += reg.item()
                val_rank  += rk.item()
                valid_val_cnt += 1

        n_t = max(valid_batches, 1)
        n_v = max(valid_val_cnt, 1)
        avg_val = val_loss / n_v
        mark    = ' ✓ 保存' if avg_val < best_valid_loss else ''

        if epoch < 3:
            print(f'Epoch {epoch:02d}  '
                  f'loss={tra_loss/n_t:.5f}'
                  f'(reg={tra_reg/n_t:.5f} rk={tra_rank/n_t:.5f})  '
                  f'val={avg_val:.5f}'
                  f'(reg={val_reg/n_v:.5f} rk={val_rank/n_v:.5f})  '
                  f'({time()-t0:.1f}s){mark}')
        else:
            print(f'Epoch {epoch:02d}  '
                  f'train={tra_loss/n_t:.5f}  '
                  f'valid={avg_val:.5f}  '
                  f'({time()-t0:.1f}s){mark}')

        if avg_val < best_valid_loss:
            best_valid_loss = avg_val
            best_state      = copy.deepcopy(model.state_dict())

    save_path = os.path.join(PRETRAIN_DIR, 'CSI300_rsr_best.pt')
    torch.save({
        'state_dict':   best_state,
        'in_dim':       RSR_IN_DIM,
        'rel_encoding': rel_encoding,
        'rel_mask':     rel_mask,
        'valid_loss':   best_valid_loss,
    }, save_path)
    print(f'\n最优valid loss: {best_valid_loss:.5f}')
    print(f'模型已保存 → {save_path}')


if __name__ == '__main__':
    main()