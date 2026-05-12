"""
pretrain.py

修复记录：
  FIX-3: batch_offsets 从 np.arange(VALID_INDEX) 改为
          np.arange(VALID_INDEX - SEQ - STEPS + 1)
          防止训练阶段采到 GT 落在验证区间的 offset
  FIX-4: ground_truth 从 close-to-close（col 5）改为 open-to-open（col 9）
          与 score_self.py 的评分公式对齐
  FIX-2: embedding 生成范围从 T-SEQ-STEPS+1 扩展到 T-SEQ+1
          让最新窗口的 embedding 也能被生成，消除预测 offset 错位
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
VALID_INDEX = 1400
TEST_INDEX  = 1470
SEED        = 123456789

FEAT_COLS = 8    # LSTM输入：col 1-8（ma5~ret_20），不含 open
OPEN_COL  = 9    # FIX-4：open 价格列，用于计算 GT


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_eod_data(data_dir, market, tickers, steps=1):
    eod_data = masks = ground_truth = base_price = None

    for index, ticker in enumerate(tickers):
        fpath  = os.path.join(data_dir, f'{market}_{ticker}_1.csv')
        single = np.genfromtxt(fpath, dtype=np.float32,
                               delimiter=',', skip_header=False)
        if index == 0:
            N, T  = len(tickers), single.shape[0]
            eod_data     = np.zeros([N, T, FEAT_COLS], dtype=np.float32)
            masks        = np.ones( [N, T],            dtype=np.float32)
            ground_truth = np.zeros([N, T],            dtype=np.float32)
            base_price   = np.zeros([N, T],            dtype=np.float32)

        for row in range(single.shape[0]):
            # mask：ret_20（col 8）缺失则无效
            if abs(single[row][8] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row >= steps:
                # ── FIX-4：GT 改用 open-to-open（col 9）────────────────────
                # 与 score_self.py 的评分公式对齐：
                #   (open[week_end] - open[week_start]) / open[week_start]
                cur_open  = single[row][OPEN_COL]
                prev_open = single[row - steps][OPEN_COL]
                if (abs(cur_open  + 1234) > 1e-8 and cur_open  > 1e-8 and
                        abs(prev_open + 1234) > 1e-8 and prev_open > 1e-8):
                    ground_truth[index][row] = (cur_open - prev_open) / prev_open

            for col in range(single.shape[1]):
                if abs(single[row][col] + 1234) < 1e-8:
                    single[row][col] = 1.1

        eod_data[index]   = single[:, 1:1+FEAT_COLS]   # col 1-8
        base_price[index] = single[:, 5]                # 收盘价（窗口归一化用）

    return eod_data, masks, ground_truth, base_price


class RankLSTM(nn.Module):
    def __init__(self, fea_dim, unit):
        super().__init__()
        self.lstm  = nn.LSTM(input_size=fea_dim, hidden_size=unit,
                             batch_first=True)
        self.fc    = nn.Linear(unit, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out, _  = self.lstm(x)
        seq_emb = out[:, -1, :]
        pred    = self.lrelu(self.fc(seq_emb))
        return seq_emb, pred


def rank_loss_fn(return_ratio, ground_truth, mask):
    all_one    = torch.ones_like(return_ratio)
    pre_pw_dif = return_ratio @ all_one.T - all_one @ return_ratio.T
    gt_pw_dif  = all_one @ ground_truth.T - ground_truth @ all_one.T
    mask_pw    = mask @ mask.T
    return torch.mean(torch.relu(pre_pw_dif * gt_pw_dif * mask_pw))


def get_batch(eod_data, mask_data, price_data, gt_data, offset, steps=STEPS):
    seq    = eod_data[:, offset: offset + SEQ, :].copy()
    mask_b = np.min(
        mask_data[:, offset: offset + SEQ + steps],
        axis=1, keepdims=True
    )
    base = seq[:, 0, 4].copy()
    base = np.where(np.abs(base) < 1e-8, np.ones_like(base), base)
    seq  = seq / base[:, np.newaxis, np.newaxis]

    price_b = price_data[:, offset + SEQ - 1] / base
    gt_b    = gt_data[:,    offset + SEQ + steps - 1]

    return (seq,
            mask_b,
            np.expand_dims(price_b, axis=1),
            np.expand_dims(gt_b,    axis=1))


def main():
    set_seed()
    os.makedirs(PRETRAIN_DIR, exist_ok=True)

    ticker_file = os.path.join(DATA_DIR, f'{MARKET}_tickers.txt')
    tickers     = np.genfromtxt(ticker_file, dtype=str)
    N           = len(tickers)
    print(f'股票数: {N}')

    eod_data, mask_data, gt_data, price_data = load_eod_data(
        DATA_DIR, MARKET, tickers, STEPS
    )
    T = mask_data.shape[1]
    F = eod_data.shape[2]
    gt_valid = gt_data[np.abs(gt_data) > 1e-8]
    print(f'交易日 T={T}  特征维度 F={F}  STEPS={STEPS}')
    print(f'GT 统计（open-to-open）: mean={gt_valid.mean():.4f}  '
          f'std={gt_valid.std():.4f}  max={np.abs(gt_valid).max():.4f}')

    model     = RankLSTM(fea_dim=F, unit=UNIT)
    optimizer = Adam(model.parameters(), lr=LR)
    mse_fn    = nn.MSELoss(reduction='none')

    best_valid_loss = np.inf
    best_state      = None

    # ── FIX-3：只包含 GT 落在训练区间内的 offset ──────────────────────
    # 原来：np.arange(VALID_INDEX) 含高 index，shuffle 后 GT 会越过 VALID_INDEX
    # 修复：上界改为 VALID_INDEX - SEQ - STEPS + 1，保证 gt_idx < VALID_INDEX
    n_train       = VALID_INDEX - SEQ - STEPS + 1
    batch_offsets = np.arange(n_train)           # ← FIX-3
    n_valid       = TEST_INDEX - VALID_INDEX

    for epoch in range(EPOCHS):
        t0 = time()
        model.train()
        np.random.shuffle(batch_offsets)

        tra_loss = 0.0
        for j in range(n_train):
            seq_b, mask_b, price_b, gt_b = get_batch(
                eod_data, mask_data, price_data, gt_data,
                batch_offsets[j]
            )
            seq_t   = torch.tensor(seq_b,   dtype=torch.float32)
            mask_t  = torch.tensor(mask_b,  dtype=torch.float32)
            price_t = torch.tensor(price_b, dtype=torch.float32)
            gt_t    = torch.tensor(gt_b,    dtype=torch.float32)

            _, pred   = model(seq_t)
            ret_ratio = (pred - price_t) / price_t
            reg  = ((mse_fn(ret_ratio, gt_t) * mask_t).sum()
                    / mask_t.sum().clamp(min=1e-8))
            rk   = rank_loss_fn(ret_ratio, gt_t, mask_t)
            loss = reg + ALPHA * rk

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tra_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cur_offset in range(
                VALID_INDEX - SEQ - STEPS + 1,
                TEST_INDEX  - SEQ - STEPS + 1
            ):
                seq_b, mask_b, price_b, gt_b = get_batch(
                    eod_data, mask_data, price_data, gt_data, cur_offset)
                seq_t   = torch.tensor(seq_b,   dtype=torch.float32)
                mask_t  = torch.tensor(mask_b,  dtype=torch.float32)
                price_t = torch.tensor(price_b, dtype=torch.float32)
                gt_t    = torch.tensor(gt_b,    dtype=torch.float32)

                _, pred   = model(seq_t)
                ret_ratio = (pred - price_t) / price_t
                reg  = ((mse_fn(ret_ratio, gt_t) * mask_t).sum()
                        / mask_t.sum().clamp(min=1e-8))
                rk   = rank_loss_fn(ret_ratio, gt_t, mask_t)
                val_loss += (reg + ALPHA * rk).item()

        avg_val = val_loss / n_valid
        mark    = ' ✓ 保存' if avg_val < best_valid_loss else ''
        print(f'Epoch {epoch:02d}  '
              f'train={tra_loss/n_train:.5f}  '
              f'valid={avg_val:.5f}  '
              f'({time()-t0:.1f}s){mark}')

        if avg_val < best_valid_loss:
            best_valid_loss = avg_val
            best_state      = copy.deepcopy(model.state_dict())

    print('\n生成 Embedding...')
    model.load_state_dict(best_state)
    model.eval()

    # ── FIX-2：生成范围扩展到 T-SEQ（含），消除预测 offset 错位 ──────
    # 原来：range(T - SEQ - STEPS + 1)，最新窗口的 embedding 未生成
    # 修复：range(T - SEQ + 1)，offset=T-SEQ 时序列覆盖 [T-SEQ, T)，合法
    embedding = np.zeros([N, T, UNIT], dtype=np.float32)
    with torch.no_grad():
        for offset in range(T - SEQ + 1):          # ← FIX-2
            seq_b, _, _, _ = get_batch(
                eod_data, mask_data, price_data, gt_data,
                # get_batch 内部访问 gt_data[:,offset+SEQ+STEPS-1]
                # 当 offset 接近 T-SEQ 时该 index 可能越界，用 min 保护
                min(offset, T - SEQ - STEPS)
            )
            # 但 seq 部分只取 offset 窗口，重新取正确的 seq
            seq_np = eod_data[:, offset: offset + SEQ, :].copy()
            base   = seq_np[:, 0, 4].copy()
            base   = np.where(np.abs(base) < 1e-8, np.ones_like(base), base)
            seq_np = seq_np / base[:, np.newaxis, np.newaxis]
            seq_t  = torch.tensor(seq_np, dtype=torch.float32)

            seq_emb, _ = model(seq_t)
            embedding[:, offset, :] = seq_emb.numpy()

    emb_fname = f'{MARKET}_rank_lstm_seq-{SEQ}_unit-{UNIT}.npy'
    np.save(os.path.join(PRETRAIN_DIR, emb_fname), embedding)
    print(f'Embedding shape: {embedding.shape}  '
          f'（有效范围: offset 0~{T-SEQ}）')

    lstm_path = os.path.join(PRETRAIN_DIR, f'{MARKET}_lstm_weights.pt')
    torch.save(best_state, lstm_path)
    print(f'已保存 → {lstm_path}')


if __name__ == '__main__':
    main()