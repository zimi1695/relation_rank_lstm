"""
predict.py

修复记录：
  BUG-1~8: 见前述历史修复记录
  FIX-2: infer_offset 从 T-SEQ-STEPS 改为 T-SEQ，使用最新完整窗口
          infer() 不再通过 make_batch 访问越界的 mask/gt，改为直接构建特征
  FIX-5: RSRModel.forward softmax 从 dim=0 改为 dim=1
          dim=1：每只股票对自身所有邻居做归一化（标准图注意力语义）
          dim=0：每列股票争抢同一source注意力（语义错误）
"""

import os
import sys
import copy
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional, List
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(__file__))
from pretrain import load_eod_data, FEAT_COLS

ROOT_DIR     = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR     = os.path.join(ROOT_DIR, 'data')
PRETRAIN_DIR = os.path.join(ROOT_DIR, 'pretrain')
OUTPUT_DIR   = os.path.join(ROOT_DIR, 'output')
LOG_DIR      = os.path.join(ROOT_DIR, 'logs')
MARKET       = 'CSI300'

SEQ            = 16
UNIT           = 64
RET_DIM        = 3
RSR_IN_DIM     = UNIT + RET_DIM
STEPS          = 5
TOP_K          = 5
LR             = 0.001
ROLL_EPOCHS  = 50 
ROLL_PATIENCE = 10 
FINAL_EPOCHS   = 100
FINAL_PATIENCE = 20
TRAIN_WINDOW = 126     # 原252，改为半年      v2: +24.6279%
VALID_WINDOW = 32       # 原63，对应缩短
PAD_BEGIN      = 29
SEED           = 123456789

REL_DENSITY_THRESHOLD = 0.05
RET_COLS = [5, 6, 7]

BACKTEST_START = pd.Timestamp('2025-10-13')
BACKTEST_END   = pd.Timestamp('2026-04-17')

HPARAM_SNAPSHOT = {
    'SEQ': SEQ, 'UNIT': UNIT, 'LR': LR,
    'ROLL_EPOCHS': ROLL_EPOCHS, 'ROLL_PATIENCE': ROLL_PATIENCE,
    'FINAL_EPOCHS': FINAL_EPOCHS, 'FINAL_PATIENCE': FINAL_PATIENCE,
    'TRAIN_WINDOW': TRAIN_WINDOW, 'VALID_WINDOW': VALID_WINDOW,
    'REL_DENSITY_THRESHOLD': REL_DENSITY_THRESHOLD,
    'STEPS': STEPS, 'TOP_K': TOP_K,
}


# ══════════════════════════════════════════════════════════════════
# 日志工具
# ══════════════════════════════════════════════════════════════════

def setup_logger(run_tag: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f'{run_tag}_{ts}'
    log_path = os.path.join(LOG_DIR, f'{log_name}.log')

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s',
                                      datefmt='%H:%M:%S'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


def save_summary(run_tag: str, hparams: dict,
                 backtest_results: Optional[List[dict]],
                 final_tickers: list, final_weights: list,
                 log_path: str):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if backtest_results:
        rets       = [r['return'] for r in backtest_results]
        cum_ret    = sum(rets)
        profit_wks = sum(1 for r in rets if r >= 0)
        avg_ret    = cum_ret / len(rets)
    else:
        cum_ret = profit_wks = avg_ret = None

    summary = {
        'run_tag':   run_tag,
        'timestamp': ts,
        'hparams':   hparams,
        'backtest': {
            'cum_return':   round(cum_ret * 100, 4) if cum_ret is not None else None,
            'avg_weekly':   round(avg_ret * 100, 4) if avg_ret is not None else None,
            'profit_weeks': profit_wks,
            'total_weeks':  len(backtest_results),
        } if backtest_results else None,
        'final_prediction': {'stocks': final_tickers, 'weights': final_weights},
        'log_file': log_path,
    }

    json_path = log_path.replace('.log', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(LOG_DIR, 'summary.csv')
    row = pd.DataFrame([{
        'run_tag':       run_tag,
        'timestamp':     ts,
        'cum_return_%':  round(cum_ret * 100, 4) if cum_ret is not None else '',
        'avg_weekly_%':  round(avg_ret * 100, 4) if avg_ret is not None else '',
        'profit_weeks':  profit_wks              if profit_wks is not None else '',
        'total_weeks':   len(backtest_results)   if backtest_results else '',
        'final_stocks':  '|'.join(final_tickers),
        **{f'hp_{k}': v for k, v in hparams.items()},
    }])
    row.to_csv(csv_path, mode='a',
               header=not os.path.exists(csv_path),
               index=False, encoding='utf-8-sig')

    return json_path, csv_path


# ══════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_relation_data(relation_file):
    rel = np.load(relation_file)
    N, _, K     = rel.shape
    total_pairs = N * N
    type_counts = rel.sum(axis=(0, 1))
    keep_mask   = type_counts <= REL_DENSITY_THRESHOLD * total_pairs
    n_keep      = keep_mask.sum()

    rel_filtered = rel[:, :, keep_mask]
    if n_keep == 0:
        rel_filtered = np.eye(N, dtype=rel.dtype)[:, :, np.newaxis]

    rel_sum    = rel_filtered.sum(axis=2)
    mask_flags = np.equal(np.zeros(rel_filtered.shape[:2], dtype=int), rel_sum)
    mask = np.where(mask_flags,
                    np.ones(rel_filtered.shape[:2]) * -1e9,
                    np.zeros(rel_filtered.shape[:2]))
    row_nonzero = (rel_sum > 0).sum(axis=1)
    return rel_filtered, mask, int(n_keep), float(row_nonzero.mean())


class RSRModel(nn.Module):
    def __init__(self, in_dim, rel_encoding, rel_mask):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.register_buffer('relation',
                             torch.tensor(rel_encoding, dtype=torch.float32))
        self.register_buffer('rel_mask',
                             torch.tensor(rel_mask, dtype=torch.float32))
        K = rel_encoding.shape[2]
        self.rel_fc  = nn.Linear(K,          1, bias=False)
        self.head_fc = nn.Linear(in_dim,     1, bias=False)
        self.tail_fc = nn.Linear(in_dim,     1, bias=False)
        self.pred_fc = nn.Linear(in_dim * 2, 1, bias=False)

    def forward(self, feature):
        N       = feature.shape[0]
        all_one = torch.ones(N, 1, device=feature.device)

        rel_w   = self.lrelu(self.rel_fc(self.relation)).squeeze(-1)
        head_w  = self.lrelu(self.head_fc(feature))
        tail_w  = self.lrelu(self.tail_fc(feature))
        weight  = head_w @ all_one.T + all_one @ tail_w.T + rel_w

        # ── FIX-5：softmax 改为 dim=1（行方向）────────────────────────
        # dim=1：每只目标股票对其所有邻居的注意力权重归一化（正确语义）
        # dim=0：原错误，每列在争夺同一 source 的注意力
        w_mask  = torch.softmax(self.rel_mask + weight, dim=1)  # ← FIX-5
        proped  = w_mask @ feature

        pred    = self.pred_fc(torch.cat([feature, proped], dim=1))
        return pred


def rank_loss_fn(pred, gt, mask):
    all_one    = torch.ones_like(pred)
    pre_pw_dif = pred @ all_one.T - all_one @ pred.T
    gt_pw_dif  = all_one @ gt.T   - gt @ all_one.T
    mask_pw    = mask @ mask.T
    return torch.mean(torch.relu(pre_pw_dif * gt_pw_dif * mask_pw))


def cs_normalize(emb_np):
    std = emb_np.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return emb_np / std


def make_batch(embedding, eod_data, mask_data, gt_data, offset):
    """训练/验证用：需要 GT 和完整 mask，offset+SEQ+STEPS-1 必须在 T 内。"""
    emb_np  = cs_normalize(embedding[:, offset, :].copy())
    ret_raw = eod_data[:, offset + SEQ - 1, :][:, RET_COLS]
    ret_np  = np.where(ret_raw < -1000, 0.0, ret_raw).astype(np.float32)
    ret_std = ret_np.std(axis=0, keepdims=True)
    ret_std = np.where(ret_std < 1e-8, 1.0, ret_std)
    ret_np  = ret_np / ret_std

    mask_np    = mask_data[:, offset + SEQ + STEPS - 1]
    gt_np      = gt_data[:,   offset + SEQ + STEPS - 1]
    feature_np = np.concatenate([emb_np, ret_np], axis=1)
    return (
        torch.tensor(feature_np, dtype=torch.float32),
        torch.tensor(mask_np,    dtype=torch.float32),
        torch.tensor(gt_np,      dtype=torch.float32),
    )


def infer(model, embedding, eod_data, mask_data, offset):
    """
    FIX-2：推理专用，不访问 offset+SEQ+STEPS-1 处的未来数据。
    当 offset=T-SEQ 时，offset+SEQ-1=T-1 是最后一天，合法。
    mask 取 min(offset+SEQ+STEPS-1, T-1) 防止越界。
    """
    T = embedding.shape[1]

    emb_np  = cs_normalize(embedding[:, offset, :].copy())
    ret_raw = eod_data[:, offset + SEQ - 1, :][:, RET_COLS]
    ret_np  = np.where(ret_raw < -1000, 0.0, ret_raw).astype(np.float32)
    ret_std = ret_np.std(axis=0, keepdims=True)
    ret_std = np.where(ret_std < 1e-8, 1.0, ret_std)
    ret_np  = ret_np / ret_std

    # ← FIX-2：mask index 不越界
    mask_idx = min(offset + SEQ + STEPS - 1, T - 1)
    mask_np  = mask_data[:, mask_idx]

    feature_np = np.concatenate([emb_np, ret_np], axis=1)
    feat_t     = torch.tensor(feature_np, dtype=torch.float32)
    mask_b     = torch.tensor(mask_np,    dtype=torch.float32)

    with torch.no_grad():
        pred = model(feat_t).squeeze(1).numpy()
    pred[mask_b.numpy() < 0.5] = -1e9
    return pred


def train_on_window(rel_encoding, rel_mask,
                    embedding, eod_data, mask_data, gt_data,
                    train_start, train_end,
                    valid_start, valid_end,
                    init_state=None,
                    max_epochs=None,
                    patience=None):
    set_seed()
    _epochs   = max_epochs if max_epochs is not None else ROLL_EPOCHS
    _patience = patience   if patience   is not None else ROLL_PATIENCE

    model = RSRModel(RSR_IN_DIM, rel_encoding, rel_mask)
    if init_state is not None:
        model.load_state_dict(init_state)

    opt       = Adam(model.parameters(), lr=LR)
    T         = embedding.shape[1]
    t_offsets = np.arange(train_start, train_end - SEQ - STEPS + 1)
    v_offsets = np.arange(valid_start, valid_end - SEQ - STEPS + 1)
    t_offsets = t_offsets[t_offsets + SEQ + STEPS - 1 < T]
    v_offsets = v_offsets[v_offsets + SEQ + STEPS - 1 < T]

    best_val   = np.inf
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    t_arr      = np.array(t_offsets)

    for _ in range(_epochs):
        model.train()
        np.random.shuffle(t_arr)
        for offset in t_arr:
            feat_t, mask_b, gt_b = make_batch(
                embedding, eod_data, mask_data, gt_data, int(offset))
            mask_b = mask_b.unsqueeze(1)
            gt_b   = gt_b.unsqueeze(1)
            if mask_b.sum() < 2:
                continue
            pred = model(feat_t)
            loss = rank_loss_fn(pred, gt_b, mask_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        val_cnt  = 0
        with torch.no_grad():
            for offset in v_offsets:
                feat_t, mask_b, gt_b = make_batch(
                    embedding, eod_data, mask_data, gt_data, int(offset))
                mask_b = mask_b.unsqueeze(1)
                gt_b   = gt_b.unsqueeze(1)
                if mask_b.sum() < 2:
                    continue
                pred      = model(feat_t)
                val_loss += rank_loss_fn(pred, gt_b, mask_b).item()
                val_cnt  += 1

        avg_val = val_loss / max(val_cnt, 1)
        if avg_val < best_val:
            best_val   = avg_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= _patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def build_date_index_map(raw_csv):
    df = pd.read_csv(raw_csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df['日期']  = pd.to_datetime(df['日期'])
    all_dates   = sorted(df['日期'].unique())
    idx_to_date = {i: all_dates[i + PAD_BEGIN]
                   for i in range(len(all_dates) - PAD_BEGIN)}
    return idx_to_date, df


def clean_ticker(code):
    return str(code).strip().lstrip("'").zfill(6)


def backtest(rel_encoding, rel_mask, tickers,
             embedding, eod_data, mask_data, gt_data,
             raw_csv, logger):
    T               = embedding.shape[1]
    idx_to_date, df = build_date_index_map(raw_csv)
    df['股票代码']  = df['股票代码'].apply(clean_ticker)
    df['日期']      = pd.to_datetime(df['日期'])
    df['开盘']      = df['开盘'].astype(float)

    week_groups = {}
    for idx in range(T):
        if idx not in idx_to_date:
            continue
        date = idx_to_date[idx]
        if not (BACKTEST_START <= date <= BACKTEST_END):
            continue
        yw = (date.isocalendar()[0], date.isocalendar()[1])
        week_groups.setdefault(yw, []).append(idx)

    prev_state   = None
    cum_return   = 0.0
    profit_weeks = 0
    results      = []

    logger.info('')
    logger.info('=' * 78)
    logger.info(f'   回测  EPOCHS={ROLL_EPOCHS}  PATIENCE={ROLL_PATIENCE}'
                f'  TRAIN_WINDOW={TRAIN_WINDOW}  cs_norm=ON  softmax=dim1')
    logger.info('=' * 78)

    for week_num, yw in enumerate(sorted(week_groups), start=1):
        indices     = week_groups[yw]
        first_idx   = min(indices)
        pred_offset = first_idx - SEQ

        if pred_offset < TRAIN_WINDOW + VALID_WINDOW:
            logger.info(f'第{week_num:02d}周 跳过（历史不足）')
            continue
        if pred_offset + SEQ + STEPS - 1 >= T:
            logger.info(f'第{week_num:02d}周 跳过（越界）')
            continue

        train_start = pred_offset - TRAIN_WINDOW - VALID_WINDOW
        train_end   = pred_offset - VALID_WINDOW
        valid_start = pred_offset - VALID_WINDOW
        valid_end   = pred_offset

        model = train_on_window(
            rel_encoding, rel_mask,
            embedding, eod_data, mask_data, gt_data,
            train_start, train_end,
            valid_start, valid_end,
            init_state=prev_state,
        )
        prev_state = copy.deepcopy(model.state_dict())

        ret          = infer(model, embedding, eod_data, mask_data, pred_offset)
        valid_ret    = ret[ret > -1e8]
        score_range  = valid_ret.max() - valid_ret.min() if len(valid_ret) > 0 else 0
        top5_pos     = np.argsort(ret)[-TOP_K:][::-1]
        top5_tickers = [tickers[i] for i in top5_pos]

        week_dates = {idx_to_date[i] for i in indices if i in idx_to_date}
        weekly_ret = 0.0
        for ticker in top5_tickers:
            s = (df[(df['股票代码'] == ticker) &
                    (df['日期'].isin(week_dates))]
                 .sort_values('日期'))
            if len(s) < 2:
                continue
            weekly_ret += ((s.iloc[-1]['开盘'] - s.iloc[0]['开盘'])
                           / s.iloc[0]['开盘'] / TOP_K)

        cum_return   += weekly_ret
        profit_weeks += int(weekly_ret >= 0)
        fd    = idx_to_date[min(indices)]
        ld    = idx_to_date[max(indices)]
        arrow = '📈' if weekly_ret >= 0 else '📉'
        logger.info(f'{arrow} 第{week_num:02d}周 '
                    f'({fd.strftime("%m.%d")}-{ld.strftime("%m.%d")})'
                    f'  | 收益率 {weekly_ret*100:+.4f}%'
                    f'  | score范围:{score_range:.4f}'
                    f'  | 股票：{"  ".join(top5_tickers)}')
        results.append({
            'week':        week_num,
            'date_start':  fd.strftime('%Y-%m-%d'),
            'date_end':    ld.strftime('%Y-%m-%d'),
            'return':      round(weekly_ret, 6),
            'score_range': round(score_range, 4),
            'tickers':     top5_tickers,
            'pred_offset': pred_offset,
        })

    total = len(results)
    logger.info('-' * 78)
    logger.info(f'💰 累计收益率     ：{cum_return*100:+.4f}%')
    if total > 0:
        logger.info(f'📊 平均每周收益率 ：{cum_return/total*100:+.4f}%')
    logger.info(f'✅ 盈利周数       ：{profit_weeks} / {total}')
    logger.info('=' * 78)
    return results


def predict_final(rel_encoding, rel_mask, tickers,
                  embedding, eod_data, mask_data, gt_data,
                  logger):
    T = embedding.shape[1]
    # ── FIX-2：使用最新完整窗口 T-SEQ（而非 T-SEQ-STEPS）────────────
    infer_offset = T - SEQ               # ← FIX-2

    train_end   = infer_offset
    train_start = train_end - TRAIN_WINDOW
    valid_start = train_end - VALID_WINDOW
    valid_end   = train_end

    logger.info(f'最终预测 infer_offset={infer_offset}  '
                f'（序列覆盖 [{infer_offset}, {infer_offset+SEQ})）')
    logger.info(f'  训练: {train_start}~{train_end}  '
                f'验证: {valid_start}~{valid_end}  '
                f'epochs={FINAL_EPOCHS}  patience={FINAL_PATIENCE}')

    model = train_on_window(
        rel_encoding, rel_mask,
        embedding, eod_data, mask_data, gt_data,
        train_start, train_end,
        valid_start, valid_end,
        init_state=None,
        max_epochs=FINAL_EPOCHS,
        patience=FINAL_PATIENCE,
    )

    ret          = infer(model, embedding, eod_data, mask_data, infer_offset)
    valid_ret    = ret[ret > -1e8]
    score_range  = valid_ret.max() - valid_ret.min() if len(valid_ret) > 0 else 0
    top5_idx     = np.argsort(ret)[-TOP_K:][::-1]
    top5_tickers = [tickers[i] for i in top5_idx]
    weights      = [1.0 / TOP_K] * TOP_K

    logger.info(f'\n预测 Top5（score范围: {score_range:.4f}）:')
    for ticker, score in zip(top5_tickers, ret[top5_idx]):
        logger.info(f'  {ticker}  score={score:.6f}')

    return top5_tickers, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--tag', type=str, default='predict')
    args = parser.parse_args()

    run_tag = f'{"backtest" if args.backtest else "predict"}_{args.tag}'
    logger, log_path = setup_logger(run_tag)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed()

    logger.info('=' * 60)
    logger.info(f'运行标签: {run_tag}')
    logger.info(f'超参数: {json.dumps(HPARAM_SNAPSHOT, ensure_ascii=False)}')
    logger.info('=' * 60)

    ticker_file = os.path.join(DATA_DIR, f'{MARKET}_tickers.txt')
    tickers     = np.genfromtxt(ticker_file, dtype=str)
    logger.info(f'股票数: {len(tickers)}')

    emb_fname = f'{MARKET}_rank_lstm_seq-{SEQ}_unit-{UNIT}.npy'
    embedding = np.load(os.path.join(PRETRAIN_DIR, emb_fname))
    logger.info(f'embedding shape: {embedding.shape}')

    eod_data, mask_data, gt_data, _ = load_eod_data(
        DATA_DIR, MARKET, tickers, STEPS
    )
    logger.info(f'交易日 T={eod_data.shape[1]}  F={eod_data.shape[2]}  STEPS={STEPS}')

    rel_filtered, rel_mask, n_keep, avg_neighbor = load_relation_data(
        os.path.join(DATA_DIR, f'{MARKET}_wiki_relation.npy')
    )
    logger.info(f'关系类型: 保留{n_keep}种  平均邻居数={avg_neighbor:.1f}')

    backtest_results = None
    if args.backtest:
        logger.info('\n[回测模式]')
        raw_csv = os.path.join(DATA_DIR, 'stock_data.csv')
        backtest_results = backtest(
            rel_filtered, rel_mask, tickers,
            embedding, eod_data, mask_data, gt_data,
            raw_csv, logger
        )

    logger.info('\n[最终预测]')
    top5_tickers, weights = predict_final(
        rel_filtered, rel_mask, tickers,
        embedding, eod_data, mask_data, gt_data,
        logger
    )

    result   = pd.DataFrame({'stock_id': top5_tickers, 'weight': weights})
    out_path = os.path.join(OUTPUT_DIR, 'result.csv')
    result.to_csv(out_path, index=False)
    logger.info(f'\n已写出 → {out_path}')
    logger.info(result.to_string(index=False))

    ws = result['weight'].sum()
    assert 0 <= ws <= 1.0, f'权重之和={ws} 不合法'
    logger.info(f'\n✓ 验证通过  股票数={len(result)}  权重之和={ws:.4f}')

    json_path, csv_path = save_summary(
        run_tag, HPARAM_SNAPSHOT,
        backtest_results,
        top5_tickers, weights,
        log_path
    )
    logger.info(f'\n📝 日志: {log_path}')
    logger.info(f'   JSON: {json_path}')
    logger.info(f'   汇总: {csv_path}')


if __name__ == '__main__':
    main()