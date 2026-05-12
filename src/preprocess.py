"""
preprocess.py

输出每只股票文件格式（10列，无表头）：
  date_index, ma5, ma10, ma20, ma30, close,
  ret_1, ret_5, ret_20, open
  缺失交易日填充 -1234
"""

import os
import numpy as np
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_FILE  = os.path.join(DATA_DIR, 'stock_data.csv')
MARKET    = 'CSI300'
PAD_BEGIN = 29


def clean_ticker(code: str) -> str:
    return str(code).strip().lstrip("'").zfill(6)


def main():
    df = pd.read_csv(RAW_FILE, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df['股票代码'] = df['股票代码'].apply(clean_ticker)
    df['日期']    = pd.to_datetime(df['日期'])
    df['收盘']    = df['收盘'].astype(float)
    df['开盘']    = df['开盘'].astype(float)   # ← 新增

    all_dates   = sorted(df['日期'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_dates     = len(all_dates)
    n_out_rows  = n_dates - PAD_BEGIN
    print(f'总交易日数: {n_dates}  →  每股输出行数: {n_out_rows}')

    tickers = sorted(df['股票代码'].unique())
    with open(os.path.join(DATA_DIR, f'{MARKET}_tickers.txt'), 'w') as f:
        f.write('\n'.join(tickers) + '\n')
    print(f'共 {len(tickers)} 只股票')

    skipped = 0
    for ticker in tickers:
        stock_df = (df[df['股票代码'] == ticker]
                    .sort_values('日期')
                    .reset_index(drop=True))
        close_arr    = stock_df['收盘'].values
        open_arr     = stock_df['开盘'].values   # ← 新增
        date_idx_arr = stock_df['日期'].map(date_to_idx).values

        if len(close_arr) == 0 or close_arr.max() < 1e-8:
            skipped += 1
            continue

        # 10列：date_index, ma5, ma10, ma20, ma30, close, ret_1, ret_5, ret_20, open
        features = np.full((n_out_rows, 10), -1234.0, dtype=np.float64)
        for row_i in range(n_out_rows):
            features[row_i][0] = row_i

        for row in range(len(stock_df)):
            cur_global_idx = int(date_idx_arr[row])
            if cur_global_idx < PAD_BEGIN:
                continue

            sums   = np.zeros(4)
            counts = np.zeros(4, dtype=int)
            for offset in range(min(30, row + 1)):
                gap = cur_global_idx - int(date_idx_arr[row - offset])
                for w_i, w in enumerate((5, 10, 20, 30)):
                    if gap < w:
                        sums[w_i]   += close_arr[row - offset]
                        counts[w_i] += 1
            mas = np.where(counts > 0, sums / counts, 0.0)

            def ret(lookback):
                if row < lookback:
                    return -1234.0
                prev = close_arr[row - lookback]
                if prev < 1e-8:
                    return -1234.0
                return (close_arr[row] - prev) / prev

            out_row = cur_global_idx - PAD_BEGIN
            features[out_row][1:5] = mas
            features[out_row][5]   = close_arr[row]
            features[out_row][6]   = ret(1)
            features[out_row][7]   = ret(5)
            features[out_row][8]   = ret(20)
            features[out_row][9]   = open_arr[row]   # ← 新增

        np.savetxt(
            os.path.join(DATA_DIR, f'{MARKET}_{ticker}_1.csv'),
            features, fmt='%.6f', delimiter=','
        )

    print(f'完成。成功 {len(tickers) - skipped} 只，跳过 {skipped} 只')
    print(f'特征维度: 9 (ma5/ma10/ma20/ma30/close/ret_1/ret_5/ret_20/open)')


if __name__ == '__main__':
    main()