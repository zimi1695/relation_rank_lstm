"""
build_relation.py

输入:
  data/wikidata/hs300_wikidata.csv   stock_code → wikidata_id 映射
  data/wikidata/hs300_connections.json  QID → QID → 关系路径列表
  data/CSI300_tickers.txt            有序股票列表

输出:
  data/CSI300_wiki_relation.npy
  shape: [N, N, K+1]  dtype=int
  最后一维: 有连接的路径one-hot编码，最后一位为自身关系标记
"""

import os
import json
import numpy as np
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
WIKI_DIR    = os.path.join(DATA_DIR, 'wikidata')
MARKET      = 'CSI300'

WIKI_CSV    = os.path.join(WIKI_DIR, 'hs300_wikidata.csv')
CONN_JSON   = os.path.join(WIKI_DIR, 'hs300_connections.json')
TICKER_FILE = os.path.join(DATA_DIR, f'{MARKET}_tickers.txt')
OUT_FILE    = os.path.join(DATA_DIR, f'{MARKET}_wiki_relation.npy')


def clean_ticker(code: str) -> str:
    return str(code).strip().lstrip("'").zfill(6)


def main():
    # ── 1. 读取有序股票列表 ───────────────────────────────
    tickers = np.genfromtxt(TICKER_FILE, dtype=str)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    N = len(tickers)
    print(f'股票总数 N={N}')

    # ── 2. 读取 stock_code → QID 映射 ───────────────────
    wiki_df = pd.read_csv(WIKI_CSV, dtype=str)
    wiki_df.columns = [c.strip() for c in wiki_df.columns]
    wiki_df['stock_code'] = wiki_df['stock_code'].apply(clean_ticker)

    # 过滤掉 NULL
    wiki_df = wiki_df[wiki_df['wikidata_id'].notna()]
    wiki_df = wiki_df[wiki_df['wikidata_id'].str.upper() != 'NULL']

    qid_to_ticker_idx = {}
    for _, row in wiki_df.iterrows():
        code = row['stock_code']
        qid  = row['wikidata_id'].strip()
        if code in ticker_to_idx:
            qid_to_ticker_idx[qid] = ticker_to_idx[code]

    print(f'成功映射 QID 数: {len(qid_to_ticker_idx)}')

    # ── 3. 读取连接数据 ───────────────────────────────────
    with open(CONN_JSON, 'r') as f:
        connections = json.load(f)
    print(f'连接数据中 QID 数: {len(connections)}')

    # ── 4. 发现所有出现过的路径类型（动态，不依赖外部过滤文件）──
    occur_paths = set()
    for sou_qid, conns in connections.items():
        if sou_qid not in qid_to_ticker_idx:
            continue
        for tar_qid, paths in conns.items():
            if tar_qid not in qid_to_ticker_idx:
                continue
            for p in paths:
                occur_paths.add('_'.join(p))

    path_to_idx = {path: i for i, path in enumerate(sorted(occur_paths))}
    K = len(path_to_idx)
    print(f'有效路径类型数 K={K}  →  关系矩阵最后一维: {K+1}')

    # ── 5. 构建关系矩阵 ───────────────────────────────────
    # shape: [N, N, K+1]，最后一维第K位为自身关系标记
    relation_matrix = np.zeros([N, N, K + 1], dtype=int)

    conn_count = 0
    for sou_qid, conns in connections.items():
        if sou_qid not in qid_to_ticker_idx:
            continue
        sou_idx = qid_to_ticker_idx[sou_qid]
        for tar_qid, paths in conns.items():
            if tar_qid not in qid_to_ticker_idx:
                continue
            tar_idx = qid_to_ticker_idx[tar_qid]
            for p in paths:
                path_key = '_'.join(p)
                if path_key in path_to_idx:
                    relation_matrix[sou_idx][tar_idx][path_to_idx[path_key]] = 1
                    conn_count += 1

    print(f'填入连接数: {conn_count}')

    # ── 6. 自身关系：对角线最后一维置1（严格复现原论文）──
    for i in range(N):
        relation_matrix[i][i][-1] = 1

    print(f'关系矩阵 shape: {relation_matrix.shape}')
    print(f'连接覆盖率: {conn_count / float(N * N):.4f}')

    # ── 7. 保存 ───────────────────────────────────────────
    np.save(OUT_FILE, relation_matrix)
    print(f'已保存 → {OUT_FILE}')

    # ── 8. 快速验证 ───────────────────────────────────────
    loaded = np.load(OUT_FILE)
    assert loaded.shape == (N, N, K + 1), 'shape 验证失败'
    diag_self = sum(loaded[i][i][-1] for i in range(N))
    assert diag_self == N, f'自身关系验证失败: {diag_self} != {N}'
    print(f'验证通过 ✓  shape={loaded.shape}  对角自身关系={diag_self}/{N}')


if __name__ == '__main__':
    main()