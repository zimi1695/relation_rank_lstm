# CSI300 股票预测项目开发日志
# 基于 Temporal Relational Ranking for Stock Prediction (RSR) 论文

---

## 项目背景
- 比赛：2026年中国高校计算机大赛—大数据挑战赛
- 目标：预测沪深300中未来一周收益最大的≤5只股票
- 评分：output/result.csv，列 stock_id + weight，权重之和∈[0,1]
- 评分公式：各股票开盘价首末收益率 × 权重 的加权求和

---

## 数据概况
- stock_data.csv：2020/1/2 ~ 2026/4/17，300只沪深300股票
- 列：股票代码 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌额 换手率 涨跌幅
- 股票代码：纯数字，补零对齐6位（如 000001）
- 总交易日：1525天，processed后1496行（前29行为MA30预热期，PAD_BEGIN=29已验证）
- 回测区间：2025-10-13 ~ 2026-04-17，共26周
- 回测开始 processed_index≈1352

---

## 文件结构

project/ ├── data/ 
│ ├── stock_data.csv                # 原始数据：沪深300股价(2020-2026) 
│ ├── CSI300_tickers.txt            # 股票代码列表 
│ ├── CSI300_{ticker}_1.csv × 300   # 每只股票的特征文件 
│ ├── CSI300_wiki_relation.npy      # Wiki关系矩阵 [300,300,58] 
│ └── wikidata/ 
│ ├── hs300_wikidata.csv 
│ └── hs300_connections.json 
├── output/ 
│ └── result.csv     # 最终输出（stock_id + weight） 
├── pretrain/ 
│ ├── CSI300_rank_lstm_seq-16_unit-64.npy  # LSTM embedding [300,1496,64] 
│ ├── CSI300_lstm_weights.pt               # LSTM权重 
│ └── CSI300_rsr_best.pt                   # 旧版RSR权重（已废弃） 
├── src/ 
│ ├── preprocess.py     
│ ├── build_relation.py 
│ ├── pretrain.py       
│ └── predict.py        
├── archive/ 
│ └── train_rsr_deprecated.py     # 已归档，不再使用 
├── logs/ # 自动生成 
│ ├── summary.csv     # 所有运行汇总对比表 
│ ├── *.log           # 每次运行的完整输出 
│ └── *.json          # 每次运行的结构化摘要 
├── score_self.py     # 比赛评分脚本（不修改） 
└── README.md

---

## 已完成文件说明

### src/preprocess.py（已完成）
- 输入：data/stock_data.csv
- 输出：data/CSI300_{ticker}_1.csv × 300只股票，data/CSI300_tickers.txt
- 特征（共10列）：
  date_index, ma5, ma10, ma20, ma30, close（原始价格）,
  ret_1（1日收益率）, ret_5（5日收益率）, ret_20（20日收益率）, open（开盘价）
- 缺失值填充 -1234
- 不做全局归一化（避免信息泄漏）
- PAD_BEGIN=29 已验证：processed第0行close与stock_data第29行完全吻合

### src/build_relation.py（已完成）
- 输入：data/wikidata/hs300_wikidata.csv，data/wikidata/hs300_connections.json
- 输出：data/CSI300_wiki_relation.npy
- shape：[300, 300, 58]，dtype=int
- 覆盖率：58%，QID映射261只股票
- 关系类型：P17_P17、P31_P31等57种路径 + 1位自身关系

### src/pretrain.py（已完成）
- 输入：data/CSI300_{ticker}_1.csv（FEAT_COLS=8，col 1-8）
- 模型：RankLSTM（LSTM + MSE + 排名损失）
- 训练范围：0 ~ VALID_INDEX=1400（训练截止2026-01附近，覆盖回测起点前）
- 验证范围：1400 ~ TEST_INDEX=1470（验证截止2026-03附近）
- 窗口内归一化：每batch除以窗口起始收盘价（col 4）
- ground truth：使用收盘价（col 5）的5日收益率
- 输出：pretrain/CSI300_rank_lstm_seq-16_unit-64.npy [300, 1496, 64]
        pretrain/CSI300_lstm_weights.pt
- 最优valid loss：0.00276

### archive/train_rsr_deprecated.py（已归档 ⚠️）
- 原为独立训练RSR图卷积层的脚本
- 问题：与predict.py完全脱节，输出的CSI300_rsr_best.pt从未被使用
- 已移至archive/，不参与任何流程

### src/predict.py（已完成）
- 架构：LSTM冻结（加载embedding.npy），只训练RSR图卷积层，与论文一致
- 关系矩阵：过滤密度>5%的关系类型，保留55种，平均邻居数从87降至20
- 输入特征：cs_normalize(embedding) + cs_normalize(ret_1/ret_5/ret_20)
- 损失函数：纯rank_loss（去掉MSE，避免将所有预测拉向0）
- 训练方式：滚动窗口（TRAIN_WINDOW=252，VALID_WINDOW=63），热启动
- 最终预测：以infer_offset为终点重新训练（FINAL_EPOCHS=100），不继承回测权重
- 日志系统：自动写入logs/，含summary.csv汇总对比表
- 当前回测结果：累计收益率+26.07%，盈利周16/26

---

## 关键技术决策记录

### 决策1：关系数据选择
- 原论文：Wiki关系 + 行业关系
- 我们的选择：Wiki关系（hs300_connections.json）
- 原因：已有完整的A股Wikidata映射数据，质量高于自建行业分类

### 决策2：框架选择
- 原论文：TensorFlow 1.x
- 我们的选择：PyTorch
- 原因：环境为Python 3.9，TF1.x不兼容

### 决策3：VALID_INDEX/TEST_INDEX划分
- 原论文：756/1008（基于2013~2017年数据）
- 我们的选择：1400/1470
- 原因：LSTM需要覆盖到回测起点前，否则回测期embedding std从0.04崩塌至0.008

### 决策4：归一化方式
- 原论文：全局price_max归一化（信息泄漏）
- 我们的选择：原始价格存储 + 窗口内归一化
- 原因：消除未来数据对历史特征的污染

### 决策5：截面归一化（cs_normalize）
- 背景：A股2025年后大盘联动性增强，300只股票embedding跨截面std从0.04收缩至0.008
- 做法：每个offset取embedding[:,offset,:]后，按64维各自除以跨股票std
- 效果：score range从0.001恢复至0.1~4.0，模型排名信号恢复正常

### 决策6：关系矩阵密度过滤
- 背景：关系类型[25]覆盖18341对、[40]覆盖18151对（各占20%），softmax退化为均匀平均
- 做法：过滤覆盖率>5%的关系类型，共丢弃3种，保留55种
- 效果：平均邻居数从87降至20，图卷积区分度恢复

### 决策7：最终预测不继承回测末尾权重
- 背景：回测末尾权重的训练窗口与最终infer_offset存在时间错位
- 做法：predict_final始终以infer_offset为终点重新划定训练窗口，从随机初始化训练
- 效果：避免窗口错位导致的预测偏差

---


## Bug修复完整记录

## BUG-1：预训练权重从未生效
- 文件：predict.py / load_pretrain_state
- 现象：每次预测等同于随机初始化，回测 +0.48%
- 根因：RankLSTM 保存的键名已含 'lstm.' 前缀（如 lstm.weight_ih_l0），
        原代码再次拼接 'lstm.' 导致变成 'lstm.lstm.weight_ih_l0'，
        永远匹配不到 RSRNet 的参数字典，4个LSTM张量全部静默跳过
- 修复：去掉前缀拼接，直接用 k 匹配
- 验证：loaded=4，LSTM权重 max_abs_diff > 1e-6

---

## BUG-2：推理 offset 错位（最新窗口 embedding 未生成）
- 文件：pretrain.py / main，predict.py / predict_final + infer
- 现象：最终预测实际上在预测"数据内最后一天"而非"下一期"
- 根因：pretrain 的 embedding 生成循环上界是 T-SEQ-STEPS+1，
        导致 offset=T-SEQ（最新完整窗口）处的 embedding 为全零；
        predict_final 用 T-SEQ-STEPS 作为 infer_offset 也偏早
- 修复：
    pretrain.py：生成循环改为 range(T-SEQ+1)，单独构建 seq，
                 不经过 get_batch（避免访问越界的 gt 索引）
    predict.py：infer_offset 改为 T-SEQ
    infer()：mask_idx 用 min(offset+SEQ+STEPS-1, T-1) 防止越界
- 验证：embedding 有效范围打印为 "offset 0~1480"，
        最终预测日志显示 "infer_offset=1480（序列覆盖[1480,1496)）"

---

## BUG-3：训练采样泄漏验证标签
- 文件：pretrain.py / main
- 现象：训练 loss 偏乐观，验证结果不可信
- 根因：batch_offsets = np.arange(VALID_INDEX)，
        当 offset 接近 VALID_INDEX 时，
        gt_idx = offset+SEQ+STEPS-1 会超过 VALID_INDEX，
        即 GT 落在验证区间的样本被 shuffle 采入训练
- 修复：上界改为 VALID_INDEX-SEQ-STEPS+1，
        严格保证所有训练样本的 GT 索引 < VALID_INDEX
- 验证：n_train = 1400-16-5+1 = 1380（原来1400，减少1.4%）

---

## BUG-4：训练目标与评分公式不对齐
- 文件：pretrain.py / load_eod_data
- 现象：模型优化的方向和比赛评分方向不完全一致
- 根因：原 GT 使用 close[t+5]/close[t]-1（收盘价5日收益），
        score_self.py 评分公式为
        (open[week_end]-open[week_start])/open[week_start]（开盘价首末收益）
- 修复：GT 改为 open[t+STEPS]/open[t]-1（col 9，开盘价5日收益），
        两端都加 abs(val+1234)>1e-8 和 val>1e-8 的有效性检查
- 验证：GT统计输出从 "mean=0.0035 std=0.0621"
        变为 "mean=0.0035 std=0.0627（open-to-open）"

---

## BUG-5：A股联动导致 embedding 截面 std 收缩，rank_loss 无信号
- 文件：predict.py / make_batch
- 现象：score范围从随机初始化的0.267训练后崩至0.001，效果等于随机
- 根因：2025年后沪深300成分股联动性增强，
        300只股票的 embedding 在截面方向的 std 从0.04收缩至0.008，
        rank_loss 找不到股票间差异，梯度把所有预测推向同一值
- 修复：新增 cs_normalize()，对 embedding[:,offset,:] 按64维各自
        除以跨300只股票的 std，强制恢复截面区分度；
        ret 特征同样做截面归一化统一量纲
- 验证：归一化后所有 offset 的 std 均为1.0，
        score范围从0.001恢复至0.1~4.0，累计收益显著提升

---

## BUG-6：关系矩阵过于稠密，softmax 退化为均匀平均
- 文件：predict.py / load_relation_data
- 现象：图卷积输出 proped 对所有股票几乎相同，失去区分度
- 根因：关系类型[25]覆盖18341对、[40]覆盖18151对（各占~20%），
        平均每只股票有87个邻居，
        softmax 在87个邻居上做平均导致 proped 同质化
- 修复：过滤覆盖率 > 5%（即覆盖 300×300×5%=450对以上）的关系类型，
        共丢弃3种，保留55种
- 验证：平均邻居数从87降至20，中位数11，score范围恢复正常

---

## BUG-7：最终预测训练窗口与 infer_offset 错位
- 文件：predict.py / predict_final
- 现象：最终预测 score范围0.10，远低于回测早期的3.77
- 根因：predict_final 复用回测末尾权重（第26周），
        该模型的训练窗口终点与 infer_offset 存在时间错位；
        且回测末尾权重已在26周滚动中过拟合到特定时间模式
- 修复：predict_final 始终以 infer_offset 为终点重新划定训练窗口，
        init_state=None 从随机初始化，使用 FINAL_EPOCHS=100
- 验证：日志显示 "训练: 1228~1480  验证: 1417~1480"，
        与 infer_offset=1480 完全对齐

---

## BUG-8：Python 3.9 不支持 list|None 类型注解
- 文件：predict.py / save_summary
- 现象：启动即报 TypeError: unsupported operand type(s) for |
- 根因：list|None 是 Python 3.10+ 语法，环境为 Python 3.9
- 修复：改用 from typing import Optional，注解改为 Optional[List[dict]]
- 验证：程序正常启动

---

## FIX-5：RSR 图卷积 softmax 方向错误（语义级错误）
- 文件：predict.py / RSRModel.forward
- 现象：代码能运行，但图卷积语义与标准图注意力定义不符
- 根因：softmax(dim=0) 是按列归一化：
          w[i][j] = exp(score[i][j]) / sum_i(exp(score[i][j]))
        含义是"300只股票争夺被同一目标股票j关注的权重"
        语义错误；标准图注意力应为每只股票对其邻居分配注意力
- 修复：改为 softmax(dim=1) 按行归一化：
          w[i][j] = exp(score[i][j]) / sum_j(exp(score[i][j]))
        含义是"股票i对所有邻居的注意力权重之和=1"
- 验证（诊断脚本输出）：
    dim=1：股票0（有邻居）→ [0.333, 0.333, 0.333]，行和=1.0 ✓
           股票1（无邻居）→ [0, 1, 0]，只关注自身 ✓
    dim=0：股票1被股票0稀释为0.5，语义混乱 ✗
- 注意：修复前 +26% 是 dim=0 与旧参数碰巧配合的结果，
        不代表 dim=0 正确；修复后需重新调参

---

## Issue #1 误报澄清：preprocess.py 无全局归一化
- 被质疑代码：close_arr.max() < 1e-8
- 实际含义：有效性检查，不是归一化
           若某股票全部收盘价最大值<0.00000001，说明数据异常，跳过
- 实际存储：ma5/ma10/ma20/ma30/close/open 均为原始价格，未归一化
           ret_1/ret_5/ret_20 为收益率，天然无量纲
- 结论：无信息泄漏，无需修复

---

## 实验结果对比（所有修复完成后）

| 版本 | 累计收益 | 盈利周 | score范围 | 主要状态 |
|------|---------|--------|---------|---------|
| 初始复现 | +0.48% | 14/26 | - | BUG-1未修复，权重未加载 |
| 虚假滚动 | +63.72% | 20/26 | - | 每周相同股票，数据泄漏 |
| 端到端LSTM | -3.88% | 11/26 | 0.001 | 梯度被图卷积稀释 |
| BUG1~4修复 | +2.43% | 10/26 | 0.001 | BUG-5未修复，信号崩塌 |
| +BUG5修复（cs_norm） | +26.07% | 16/26 | 0.1~4.0 | softmax仍是dim=0 |
| +BUG6修复（关系过滤） | +26.07% | 16/26 | 0.1~4.0 | 与上一版相同 |
| +FIX2/3/4/5全部修复 | +11.32% | 12/26 | 0.1~2.7 | 代码语义全部正确，待调参 |

当前状态：所有已知 Bug 已修复，代码语义与论文对齐。
下一步：在正确代码基础上调参（TRAIN_WINDOW / ROLL_EPOCHS / LR）。

## 已知局限

### 局限1：score范围后期持续下降
根本原因：2025年A股大盘联动性增强，cs_normalize只能部分缓解，
不是代码问题，是市场特性。

### 局限2：test.csv复权口径未验证
比赛评测时才有test.csv，目前无法验证其与stock_data.csv的复权方式是否一致。
如出现分数与回测偏差较大，优先排查此项。

---
