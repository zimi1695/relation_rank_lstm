# CSI300 股票预测 - 基于RSR模型 (Temporal Relational Ranking)

# CSI300 股票预测项目开发日志  ----  见 "relation_rank_lstm\design\design.md" 

1 这篇论文发表于 ACM TOIS（顶级信息系统期刊）解决什么问题？

股票预测的目标是预测股票未来走势，帮助投资者做出好的投资决策。传统方法基于时间序列模型；随着深度神经网络在序列数据建模上的成功，深度学习成为股票预测的热门选择。
但旧方法有两个核心缺陷：

缺陷1：目标不对
 大多数现有的深度学习方案并没有针对投资目标（即选出预期收益最高的股票）进行优化。它们通常把股票预测当成分类问题（预测涨跌趋势）或回归问题（预测价格）。

缺陷2：忽略股票之间的关系
 更重要的是，这些方法大多把每只股票视为彼此独立的个体，忽视了股票（或公司）之间丰富关系中蕴含的宝贵信息——比如两只股票同属一个行业，或者两家公司存在供应商-客户关系。

2 论文的核心方法：RSR模型

本文提出了一种新的深度学习方案，名为**关系股票排名（Relational Stock Ranking，RSR）**。
RSR在两个主要方面超越了现有方案：① 将深度学习模型定制为股票**排名**任务；② 以**时间敏感**的方式捕捉股票间的关系。 
具体来说，论文将股票预测**重新定义为一个排名任务**——目标是直接预测按期望标准（如收益率）排序的股票列表，并提出了端到端框架RSR来解决这个排名问题。

3 模型结构：三层架构（重点）

RSR包含三层：**顺序嵌入层（Sequential Embedding Layer）**、**关系嵌入层（Relational Embedding Layer）**和**预测层（Prediction Layer）**。
层级	作用	技术
第一层：顺序嵌入层	学习每只股票自身的历史价格走势规律	LSTM（长短期记忆网络）
第二层：关系嵌入层	融合股票之间的关系信息（行业、供应链等）	时间图卷积 TGC（核心创新）
第三层：预测层	综合以上信息，给每只股票打分排名	全连接层
考虑到股票市场强烈的时间动态性，历史状态是预测未来走势最有影响力的因素，因此首先用顺序嵌入层来捕获历史数据中的时序依赖关系。 
通过设计新的**时间图卷积（Temporal Graph Convolution, TGC）**，对顺序嵌入进行修正，以时间敏感的方式考虑股票关系；最终将顺序嵌入和关系嵌入拼接后输入全连接层，得到每只股票的排名分数。

4 核心创新：时间图卷积（TGC）

传统图学习技术无法捕捉股票市场的**时间演化特性**（例如两只股票之间的影响强度可能迅速变化），因为图是在某个固定时间点确定的。
TGC的创新就在于：把"图（股票关系网络）"和"时间（历史序列）"同时建模，让关系的权重随时间动态变化。

5 论文用了哪两种关系数据？

行业关系（Sector/Industry Relation）：同一行业的股票相互影响（代码中的sector_industry文件夹）
维基关系（Wiki Relation）：通过维基百科知识图谱挖掘出的公司间关系，如供应商、竞争对手等（代码中的wikidata文件夹）

6 实验效果
大量实验证明了RSR方法的优越性，它在纽约证券交易所（NYSE）和纳斯达克（NASDAQ）上分别实现了平均 **98%** 和 **71%** 的平均收益率，超越了当时最先进的股票预测方法。

## 执行流程

Step 1: 数据预处理
  输入: data/stock_data.csv
  输出: data/processed/{CSI300}_{ticker}_1.csv (每只股票一个文件)
  验证: 文件数量 == 股票数量，每文件6列，缺失值为-1234

Step 2: 构建行业关系矩阵
  输入: akshare sw_index_cons API
  输出: data/csi300_industry_relation.npy
  验证: shape == [N, N, K+1]，对角线最后一维全为1

Step 3: 预训练Sequential Embedding (RankLSTM)
  输入: data/processed/ 下所有特征文件
  输出: pretrain/CSI300_rank_lstm_seq-{l}_unit-{u}.npy
  验证: shape == [N_stocks, N_trading_days, unit]

Step 4: 训练RSR模型并预测
  输入: pretrain embedding + 关系矩阵
  输出: output/result.csv
  验证: ≤5行，stock_id为纯数字代码，weight之和∈[0,1]

## 关键设计决策

- 框架: PyTorch (原论文TF1.x，因环境不兼容改写)
- 关系数据: Wiki关系
- 数据集划分: train/valid/test = 756/252/剩余 (沿用原论文硬编码)
- 输出格式: 严格遵守score_self.py要求

## 环境信息
- OS：WSL Ubuntu
- Python：3.9.25（bigdata conda环境）
- PyTorch：2.x
- 主要依赖：torch, pandas, numpy

## 运行命令
bash
# 完整流程（首次运行）
python src/preprocess.py
python src/build_relation.py
python src/pretrain.py
python src/predict.py --tag default

# 仅生成result.csv（日常提交）
python src/predict.py --tag 提交v1

# 回测+生成result.csv
python src/predict.py --backtest --tag 回测v1

# 查看所有运行对比
cat logs/summary.csv