#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从Wikidata提取QID之间的关系路径，生成connections.json（修复版）
解决414 URI Too Long错误
"""

import pandas as pd
import requests
import json
import time
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikidataConnectionBuilder:
    """构建Wikidata实体之间的连接路径"""
    
    def __init__(self, mapping_file='hs300_wikidata.csv'):
        """
        初始化
        
        Args:
            mapping_file: 股票代码到QID的映射文件
        """
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.wikidata_api = "https://www.wikidata.org/w/api.php"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikidataResearchBot/1.0 (Educational Purpose)'
        })
        
        # 加载映射并过滤有效QID
        self.load_mapping(mapping_file)
        
        # 存储连接结果
        self.connections = defaultdict(lambda: defaultdict(list))
        
    def load_mapping(self, file_path: str):
        """加载股票代码到QID的映射"""
        df = pd.read_csv(file_path)
        
        # 过滤掉NULL的QID
        valid_df = df[df['wikidata_id'] != 'NULL'].copy()
        
        self.stock_to_qid = dict(zip(valid_df['stock_code'], valid_df['wikidata_id']))
        self.qid_to_stock = dict(zip(valid_df['wikidata_id'], valid_df['stock_code']))
        self.qids = list(valid_df['wikidata_id'].unique())
        
        logger.info(f"加载了 {len(self.qids)} 个有效的Wikidata QID")
        logger.info(f"QID示例: {self.qids[:5]}")
    
    def sparql_query(self, query: str, max_retries: int = 3) -> List[Dict]:
        """
        执行SPARQL查询（使用POST避免URL长度限制）
        """
        for attempt in range(max_retries):
            try:
                # 使用POST请求，避免URI Too Long错误
                response = self.session.post(
                    self.sparql_endpoint,
                    data={'query': query},
                    headers={'Accept': 'application/json'},
                    timeout=60
                )
                
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"速率限制，等待{wait_time}秒...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return data.get('results', {}).get('bindings', [])
                
            except Exception as e:
                logger.error(f"SPARQL查询失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
        
        return []
    
    def extract_common_property_relations_batch(self, prop_id: str, prop_name: str, batch_size: int = 30):
        """
        分批提取基于共同属性的关系
        
        Args:
            prop_id: 属性ID（如P452）
            prop_name: 属性名称
            batch_size: 每批处理的QID数量
        """
        logger.info(f"处理属性: {prop_name} ({prop_id})")
        
        total_batches = (len(self.qids) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.qids))
            batch_qids = self.qids[start_idx:end_idx]
            
            logger.info(f"  批次 {batch_idx+1}/{total_batches}: 处理 {len(batch_qids)} 个QID")
            
            # 构建当前批次的QID列表
            qid_list = ' '.join([f'wd:{qid}' for qid in batch_qids])
            
            # 第一步：获取该批次QID的属性值
            query1 = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            
            SELECT DISTINCT ?qid ?value
            WHERE {{
              VALUES ?qid {{ {qid_list} }}
              ?qid wdt:{prop_id} ?value .
            }}
            """
            
            results1 = self.sparql_query(query1)
            
            # 按值分组
            value_to_qids = defaultdict(list)
            for result in results1:
                qid = result['qid']['value'].split('/')[-1]
                value = result['value']['value'].split('/')[-1]
                value_to_qids[value].append(qid)
            
            # 对于每个共同值，建立QID之间的连接
            for value, qids_with_value in value_to_qids.items():
                if len(qids_with_value) >= 2:
                    # 这些QID有共同的属性值
                    for i, qid1 in enumerate(qids_with_value):
                        for qid2 in qids_with_value[i+1:]:
                            path = [prop_id, prop_id]
                            
                            # 双向添加
                            if path not in self.connections[qid1][qid2]:
                                self.connections[qid1][qid2].append(path)
                            if path not in self.connections[qid2][qid1]:
                                self.connections[qid2][qid1].append(path)
            
            logger.info(f"  当前总连接数: {self._count_connections()}")
            time.sleep(2)  # 延迟避免速率限制
    
    def extract_common_property_relations(self):
        """
        提取基于共同属性的关系（使用分批方法）
        """
        logger.info("="*60)
        logger.info("提取基于共同属性的关系")
        logger.info("="*60)
        
        # 重要的属性列表
        important_properties = [
            ('P452', '所属行业'),
            ('P17', '所属国家'),
            ('P159', '总部位置'),
            ('P414', '交易所'),
            ('P127', '所有者'),
            ('P749', '母公司'),
            ('P361', '属于'),
        ]
        
        for prop_id, prop_name in important_properties:
            self.extract_common_property_relations_batch(prop_id, prop_name, batch_size=30)
            time.sleep(3)
        
        logger.info(f"✓ 共同属性关系提取完成")
    
    def extract_first_order_relations_batch(self, batch_size: int = 20):
        """
        分批提取一阶关系
        """
        logger.info("="*60)
        logger.info("提取一阶关系（直接连接）")
        logger.info("="*60)
        
        total_batches = (len(self.qids) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.qids))
            batch_qids = self.qids[start_idx:end_idx]
            
            logger.info(f"批次 {batch_idx+1}/{total_batches}: 处理 {len(batch_qids)} 个QID")
            
            # 对于当前批次，与所有其他QID查找连接
            source_list = ' '.join([f'wd:{qid}' for qid in batch_qids])
            target_list = ' '.join([f'wd:{qid}' for qid in self.qids])
            
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            
            SELECT DISTINCT ?source ?target ?property
            WHERE {{
              VALUES ?source {{ {source_list} }}
              VALUES ?target {{ {target_list} }}
              
              ?source ?p ?target .
              FILTER(?source != ?target)
              
              ?property wikibase:directClaim ?p .
            }}
            LIMIT 5000
            """
            
            results = self.sparql_query(query)
            
            for result in results:
                source_qid = result['source']['value'].split('/')[-1]
                target_qid = result['target']['value'].split('/')[-1]
                property_id = result['property']['value'].split('/')[-1]
                
                # 记录一阶关系
                path = [property_id]
                if path not in self.connections[source_qid][target_qid]:
                    self.connections[source_qid][target_qid].append(path)
            
            logger.info(f"  当前已找到 {self._count_connections()} 条连接")
            time.sleep(3)
        
        logger.info(f"✓ 一阶关系提取完成")
    
    def _count_connections(self) -> int:
        """计算当前的连接总数"""
        count = 0
        for source in self.connections:
            for target in self.connections[source]:
                count += len(self.connections[source][target])
        return count
    
    def save_connections(self, output_file='hs300_connections.json'):
        """
        保存连接到JSON文件
        """
        # 转换defaultdict为普通dict
        output = {}
        for source_qid in self.connections:
            output[source_qid] = {}
            for target_qid in self.connections[source_qid]:
                paths = self.connections[source_qid][target_qid]
                if paths:
                    output[source_qid][target_qid] = paths
        
        # 保存到JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 连接已保存到: {output_file}")
        
        # 保存统计信息
        stats = self._generate_statistics()
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 统计信息已保存到: {stats_file}")
    
    def _generate_statistics(self) -> Dict:
        """生成统计信息"""
        stats = {
            'total_qids': len(self.qids),
            'total_connections': 0,
            'qids_with_connections': 0,
            'first_order_count': 0,
            'second_order_count': 0,
            'property_distribution': defaultdict(int),
            'connection_degree': {}
        }
        
        for source_qid in self.connections:
            if self.connections[source_qid]:
                stats['qids_with_connections'] += 1
                degree = 0
                
                for target_qid in self.connections[source_qid]:
                    paths = self.connections[source_qid][target_qid]
                    degree += len(paths)
                    stats['total_connections'] += len(paths)
                    
                    for path in paths:
                        if len(path) == 1:
                            stats['first_order_count'] += 1
                            stats['property_distribution'][path[0]] += 1
                        elif len(path) == 2:
                            stats['second_order_count'] += 1
                            stats['property_distribution'][f"{path[0]}_{path[1]}"] += 1
                
                stats['connection_degree'][source_qid] = degree
        
        # 计算平均连接度
        if stats['qids_with_connections'] > 0:
            stats['avg_connections_per_qid'] = (
                stats['total_connections'] / stats['qids_with_connections']
            )
        
        # 转换defaultdict为普通dict
        stats['property_distribution'] = dict(stats['property_distribution'])
        
        return stats
    
    def load_and_continue(self, checkpoint_file='hs300_connections.json'):
        """从检查点文件加载已有的连接"""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for source_qid, targets in data.items():
                for target_qid, paths in targets.items():
                    self.connections[source_qid][target_qid] = paths
            
            logger.info(f"从检查点加载了 {self._count_connections()} 条连接")
            return True
        except FileNotFoundError:
            logger.info("未找到检查点文件，从头开始")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='构建Wikidata实体连接')
    parser.add_argument('--input', default='hs300_wikidata.csv', 
                       help='输入的股票-QID映射文件')
    parser.add_argument('--output', default='hs300_connections.json',
                       help='输出的连接JSON文件')
    parser.add_argument('--only-common-property', action='store_true',
                       help='只提取共同属性关系（最快，推荐）')
    parser.add_argument('--with-first-order', action='store_true',
                       help='同时提取一阶关系（较慢）')
    parser.add_argument('--batch-size', type=int, default=30,
                       help='每批处理的QID数量（默认30）')
    
    args = parser.parse_args()
    
    # 初始化构建器
    builder = WikidataConnectionBuilder(args.input)
    
    try:
        # 提取共同属性关系（推荐）
        builder.extract_common_property_relations()
        
        # 如果指定，同时提取一阶关系
        if args.with_first_order:
            logger.info("\n同时提取一阶关系...")
            builder.extract_first_order_relations_batch(batch_size=args.batch_size)
        
        # 保存结果
        builder.save_connections(args.output)
        
        # 显示统计
        stats = builder._generate_statistics()
        print("\n" + "="*60)
        print("连接提取完成！")
        print("="*60)
        print(f"总QID数: {stats['total_qids']}")
        print(f"有连接的QID数: {stats['qids_with_connections']}")
        print(f"总连接数: {stats['total_connections']}")
        print(f"一阶关系数: {stats['first_order_count']}")
        print(f"二阶关系数: {stats['second_order_count']}")
        print(f"平均每个QID的连接数: {stats.get('avg_connections_per_qid', 0):.2f}")
        
        print("\n前10个最常见的关系类型:")
        sorted_props = sorted(stats['property_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)
        for prop, count in sorted_props[:10]:
            print(f"  {prop}: {count}")
        
    except KeyboardInterrupt:
        logger.info("\n用户中断，保存当前进度...")
        builder.save_connections(args.output.replace('.json', '_checkpoint.json'))
        logger.info("进度已保存")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存当前进度
        try:
            builder.save_connections(args.output.replace('.json', '_error_backup.json'))
            logger.info("已保存错误备份")
        except:
            pass


if __name__ == '__main__':
    main()