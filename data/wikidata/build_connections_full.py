#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量提取Wikidata关系（包含一阶、二阶和共同属性关系）
"""

import pandas as pd
import requests
import json
import time
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikidataConnectionBuilder:
    """构建Wikidata实体之间的连接路径"""
    
    def __init__(self, mapping_file='hs300_wikidata.csv'):
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikidataResearchBot/1.0 (Educational Purpose)'
        })
        
        # 加载映射
        self.load_mapping(mapping_file)
        
        # 存储连接结果
        self.connections = defaultdict(lambda: defaultdict(list))
        
        # 记录开始时间
        self.start_time = time.time()
        
    def load_mapping(self, file_path: str):
        """加载股票代码到QID的映射"""
        df = pd.read_csv(file_path)
        valid_df = df[df['wikidata_id'] != 'NULL'].copy()
        
        self.stock_to_qid = dict(zip(valid_df['stock_code'], valid_df['wikidata_id']))
        self.qid_to_stock = dict(zip(valid_df['wikidata_id'], valid_df['stock_code']))
        self.qids = list(valid_df['wikidata_id'].unique())
        
        logger.info(f"加载了 {len(self.qids)} 个有效的Wikidata QID")
        
    def sparql_query(self, query: str, max_retries: int = 3) -> List[Dict]:
        """执行SPARQL查询（使用POST）"""
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.sparql_endpoint,
                    data={'query': query},
                    headers={'Accept': 'application/json'},
                    timeout=90
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
    
    def extract_common_property_relations(self, batch_size: int = 30):
        """提取基于共同属性的关系"""
        logger.info("="*60)
        logger.info("阶段1/3: 提取共同属性关系")
        logger.info("="*60)
        
        properties = [
            ('P452', '所属行业'),
            ('P17', '所属国家'),
            ('P159', '总部位置'),
            ('P414', '交易所'),
            ('P127', '所有者'),
            ('P749', '母公司'),
            ('P361', '属于'),
        ]
        
        for prop_idx, (prop_id, prop_name) in enumerate(properties):
            logger.info(f"[{prop_idx+1}/{len(properties)}] 处理属性: {prop_name} ({prop_id})")
            
            total_batches = (len(self.qids) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.qids))
                batch_qids = self.qids[start_idx:end_idx]
                
                qid_list = ' '.join([f'wd:{qid}' for qid in batch_qids])
                
                query = f"""
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                
                SELECT DISTINCT ?qid ?value
                WHERE {{
                  VALUES ?qid {{ {qid_list} }}
                  ?qid wdt:{prop_id} ?value .
                }}
                """
                
                results = self.sparql_query(query)
                
                # 按值分组
                value_to_qids = defaultdict(list)
                for result in results:
                    qid = result['qid']['value'].split('/')[-1]
                    value = result['value']['value'].split('/')[-1]
                    value_to_qids[value].append(qid)
                
                # 建立连接
                for value, qids_with_value in value_to_qids.items():
                    if len(qids_with_value) >= 2:
                        for i, qid1 in enumerate(qids_with_value):
                            for qid2 in qids_with_value[i+1:]:
                                path = [prop_id, prop_id]
                                if path not in self.connections[qid1][qid2]:
                                    self.connections[qid1][qid2].append(path)
                                if path not in self.connections[qid2][qid1]:
                                    self.connections[qid2][qid1].append(path)
                
                time.sleep(2)
            
            logger.info(f"  完成 {prop_name}，当前总连接数: {self._count_connections()}")
            time.sleep(3)
        
        logger.info(f"✓ 共同属性关系提取完成，连接数: {self._count_connections()}")
        self._print_elapsed_time()
    
    def extract_first_order_relations(self, batch_size: int = 20):
        """提取一阶直接关系"""
        logger.info("="*60)
        logger.info("阶段2/3: 提取一阶直接关系")
        logger.info("="*60)
        
        total_batches = (len(self.qids) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.qids))
            batch_qids = self.qids[start_idx:end_idx]
            
            logger.info(f"批次 {batch_idx+1}/{total_batches}: 处理 {len(batch_qids)} 个QID")
            
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
                
                path = [property_id]
                if path not in self.connections[source_qid][target_qid]:
                    self.connections[source_qid][target_qid].append(path)
            
            logger.info(f"  当前总连接数: {self._count_connections()}")
            time.sleep(3)
        
        logger.info(f"✓ 一阶关系提取完成")
        self._print_elapsed_time()
    
    def extract_second_order_relations(self, max_bridges_per_qid: int = 50):
        """提取二阶关系（通过桥接实体）"""
        logger.info("="*60)
        logger.info("阶段3/3: 提取二阶桥接关系")
        logger.info("="*60)
        logger.warning(f"这是最耗时的阶段，预计需要6-10小时...")
        
        total = len(self.qids)
        
        for idx, source_qid in enumerate(self.qids):
            logger.info(f"\n[{idx+1}/{total}] 处理 QID: {source_qid}")
            
            # 查询该QID指向的桥接实体
            query_bridges = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            
            SELECT DISTINCT ?bridge ?propertyOut
            WHERE {{
              wd:{source_qid} ?pOut ?bridge .
              FILTER(!isLiteral(?bridge))
              
              ?propertyOut wikibase:directClaim ?pOut .
            }}
            LIMIT {max_bridges_per_qid}
            """
            
            bridges = self.sparql_query(query_bridges)
            logger.info(f"  找到 {len(bridges)} 个桥接实体")
            
            # 对每个桥接实体，查找其他连接的QID
            for bridge_idx, bridge_result in enumerate(bridges):
                if (bridge_idx + 1) % 10 == 0:
                    logger.info(f"  处理桥接实体 {bridge_idx+1}/{len(bridges)}")
                
                bridge_uri = bridge_result['bridge']['value']
                bridge_id = bridge_uri.split('/')[-1]
                property_out = bridge_result['propertyOut']['value'].split('/')[-1]
                
                # 查询其他QID连接到该桥接实体
                other_qids = [q for q in self.qids if q != source_qid]
                target_list = ' '.join([f'wd:{qid}' for qid in other_qids[:100]])  # 限制100个避免太长
                
                query_targets = f"""
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                
                SELECT DISTINCT ?targetQid ?propertyIn
                WHERE {{
                  VALUES ?targetQid {{ {target_list} }}
                  ?targetQid ?pIn <{bridge_uri}> .
                  ?propertyIn wikibase:directClaim ?pIn .
                }}
                """
                
                targets = self.sparql_query(query_targets)
                
                for target_result in targets:
                    target_qid = target_result['targetQid']['value'].split('/')[-1]
                    property_in = target_result['propertyIn']['value'].split('/')[-1]
                    
                    path = [property_out, property_in]
                    if path not in self.connections[source_qid][target_qid]:
                        self.connections[source_qid][target_qid].append(path)
                
                time.sleep(1)  # 每个桥接实体延迟1秒
            
            # 每处理10个QID保存一次检查点
            if (idx + 1) % 10 == 0:
                self.save_connections('hs300_connections_checkpoint.json')
                logger.info(f"检查点已保存 ({idx+1}/{total})")
            
            logger.info(f"  当前总连接数: {self._count_connections()}")
            self._print_elapsed_time()
            time.sleep(2)
        
        logger.info(f"✓ 二阶关系提取完成")
        self._print_elapsed_time()
    
    def _count_connections(self) -> int:
        """计算当前的连接总数"""
        count = 0
        for source in self.connections:
            for target in self.connections[source]:
                count += len(self.connections[source][target])
        return count
    
    def _print_elapsed_time(self):
        """打印已用时间"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        logger.info(f"⏱️  已用时间: {hours}小时 {minutes}分钟 {seconds}秒")
    
    def save_connections(self, output_file='hs300_connections.json'):
        """保存连接到JSON文件"""
        output = {}
        for source_qid in self.connections:
            output[source_qid] = {}
            for target_qid in self.connections[source_qid]:
                paths = self.connections[source_qid][target_qid]
                if paths:
                    output[source_qid][target_qid] = paths
        
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
            'connection_degree': {},
            'extraction_time_seconds': int(time.time() - self.start_time)
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
        
        if stats['qids_with_connections'] > 0:
            stats['avg_connections_per_qid'] = (
                stats['total_connections'] / stats['qids_with_connections']
            )
        
        stats['property_distribution'] = dict(stats['property_distribution'])
        
        return stats
    
    def load_checkpoint(self, checkpoint_file):
        """从检查点加载"""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for source_qid, targets in data.items():
                for target_qid, paths in targets.items():
                    self.connections[source_qid][target_qid] = paths
            
            logger.info(f"从检查点加载了 {self._count_connections()} 条连接")
            return True
        except FileNotFoundError:
            logger.info("未找到检查点文件")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='构建Wikidata实体连接')
    parser.add_argument('--input', default='hs300_wikidata.csv', 
                       help='输入的股票-QID映射文件')
    parser.add_argument('--output', default='hs300_connections.json',
                       help='输出的连接JSON文件')
    parser.add_argument('--full-extract', action='store_true',
                       help='全量提取（包含一阶+二阶+共同属性）')
    parser.add_argument('--skip-second-order', action='store_true',
                       help='跳过二阶关系（节省时间）')
    parser.add_argument('--only-common', action='store_true',
                       help='仅提取共同属性关系（最快）')
    parser.add_argument('--max-bridges', type=int, default=50,
                       help='每个QID的最大桥接实体数（默认50）')
    parser.add_argument('--continue-from', type=str,
                       help='从检查点继续')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Wikidata关系提取程序")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    builder = WikidataConnectionBuilder(args.input)
    
    # 如果有检查点，加载
    if args.continue_from:
        builder.load_checkpoint(args.continue_from)
    
    try:
        if args.only_common:
            # 仅共同属性（20-30分钟）
            builder.extract_common_property_relations()
            
        elif args.full_extract:
            # 全量提取（7-11小时）
            logger.info("⚠️  全量提取模式，预计需要7-11小时")
            response = input("确认继续？(yes/no): ")
            if response.lower() != 'yes':
                logger.info("已取消")
                return
            
            builder.extract_common_property_relations()
            builder.extract_first_order_relations()
            builder.extract_second_order_relations(max_bridges_per_qid=args.max_bridges)
            
        else:
            # 默认：共同属性 + 一阶关系（1小时左右）
            builder.extract_common_property_relations()
            builder.extract_first_order_relations()
            
            if not args.skip_second_order:
                logger.info("\n是否继续提取二阶关系？（需要6-10小时）")
                response = input("继续？(yes/no): ")
                if response.lower() == 'yes':
                    builder.extract_second_order_relations(max_bridges_per_qid=args.max_bridges)
        
        # 保存最终结果
        builder.save_connections(args.output)
        
        # 显示统计
        stats = builder._generate_statistics()
        print("\n" + "="*60)
        print("提取完成！")
        print("="*60)
        print(f"总QID数: {stats['total_qids']}")
        print(f"有连接的QID数: {stats['qids_with_connections']}")
        print(f"总连接数: {stats['total_connections']}")
        print(f"一阶关系数: {stats['first_order_count']}")
        print(f"二阶关系数: {stats['second_order_count']}")
        print(f"平均连接数/QID: {stats.get('avg_connections_per_qid', 0):.2f}")
        print(f"总耗时: {stats['extraction_time_seconds']//3600}小时 {(stats['extraction_time_seconds']%3600)//60}分钟")
        
        print("\n前10个最常见的关系类型:")
        sorted_props = sorted(stats['property_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)
        for prop, count in sorted_props[:10]:
            print(f"  {prop}: {count}")
        
    except KeyboardInterrupt:
        logger.info("\n\n用户中断，保存当前进度...")
        builder.save_connections(args.output.replace('.json', '_interrupted.json'))
        logger.info("进度已保存，可使用 --continue-from 继续")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            builder.save_connections(args.output.replace('.json', '_error_backup.json'))
            logger.info("已保存错误备份")
        except:
            pass


if __name__ == '__main__':
    main()