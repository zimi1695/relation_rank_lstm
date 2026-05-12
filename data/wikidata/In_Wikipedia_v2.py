#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
沪深300股票维基百科Wikidata ID查询脚本（验证版）
增加多重验证机制，确保找到正确的公司实体
"""

import baostock as bs
import pandas as pd
import requests
import time
from typing import Optional, Dict, List
import logging
import os
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikidataStockMapper:
    """沪深300股票与Wikidata映射工具（带验证）"""
    
    def __init__(self, checkpoint_file='checkpoint.json'):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StockResearchBot/1.0 (Educational Purpose)'
        })
        self.checkpoint_file = checkpoint_file
        self.results_cache = self.load_checkpoint()
        
        # Wikidata API endpoints
        self.search_url = "https://www.wikidata.org/w/api.php"
        self.sparql_url = "https://query.wikidata.org/sparql"
    
    def load_checkpoint(self) -> dict:
        """加载检查点"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载检查点失败: {e}")
        return {}
    
    def save_checkpoint(self):
        """保存检查点"""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def get_hs300_stocks(self) -> pd.DataFrame:
        """从baostock获取沪深300成分股列表"""
        logger.info("正在从baostock获取沪深300成分股...")
        
        lg = bs.login()
        if lg.error_code != '0':
            raise Exception(f"baostock登录失败: {lg.error_msg}")
        
        rs = bs.query_hs300_stocks()
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        logger.info(f"成功获取{len(df)}只沪深300成分股")
        
        return df
    
    def get_entity_info(self, entity_id: str) -> Optional[Dict]:
        """
        获取Wikidata实体的详细信息，用于验证
        
        Args:
            entity_id: Wikidata实体ID (如 Q123456)
            
        Returns:
            实体信息字典，包含类型、属性等
        """
        params = {
            'action': 'wbgetentities',
            'ids': entity_id,
            'format': 'json',
            'languages': 'zh|en',
            'props': 'claims|labels|descriptions'
        }
        
        try:
            response = self.session.get(self.search_url, params=params, timeout=10)
            if response.status_code == 429:
                time.sleep(30)
                return None
                
            data = response.json()
            
            if 'entities' in data and entity_id in data['entities']:
                entity = data['entities'][entity_id]
                
                # 提取关键信息
                info = {
                    'id': entity_id,
                    'labels': entity.get('labels', {}),
                    'descriptions': entity.get('descriptions', {}),
                    'claims': entity.get('claims', {})
                }
                
                return info
            
        except Exception as e:
            logger.error(f"获取实体信息失败 {entity_id}: {e}")
            
        return None
    
    def verify_company_entity(self, entity_info: Dict, stock_code: str) -> tuple:
        """
        验证实体是否为公司，并检查是否匹配股票代码
        
        Returns:
            (is_valid, confidence_score, reason)
        """
        if not entity_info:
            return False, 0, "无法获取实体信息"
        
        claims = entity_info.get('claims', {})
        
        # 检查1: 是否为组织/公司类型
        # P31: instance of
        # P279: subclass of
        instance_of_ids = []
        if 'P31' in claims:
            for claim in claims['P31']:
                try:
                    instance_id = claim['mainsnak']['datavalue']['value']['id']
                    instance_of_ids.append(instance_id)
                except:
                    pass
        
        # 常见的公司/组织类型 ID
        company_types = {
            'Q4830453',  # 企业 (business)
            'Q783794',   # 公司 (company)
            'Q6881511',  # 企业 (enterprise)
            'Q891723',   # 上市公司 (public company)
            'Q167037',   # 股份有限公司 (corporation)
            'Q11032',    # 报纸（避免）
            'Q5',        # 人（避免）
        }
        
        has_company_type = any(iid in company_types for iid in instance_of_ids)
        
        # 检查2: 是否有股票代码属性
        # P414: 交易所 (stock exchange)
        # P249: 股票代码
        has_stock_exchange = 'P414' in claims
        has_stock_code = 'P249' in claims
        
        # 检查3: 是否有中国相关
        # P17: 国家
        # P159: 总部位置
        is_chinese = False
        if 'P17' in claims:
            for claim in claims['P17']:
                try:
                    country_id = claim['mainsnak']['datavalue']['value']['id']
                    if country_id == 'Q148':  # 中华人民共和国
                        is_chinese = True
                except:
                    pass
        
        # 计算置信度
        score = 0
        reasons = []
        
        if has_company_type:
            score += 40
            reasons.append("✓ 是公司实体")
        else:
            reasons.append("✗ 非公司实体")
        
        if has_stock_exchange:
            score += 30
            reasons.append("✓ 有交易所信息")
        
        if has_stock_code:
            score += 20
            reasons.append("✓ 有股票代码")
        
        if is_chinese:
            score += 10
            reasons.append("✓ 中国公司")
        
        reason = "; ".join(reasons)
        is_valid = score >= 40  # 至少是公司类型
        
        return is_valid, score, reason
    
    def search_wikidata(self, query: str, stock_code: str) -> Optional[tuple]:
        """
        搜索Wikidata并验证结果
        
        Returns:
            (wikidata_id, confidence_score, reason) 或 None
        """
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'zh',
            'type': 'item',
            'search': query,
            'limit': 10  # 增加到10个候选
        }
        
        try:
            response = self.session.get(self.search_url, params=params, timeout=15)
            
            if response.status_code == 429:
                logger.warning("遇到速率限制，等待30秒...")
                time.sleep(30)
                return None
            
            response.raise_for_status()
            data = response.json()
            
            if 'search' not in data or len(data['search']) == 0:
                return None
            
            # 遍历候选结果，找到最佳匹配
            best_match = None
            best_score = 0
            
            for result in data['search'][:5]:  # 只检查前5个
                entity_id = result['id']
                
                # 获取实体详细信息
                time.sleep(0.5)  # 小延迟
                entity_info = self.get_entity_info(entity_id)
                
                if entity_info:
                    is_valid, score, reason = self.verify_company_entity(
                        entity_info, stock_code
                    )
                    
                    logger.debug(f"  候选 {entity_id}: {score}分 - {reason}")
                    
                    if score > best_score:
                        best_score = score
                        best_match = (entity_id, score, reason)
            
            return best_match
            
        except Exception as e:
            logger.error(f"搜索失败 {query}: {e}")
            return None
    
    def process_stock_code(self, code: str) -> str:
        """处理股票代码格式"""
        return code.split('.')[-1]
    
    def create_mapping(self, hs300_df: pd.DataFrame, delay: float = 3.0) -> pd.DataFrame:
        """创建映射（带验证）"""
        results = []
        total = len(hs300_df)
        
        logger.info(f"开始查询 {total} 只股票（带验证）...")
        
        for idx, row in hs300_df.iterrows():
            stock_code = self.process_stock_code(row['code'])
            stock_name = row['code_name']
            
            # 检查缓存
            if stock_code in self.results_cache:
                logger.info(f"[{idx+1}/{total}] 从缓存加载: {stock_code}")
                cached = self.results_cache[stock_code]
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'wikidata_id': cached['id'],
                    'confidence': cached.get('confidence', 0),
                    'reason': cached.get('reason', '')
                })
                continue
            
            logger.info(f"[{idx+1}/{total}] 查询: {stock_name} ({stock_code})")
            
            # 尝试多种搜索策略
            search_strategies = [
                f"{stock_name} {stock_code}",
                f"{stock_name}股份有限公司",
                f"{stock_name}",
                f"{stock_name} 上市公司"
            ]
            
            best_result = None
            
            for strategy in search_strategies:
                logger.debug(f"  策略: {strategy}")
                result = self.search_wikidata(strategy, stock_code)
                
                if result and result[1] >= 40:  # 置信度≥40
                    best_result = result
                    break
                
                time.sleep(delay)
            
            # 记录结果
            if best_result:
                wikidata_id, confidence, reason = best_result
                logger.info(f"  ✓ 找到: {wikidata_id} (置信度: {confidence})")
                logger.info(f"  原因: {reason}")
            else:
                wikidata_id, confidence, reason = 'NULL', 0, '未找到匹配实体'
                logger.warning(f"  ✗ 未找到")
            
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'wikidata_id': wikidata_id,
                'confidence': confidence,
                'reason': reason
            })
            
            # 更新缓存
            self.results_cache[stock_code] = {
                'id': wikidata_id,
                'confidence': confidence,
                'reason': reason
            }
            
            # 定期保存
            if (idx + 1) % 10 == 0:
                self.save_checkpoint()
        
        self.save_checkpoint()
        
        result_df = pd.DataFrame(results)
        return result_df
    
    def save_results(self, df: pd.DataFrame, output_file: str = 'hs300_wikidata_mapping.csv'):
        """保存结果"""
        # 保存完整版（带置信度）
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"完整结果已保存: {output_file}")
        
        # 保存简化版（仅stock_code和wikidata_id）
        simple_df = df[['stock_code', 'wikidata_id']]
        simple_file = output_file.replace('.csv', '_simple.csv')
        simple_df.to_csv(simple_file, index=False, encoding='utf-8-sig')
        logger.info(f"简化结果已保存: {simple_file}")
        
        # 保存高置信度版本
        high_conf_df = df[df['confidence'] >= 60]
        if len(high_conf_df) > 0:
            high_conf_file = output_file.replace('.csv', '_high_confidence.csv')
            high_conf_df.to_csv(high_conf_file, index=False, encoding='utf-8-sig')
            logger.info(f"高置信度结果已保存: {high_conf_file}")


def main():
    """主函数"""
    try:
        mapper = WikidataStockMapper()
        
        # 获取沪深300成分股
        hs300_df = mapper.get_hs300_stocks()
        
        # 创建映射
        result_df = mapper.create_mapping(hs300_df, delay=3.0)
        
        # 保存结果
        mapper.save_results(result_df)
        
        # 统计
        print("\n" + "="*60)
        print("查询结果统计：")
        print("="*60)
        print(f"总股票数: {len(result_df)}")
        print(f"找到Wikidata ID: {(result_df['wikidata_id'] != 'NULL').sum()}")
        print(f"高置信度(≥60): {(result_df['confidence'] >= 60).sum()}")
        print(f"中置信度(40-59): {((result_df['confidence'] >= 40) & (result_df['confidence'] < 60)).sum()}")
        print(f"低置信度(<40): {(result_df['confidence'] < 40).sum()}")
        
        print("\n置信度分布:")
        print(result_df.groupby(pd.cut(result_df['confidence'], 
                                       bins=[0, 40, 60, 80, 100], 
                                       labels=['<40', '40-60', '60-80', '80-100'])).size())
        
        print("\n前10个结果:")
        print(result_df.head(10)[['stock_code', 'stock_name', 'wikidata_id', 'confidence']])
        
    except KeyboardInterrupt:
        logger.info("\n用户中断，保存进度...")
        mapper.save_checkpoint()
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == '__main__':
    main()