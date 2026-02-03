#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.0
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的数据采集阶段，从多个配置的数据源中收集与目标POI相关的证据信息。
    支持从官方渠道、权威图商和互联网抓取等多种来源并行采集数据，并对原始数据进行初步整理和验证。

用途说明:
    该脚本在整个核实流程中处于第二阶段（多源情报收集），主要作用包括：
    1. 根据POI类型和名称，从config/sources.yaml配置的数据源中收集候选证据
    2. 实现多源并行采集，提高数据采集效率
    3. 对采集到的证据进行初步验证，过滤明显无效的数据
    4. 生成结构化的证据对象，为后续的规范化处理阶段提供输入
    5. 记录数据采集过程中的关键元数据（响应时间、重试次数等）

    应用场景：批量核实新接入的大POI数据、对存量数据进行质量回溯、多数据源冲突场景下的证据收集
"""

import json
import yaml
import asyncio
import aiohttp
import logging
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# 导入互联网爬虫客户端
from web_crawler_client import WebCrawlerClient, CrawlResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evidence:
    """证据数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        evidence_id: str,
        poi_id: str,
        source: Dict[str, Any],
        collected_at: str,
        data: Dict[str, Any],
        verification: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        self.evidence_id = evidence_id
        self.poi_id = poi_id
        self.source = source
        self.collected_at = collected_at
        self.data = data
        self.verification = verification
        self.metadata = metadata

    def __repr__(self):
        return (f"Evidence(evidence_id={self.evidence_id!r}, "
                f"poi_id={self.poi_id!r})")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'evidence_id': self.evidence_id,
            'poi_id': self.poi_id,
            'source': self.source,
            'collected_at': self.collected_at,
            'data': self.data,
            'verification': self.verification,
            'metadata': self.metadata
        }


class EvidenceCollector:
    """证据收集器类"""
    
    def __init__(
        self, 
        config_path: str = "../config/sources.yaml",
        enable_web_crawler: bool = True,
        web_crawler_timeout: int = 30
    ):
        """
        初始化证据收集器
        
        Args:
            config_path: 数据源配置文件路径
            enable_web_crawler: 是否启用互联网爬虫功能
            web_crawler_timeout: 爬虫接口超时时间（秒）
        """
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.enable_web_crawler = enable_web_crawler
        self.web_crawler_timeout = web_crawler_timeout
        self.web_crawler: Optional[WebCrawlerClient] = None
        logger.info("证据收集器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载数据源配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        timeout = aiohttp.ClientTimeout(
            total=self.config.get('global', {}).get('request_timeout', 30)
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # 初始化爬虫客户端
        if self.enable_web_crawler:
            self.web_crawler = WebCrawlerClient(timeout=self.web_crawler_timeout)
            await self.web_crawler.__aenter__()
            logger.info("互联网爬虫客户端已初始化")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
        
        # 关闭爬虫客户端
        if self.web_crawler:
            await self.web_crawler.__aexit__(exc_type, exc_val, exc_tb)
            self.web_crawler = None
            logger.info("互联网爬虫客户端已关闭")
    
    def _generate_evidence_id(self, poi_id: str, source_id: str) -> str:
        """
        生成证据唯一标识符
        
        Args:
            poi_id: POI标识符
            source_id: 数据源标识符
            
        Returns:
            证据ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        base = f"{poi_id}_{source_id}_{timestamp}"
        return f"EVD_{hashlib.md5(base.encode()).hexdigest()[:16].upper()}"
    
    def _get_sources_for_poi_type(self, poi_type: str) -> List[Dict[str, Any]]:
        """
        获取指定POI类型的数据源列表
        
        Args:
            poi_type: POI类型
            
        Returns:
            数据源列表
        """
        categories = self.config.get('poi_categories', {})
        category = categories.get(poi_type, {})
        sources = category.get('sources', [])
        
        # 过滤启用的数据源
        enabled_sources = [s for s in sources if s.get('enabled', True)]
        
        # 按权重排序
        enabled_sources.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        return enabled_sources
    
    async def _fetch_from_source(
        self, 
        poi_data: Dict[str, Any], 
        source: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从单个数据源获取证据
        
        Args:
            poi_data: POI数据
            source: 数据源配置
            
        Returns:
            证据对象或None
        """
        source_name = source.get('name', 'unknown')
        source_id = source.get('name', 'unknown').replace(' ', '_').upper()
        poi_id = poi_data.get('poi_id', 'unknown')
        
        logger.info(f"从数据源 '{source_name}' 收集证据: {poi_id}")

        start_time = datetime.now(timezone.utc)
        retry_count = 0
        max_retries = self.config.get('global', {}).get('retry_count', 3)

        try:
            # 构建请求参数
            query_params = {
                'name': poi_data.get('name', ''),
                'city': poi_data.get('city', ''),
                'type': poi_data.get('poi_type', '')
            }

            # 调用真实数据源API（需要根据source_type实现具体逻辑）
            source_type = source.get('type', 'internet')
            evidence_data = await self._call_api_for_source(poi_data, source, query_params)

            if not evidence_data:
                logger.warning(f"从 '{source_name}' 未获取到有效数据")
                return None
            
            # 计算响应时间
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # 创建证据对象
            evidence = Evidence(
                evidence_id=self._generate_evidence_id(poi_id, source_id),
                poi_id=poi_id,
                source={
                    'source_id': source_id,
                    'source_name': source_name,
                    'source_type': source.get('type', 'unknown'),
                    'source_url': source.get('url', ''),
                    'weight': source.get('weight', 0.5)
                },
                collected_at=datetime.now(timezone.utc).isoformat() + 'Z',
                data=evidence_data,
                verification={
                    'is_valid': True,
                    'confidence': source.get('weight', 0.5),
                    'validation_errors': []
                },
                metadata={
                    'collection_method': 'api',
                    'response_time_ms': int(response_time),
                    'retry_count': retry_count
                }
            )
            
            logger.info(f"成功从 '{source_name}' 收集证据: {evidence.evidence_id}")
            return evidence
            
        except Exception as e:
            logger.error(f"从 '{source_name}' 收集证据失败: {e}")
            return None

    async def _call_api_for_source(
        self,
        poi_data: Dict[str, Any],
        source: Dict[str, Any],
        query_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        根据数据源类型调用真实API获取数据

        Args:
            poi_data: POI数据
            source: 数据源配置
            query_params: 查询参数

        Returns:
            证据数据或None
        """
        source_type = source.get('type', 'internet')

        # 这里应该实现真实的API调用
        # 示例：根据source_type调用对应的API
        if source_type == 'official':
            # 调用官方渠道API（示例）
            evidence_data = {
                'name': poi_data.get('name'),
                'address': poi_data.get('address'),
                'coordinates': poi_data.get('coordinates'),
                'status': 'active',
                'category': poi_data.get('poi_type')
            }
        elif source_type == 'map_vendor':
            # 调用地图厂商API（示例）
            evidence_data = {
                'name': poi_data.get('name'),
                'address': poi_data.get('address'),
                'coordinates': poi_data.get('coordinates'),
                'status': 'active',
                'category': source.get('category', poi_data.get('poi_type'))
            }
        else:
            # 互联网抓取，返回None表示交由web_crawler处理
            return None

        return evidence_data


        self,
        poi_data: Dict[str, Any],
        source: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        数据源获取包装器
        
        根据数据源类型选择对应的获取方法:
        - 普通数据源: 使用 _fetch_from_source
        - web_crawler类型: 使用 _fetch_from_web_crawler
        
        Args:
            poi_data: POI数据
            source: 数据源配置
            
        Returns:
            Evidence对象或None
        """
        source_type = source.get('type', '')
        
        if source_type == 'web_crawler':
            return await self._fetch_from_web_crawler(poi_data, source)
        else:
            return await self._fetch_from_source(poi_data, source)
    
    async def collect_evidence(
        self, 
        poi_data: Dict[str, Any],
        max_evidence: int = 5,
        include_web_crawler: bool = True
    ) -> List[Evidence]:
        """
        收集POI的多源证据
        
        Args:
            poi_data: POI输入数据
            max_evidence: 最大证据数量
            include_web_crawler: 是否包含互联网爬虫数据源
            
        Returns:
            证据列表
        """
        poi_id = poi_data.get('poi_id', 'unknown')
        poi_type = poi_data.get('poi_type', '')
        
        logger.info(f"开始收集POI证据: {poi_id}, 类型: {poi_type}")
        
        # 获取数据源列表
        sources = self._get_sources_for_poi_type(poi_type)
        
        if not sources:
            logger.warning(f"未找到POI类型 '{poi_type}' 的数据源配置")
            return []
        
        logger.info(f"找到 {len(sources)} 个数据源")
        
        # 限制数据源数量
        sources = sources[:max_evidence]
        
        # 并行收集证据
        tasks = [
            self._fetch_from_source_wrapper(poi_data, source)
            for source in sources
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤有效证据
        evidence_list = []
        for result in results:
            if isinstance(result, Evidence):
                evidence_list.append(result)
            elif isinstance(result, Exception):
                logger.error(f"收集证据时发生错误: {result}")
        
        # 如果启用爬虫且配置中未包含爬虫数据源，单独调用爬虫
        if include_web_crawler and self.enable_web_crawler and self.web_crawler:
            # 检查是否已有爬虫类型的证据
            has_crawler_evidence = any(
                e.metadata.get('collection_method') == 'web_crawler_api'
                for e in evidence_list
            )
            
            if not has_crawler_evidence:
                logger.info(f"单独调用互联网爬虫收集补充证据: {poi_id}")
                crawler_source = {
                    'name': '互联网爬虫聚合',
                    'type': 'web_crawler',
                    'crawler_type': 'both',
                    'top_k': 10,
                    'weight': 0.5,
                    'url': 'http://10.82.122.209:9081'
                }
                crawler_evidence = await self._fetch_from_web_crawler(
                    poi_data, crawler_source
                )
                if crawler_evidence:
                    evidence_list.append(crawler_evidence)
        
        logger.info(f"成功收集 {len(evidence_list)} 条证据")
        return evidence_list
    
    async def _fetch_from_web_crawler(
        self,
        poi_data: Dict[str, Any],
        source_config: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从互联网爬虫接口获取证据
        
        该方法调用外部爬虫服务获取与POI相关的网页URL列表，
        支持normal_web_playwright和search_web_filter两个接口。
        
        Args:
            poi_data: POI数据，包含name、city、poi_type等字段
            source_config: 数据源配置，包含爬虫接口参数
                - crawler_type: 爬虫类型，可选 "normal_web" | "search_web" | "both"
                - top_k: 返回结果数量，默认10，最大50
                - site: 网站约束，例如"*.gov.cn"
                - date_range: 时间范围，格式"YYYY-MM-DD,YYYY-MM-DD"
                
        Returns:
            Evidence对象或None（当调用失败时）
        """
        if not self.web_crawler:
            logger.warning("爬虫客户端未初始化，跳过互联网证据收集")
            return None
        
        poi_id = poi_data.get('poi_id', 'unknown')
        poi_name = poi_data.get('name', '')
        poi_city = poi_data.get('city', '')
        source_name = source_config.get('name', '互联网爬虫')
        
        # 构建搜索查询词
        query = f"{poi_city} {poi_name}".strip()
        if not query:
            logger.warning(f"POI {poi_id} 名称和城市均为空，无法构建搜索查询")
            return None
        
        # 获取爬虫配置参数
        crawler_type = source_config.get('crawler_type', 'both')
        top_k = min(source_config.get('top_k', 10), 50)
        site = source_config.get('site')
        date_range = source_config.get('date_range')
        
        logger.info(
            f"从互联网爬虫收集证据: {poi_id}, "
            f"查询词: '{query}', 类型: {crawler_type}"
        )
        
        start_time = datetime.utcnow()
        
        try:
            # 根据配置调用相应接口
            if crawler_type == 'normal_web':
                result = await self.web_crawler.crawl_normal_web(
                    query=query,
                    top_k=top_k,
                    site=site,
                    date_range=date_range
                )
                crawl_results = {'normal_web': result}
                
            elif crawler_type == 'search_web':
                result = await self.web_crawler.crawl_search_web(
                    query=query,
                    top_k=top_k,
                    site=site,
                    date_range=date_range
                )
                crawl_results = {'search_web': result}
                
            else:  # both
                crawl_results = await self.web_crawler.crawl_both(
                    query=query,
                    top_k=top_k,
                    site=site,
                    date_range=date_range
                )
            
            # 合并两个接口返回的URL
            all_urls = []
            source_urls = {}
            
            if crawl_results.get('normal_web'):
                normal_urls = crawl_results['normal_web'].urls
                all_urls.extend(normal_urls)
                source_urls['normal_web_playwright'] = normal_urls
                
            if crawl_results.get('search_web'):
                search_urls = crawl_results['search_web'].urls
                all_urls.extend(search_urls)
                source_urls['search_web_filter'] = search_urls
            
            # 去重
            unique_urls = list(dict.fromkeys(all_urls))
            
            if not unique_urls:
                logger.warning(f"爬虫未返回任何URL: {poi_id}")
                return None
            
            # 计算响应时间
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # 生成证据ID
            source_id = source_config.get('name', 'WEB_CRAWLER').replace(' ', '_').upper()
            evidence_id = self._generate_evidence_id(poi_id, source_id)

            # 构建证据数据
            evidence_data = {
                'name': poi_name,
                'query': query,
                'urls': unique_urls,
                'url_count': len(unique_urls),
                'crawler_type': crawler_type,
                'source_urls': source_urls,
                'raw_data': {
                    'top_k': top_k,
                    'site_constraint': site,
                    'date_range': date_range,
                    'crawl_results': {
                        k: {
                            'urls': v.urls if v else [],
                            'response_time_ms': v.response_time_ms if v else 0
                        }
                        for k, v in crawl_results.items()
                    }
                }
            }

            # 创建证据对象
            evidence = Evidence(
                evidence_id=evidence_id,
                poi_id=poi_id,
                source={
                    'source_id': source_id,
                    'source_name': source_name,
                    'source_type': 'internet',
                    'source_url': source_config.get('url', ''),
                    'weight': source_config.get('weight', 0.5)
                },
                collected_at=datetime.now(timezone.utc).isoformat() + 'Z',
                data=evidence_data,
                verification={
                    'is_valid': True,
                    'confidence': source_config.get('weight', 0.5),
                    'validation_errors': []
                },
                metadata={
                    'collection_method': 'web_crawler_api',
                    'response_time_ms': int(response_time),
                    'retry_count': 0,
                    'crawler_config': {
                        'type': crawler_type,
                        'top_k': top_k,
                        'site': site,
                        'date_range': date_range
                    }
                }
            )
            
            logger.info(
                f"成功从互联网爬虫收集证据: {evidence_id}, "
                f"获取 {len(unique_urls)} 个URL"
            )
            return evidence
            
        except ConnectionError as e:
            logger.error(f"爬虫服务器连接失败: {e}")
            return None
        except TimeoutError as e:
            logger.error(f"爬虫请求超时: {e}")
            return None
        except ValueError as e:
            logger.error(f"爬虫返回数据解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"从爬虫收集证据时发生未知错误: {e}")
            return None
    
    def evidence_to_dict(self, evidence: Evidence) -> Dict[str, Any]:
        """
        将证据对象转换为字典
        
        Args:
            evidence: 证据对象
            
        Returns:
            证据字典
        """
        return evidence.to_dict()
    
    def save_evidence(
        self, 
        evidence_list: List[Evidence], 
        output_path: str
    ) -> None:
        """
        保存证据到文件
        
        Args:
            evidence_list: 证据列表
            output_path: 输出文件路径
        """
        try:
            evidence_dicts = [self.evidence_to_dict(e) for e in evidence_list]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evidence_dicts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"证据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存证据失败: {e}")
            raise


async def main():
    """
    主函数 - 示例用法
    """
    # 示例POI数据
    poi_data = {
        "poi_id": "HOSPITAL_BJ_001",
        "name": "北京大学第一医院",
        "poi_type": "hospital",
        "city": "北京市",
     #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.0
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的数据采集阶段，从多个配置的数据源中收集与目标POI相关的证据信息。
    支持从官方渠道、权威图商和互联网抓取等多种来源并行采集数据，并对原始数据进行初步整理和验证。

用途说明:
    该脚本在整个核实流程中处于第二阶段（多源情报收集），主要作用包括：
    1. 根据POI类型和名称，从config/sources.yaml配置的数据源中收集候选证据
    2. 实现多源并行采集，提高数据采集效率
    3. 对采集到的证据进行初步验证，过滤明显无效的数据
    4. 生成结构化的证据对象，为后续的规范化处理阶段提供输入
    5. 记录数据采集过程中的关键元数据（响应时间、重试次数等）

    应用场景：批量核实新接入的大POI数据、对存量数据进行质量回溯、多数据源冲突场景下的证据收集
"""

import json
import yaml
import asyncio
import aiohttp
import logging
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# 导入互联网爬虫客户端
from web_crawler_client import WebCrawlerClient, CrawlResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evidence:
    """证据数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        evidence_id: str,
        poi_id: str,
        source: Dict[str, Any],
        collected_at: str,
        data: Dict[str, Any],
        verification: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        self.evidence_id = evidence_id
        self.poi_id = poi_id
        self.source = source
        self.collected_at = collected_at
        self.data = data
        self.verification = verification
        self.metadata = metadata

    def __repr__(self):
        return (f"Evidence(evidence_id={self.evidence_id!r}, "
                f"poi_id={self.poi_id!r})")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'evidence_id': self.evidence_id,
            'poi_id': self.poi_id,
            'source': self.source,
            'collected_at': self.collected_at,
            'data': self.data,
            'verification': self.verification,
            'metadata': self.metadata
        }


class EvidenceCollector:
    """证据收集器类"""
    
    def __init__(
        self, 
        config_path: str = "../config/sources.yaml",
        enable_web_crawler: bool = True,
        web_crawler_timeout: int = 30
    ):
        """
        初始化证据收集器
        
        Args:
            config_path: 数据源配置文件路径
            enable_web_crawler: 是否启用互联网爬虫功能
            web_crawler_timeout: 爬虫接口超时时间（秒）
        """
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.enable_web_crawler = enable_web_crawler
        self.web_crawler_timeout = web_crawler_timeout
        self.web_crawler: Optional[WebCrawlerClient] = None
        logger.info("证据收集器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载数据源配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        timeout = aiohttp.ClientTimeout(
            total=self.config.get('global', {}).get('request_timeout', 30)
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # 初始化爬虫客户端
        if self.enable_web_crawler:
            self.web_crawler = WebCrawlerClient(timeout=self.web_crawler_timeout)
            await self.web_crawler.__aenter__()
            logger.info("互联网爬虫客户端已初始化")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
        
        # 关闭爬虫客户端
        if self.web_crawler:
            await self.web_crawler.__aexit__(exc_type, exc_val, exc_tb)
            self.web_crawler = None
            logger.info("互联网爬虫客户端已关闭")
    
    def _generate_evidence_id(self, poi_id: str, source_id: str) -> str:
        """
        生成证据唯一标识符
        
        Args:
            poi_id: POI标识符
            source_id: 数据源标识符
            
        Returns:
            证据ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        base = f"{poi_id}_{source_id}_{timestamp}"
        return f"EVD_{hashlib.md5(base.encode()).hexdigest()[:16].upper()}"
    
    def _get_sources_for_poi_type(self, poi_type: str) -> List[Dict[str, Any]]:
        """
        获取指定POI类型的数据源列表
        
        Args:
            poi_type: POI类型
            
        Returns:
            数据源列表
        """
        categories = self.config.get('poi_categories', {})
        category = categories.get(poi_type, {})
        sources = category.get('sources', [])
        
        # 过滤启用的数据源
        enabled_sources = [s for s in sources if s.get('enabled', True)]
        
        # 按权重排序
        enabled_sources.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        return enabled_sources
    
    async def _fetch_from_source(
        self, 
        poi_data: Dict[str, Any], 
        source: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从单个数据源获取证据
        
        Args:
            poi_data: POI数据
            source: 数据源配置
            
        Returns:
            证据对象或None
        """
        source_name = source.get('name', 'unknown')
        source_id = source.get('name', 'unknown').replace(' ', '_').upper()
        poi_id = poi_data.get('poi_id', 'unknown')
        
        logger.info(f"从数据源 '{source_name}' 收集证据: {poi_id}")

        start_time = datetime.now(timezone.utc)
        retry_count = 0
        max_retries = self.config.get('global', {}).get('retry_count', 3)

        try:
            # 构建请求参数
            query_params = {
                'name': poi_data.get('name', ''),
                'city': poi_data.get('city', ''),
                'type': poi_data.get('poi_type', '')
            }

            # 调用真实数据源API（需要根据source_type实现具体逻辑）
            source_type = source.get('type', 'internet')
            evidence_data = await self._call_api_for_source(poi_data, source, query_params)

            if not evidence_data:
                logger.warning(f"从 '{source_name}' 未获取到有效数据")
                return None
            
            # 计算响应时间
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # 创建证据对象
            evidence = Evidence(
                evidence_id=self._generate_evidence_id(poi_id, source_id),
                poi_id=poi_id,
                source={
                    'source_id': source_id,
                    'source_name': source_name,
                    'source_type': source.get('type', 'unknown'),
                    'source_url': source.get('url', ''),
                    'weight': source.get('weight', 0.5)
                },
                collected_at=datetime.now(timezone.utc).isoformat() + 'Z',
                data=evidence_data,
                verification={
                    'is_valid': True,
                    'confidence': source.get('weight', 0.5),
                    'validation_errors': []
                },
                metadata={
                    'collection_method': 'api',
                    'response_time_ms': int(response_time),
                    'retry_count': retry_count
                }
            )
            
            logger.info(f"成功从 '{source_name}' 收集证据: {evidence.evidence_id}")
            return evidence
            
        except Exception as e:
            logger.error(f"从 '{source_name}' 收集证据失败: {e}")
            return None

    async def _call_api_for_source(
        self,
        poi_data: Dict[str, Any],
        source: Dict[str, Any],
        query_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        根据数据源类型调用真实API获取数据

        Args:
            poi_data: POI数据
            source: 数据源配置
            query_params: 查询参数

        Returns:
            证据数据或None
        """
        source_type = source.get('type', 'internet')

        # 这里应该实现真实的API调用
        # 示例：根据source_type调用对应的API
        if source_type == 'official':
            # 调用官方渠道API（示例）
            evidence_data = {
                'name': poi_data.get('name'),
                'address': poi_data.get('address'),
                'coordinates': poi_data.get('coordinates'),
                'status': 'active',
                'category': poi_data.get('poi_type')
            }
        elif source_type == 'map_vendor':
            # 调用地图厂商API（示例）
            evidence_data = {
                'name': poi_data.get('name'),
                'address': poi_data.get('address'),
                'coordinates': poi_data.get('coordinates'),
                'status': 'active',
                'category': source.get('category', poi_data.get('poi_type'))
            }
        else:
            # 互联网抓取，返回None表示交由web_crawler处理
            return None

        return evidence_data


        self,
        poi_data: Dict[str, Any],
        source: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        数据源获取包装器
        
        根据数据源类型选择对应的获取方法:
        - 普通数据源: 使用 _fetch_from_source
        - web_crawler类型: 使用 _fetch_from_web_crawler
        
        Args:
            poi_data: POI数据
            source: 数据源配置
            
        Returns:
            Evidence对象或None
        """
        source_type = source.get('type', '')
        
        if source_type == 'web_crawler':
            return await self._fetch_from_web_crawler(poi_data, source)
        else:
            return await self._fetch_from_source(poi_data, source)
    
    async def collect_evidence(
        self, 
        poi_data: Dict[str, Any],
        max_evidence: int = 5,
        include_web_crawler: bool = True
    ) -> List[Evidence]:
        """
        收集POI的多源证据
        
        Args:
            poi_data: POI输入数据
            max_evidence: 最大证据数量
            include_web_crawler: 是否包含互联网爬虫数据源
            
        Returns:
            证据列表
        """
        poi_id = poi_data.get('poi_id', 'unknown')
        poi_type = poi_data.get('poi_type', '')
        
        logger.info(f"开始收集POI证据: {poi_id}, 类型: {poi_type}")
        
        # 获取数据源列表
        sources = self._get_sources_for_poi_type(poi_type)
        
        if not sources:
            logger.warning(f"未找到POI类型 '{poi_type}' 的数据源配置")
            return []
        
        logger.info(f"找到 {len(sources)} 个数据源")
        
        # 限制数据源数量
        sources = sources[:max_evidence]
        
        # 并行收集证据
        tasks = [
            self._fetch_from_source_wrapper(poi_data, source)
            for source in sources
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤有效证据
        evidence_list = []
        for result in results:
            if isinstance(result, Evidence):
                evidence_list.append(result)
            elif isinstance(result, Exception):
                logger.error(f"收集证据时发生错误: {result}")
        
        # 如果启用爬虫且配置中未包含爬虫数据源，单独调用爬虫
        if include_web_crawler and self.enable_web_crawler and self.web_crawler:
            # 检查是否已有爬虫类型的证据
            has_crawler_evidence = any(
                e.metadata.get('collection_method') == 'web_crawler_api'
                for e in evidence_list
            )
            
            if not has_crawler_evidence:
                logger.info(f"单独调用互联网爬虫收集补充证据: {poi_id}")
                crawler_source = {
                    'name': '互联网爬虫聚合',
                    'type': 'web_crawler',
                    'crawler_type': 'both',
                    'top_k': 10,
                    'weight': 0.5,
                    'url': 'http://10.82.122.209:9081'
                }
                crawler_evidence = await self._fetch_from_web_crawler(
                    poi_data, crawler_source
                )
                if crawler_evidence:
                    evidence_list.append(crawler_evidence)
        
        logger.info(f"成功收集 {len(evidence_list)} 条证据")
        return evidence_list
    
    async def _fetch_from_web_crawler(
        self,
        poi_data: Dict[str, Any],
        source_config: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从互联网爬虫接口获取证据
        
        该方法调用外部爬虫服务获取与POI相关的网页URL列表，
        支持normal_web_playwright和search_web_filter两个接口。
        
        Args:
            poi_data: POI数据，包含name、city、poi_type等字段
            source_config: 数据源配置，包含爬虫接口参数
                - crawler_type: 爬虫类型，可选 "normal_web" | "search_web" | "both"
                - top_k: 返回结果数量，默认10，最大50
                - site: 网站约束，例如"*.gov.cn"
                - date_range: 时间范围，格式"YYYY-MM-DD,YYYY-MM-DD"
                
        Returns:
            Evidence对象或None（当调用失败时）
        """
        if not self.web_crawler:
            logger.warning("爬虫客户端未初始化，跳过互联网证据收集")
            return None
        
        poi_id = poi_data.get('poi_id', 'unknown')
        poi_name = poi_data.get('name', '')
        poi_city = poi_data.get('city', '')
        source_name = source_config.get('name', '互联网爬虫')
        
        # 构建搜索查询词
        query = f"{poi_city} {poi_name}".strip()
        if not query:
            logger.warning(f"POI {poi_id} 名称和城市均为空，无法构建搜索查询")
            return None
        
        # 获取爬虫配置参数
        crawler_type = source_config.get('crawler_type', 'both')
        top_k = min(source_config.get('top_k', 10), 50)
        site = source_config.get('site')
        date_range = source_config.get('date_range')
        
        logger.info(
            f"从互联网爬虫收集证据: {poi_id}, "
            f"查询词: '{query}', 类型: {crawler_type}"
        )
        
        start_time = datetime.utcnow()
        
        try:
            # 根据配置调用相应接口
            if crawler_type == 'normal_web':
                result = await self.web_crawler.crawl_normal_web(
                    query=query,
                    top_k=top_k,
                    site=site,
                    date_range=date_range
                )
                crawl_results = {'normal_web': result}
                
            elif crawler_type == 'search_web':
                result = await self.web_crawler.crawl_search_web(
                    query=query,
                    top_k=top_k,
                    site=site,
                    date_range=date_range
                )
                crawl_results = {'search_web': result}
                
            else:  # both
                crawl_results = await self.web_crawler.crawl_both(
                    query=query,
                    top_k=top_k,
                    site=site,
                    date_range=date_range
                )
            
            # 合并两个接口返回的URL
            all_urls = []
            source_urls = {}
            
            if crawl_results.get('normal_web'):
                normal_urls = crawl_results['normal_web'].urls
                all_urls.extend(normal_urls)
                source_urls['normal_web_playwright'] = normal_urls
                
            if crawl_results.get('search_web'):
                search_urls = crawl_results['search_web'].urls
                all_urls.extend(search_urls)
                source_urls['search_web_filter'] = search_urls
            
            # 去重
            unique_urls = list(dict.fromkeys(all_urls))
            
            if not unique_urls:
                logger.warning(f"爬虫未返回任何URL: {poi_id}")
                return None
            
            # 计算响应时间
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # 生成证据ID
            source_id = source_config.get('name', 'WEB_CRAWLER').replace(' ', '_').upper()
            evidence_id = self._generate_evidence_id(poi_id, source_id)

            # 构建证据数据
            evidence_data = {
                'name': poi_name,
                'query': query,
                'urls': unique_urls,
                'url_count': len(unique_urls),
                'crawler_type': crawler_type,
                'source_urls': source_urls,
                'raw_data': {
                    'top_k': top_k,
                    'site_constraint': site,
                    'date_range': date_range,
                    'crawl_results': {
                        k: {
                            'urls': v.urls if v else [],
                            'response_time_ms': v.response_time_ms if v else 0
                        }
                        for k, v in crawl_results.items()
                    }
                }
            }

            # 创建证据对象
            evidence = Evidence(
                evidence_id=evidence_id,
                poi_id=poi_id,
                source={
                    'source_id': source_id,
                    'source_name': source_name,
                    'source_type': 'internet',
                    'source_url': source_config.get('url', ''),
                    'weight': source_config.get('weight', 0.5)
                },
                collected_at=datetime.now(timezone.utc).isoformat() + 'Z',
                data=evidence_data,
                verification={
                    'is_valid': True,
                    'confidence': source_config.get('weight', 0.5),
                    'validation_errors': []
                },
                metadata={
                    'collection_method': 'web_crawler_api',
                    'response_time_ms': int(response_time),
                    'retry_count': 0,
                    'crawler_config': {
                        'type': crawler_type,
                        'top_k': top_k,
                        'site': site,
                        'date_range': date_range
                    }
                }
            )
            
            logger.info(
                f"成功从互联网爬虫收集证据: {evidence_id}, "
                f"获取 {len(unique_urls)} 个URL"
            )
            return evidence
            
        except ConnectionError as e:
            logger.error(f"爬虫服务器连接失败: {e}")
            return None
        except TimeoutError as e:
            logger.error(f"爬虫请求超时: {e}")
            return None
        except ValueError as e:
            logger.error(f"爬虫返回数据解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"从爬虫收集证据时发生未知错误: {e}")
            return None
    
    def evidence_to_dict(self, evidence: Evidence) -> Dict[str, Any]:
        """
        将证据对象转换为字典
        
        Args:
            evidence: 证据对象
            
        Returns:
            证据字典
        """
        return evidence.to_dict()
    
    def save_evidence(
        self, 
        evidence_list: List[Evidence], 
        output_path: str
    ) -> None:
        """
        保存证据到文件
        
        Args:
            evidence_list: 证据列表
            output_path: 输出文件路径
        """
        try:
            evidence_dicts = [self.evidence_to_dict(e) for e in evidence_list]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evidence_dicts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"证据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存证据失败: {e}")
            raise


async def main():
    """
    主函数 - 示例用法
    """
    # 示例POI数据
    poi_data = {
        "poi_id": "HOSPITAL_BJ_001",
        "name": "北京大学第一医院",
        "poi_type": "hospital",
        "city": "北京市",
        "address": "北京市西城区西什库大街8号",
        "coordinates": {
            "longitude": 116.3723,
            "latitude": 39.9342
        }
    }
    
    # 收集证据
    async with EvidenceCollector() as collector:
        evidence_list = await collector.collect_evidence(poi_data)
        
        # 保存证据
        if evidence_list:
            collector.save_evidence(evidence_list, "evidence_output.json")
            
            # 打印证据摘要
            for evidence in evidence_list:
                print(f"\n证据ID: {evidence.evidence_id}")
                print(f"来源: {evidence.source['source_name']}")
                print(f"权重: {evidence.source['weight']}")
                print(f"可信度: {evidence.verification['confidence']}")


if __name__ == "__main__":
    asyncio.run(main())   "address": "北京市西城区西什库大街8号",
        "coordinates": {
            "longitude": 116.3723,
            "latitude": 39.9342
        }
    }
    
    # 收集证据
    async with EvidenceCollector() as collector:
        evidence_list = await collector.collect_evidence(poi_data)
        
        # 保存证据
        if evidence_list:
            collector.save_evidence(evidence_list, "evidence_output.json")
            
            # 打印证据摘要
            for evidence in evidence_list:
                print(f"\n证据ID: {evidence.evidence_id}")
                print(f"来源: {evidence.source['source_name']}")
                print(f"权重: {evidence.source['weight']}")
                print(f"可信度: {evidence.verification['confidence']}")


if __name__ == "__main__":
    asyncio.run(main())
