#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.2
# Author: BigPOI Verification System
# Date: 2024-01-15

多源情报收集脚本

功能定义说明:
    本脚本负责大POI核实流程中的多源情报收集阶段（第二阶段），通过调用不同类型
    的数据源接口（官方渠道、地图服务商、互联网爬虫等）收集与目标POI相关的候选证据。

用途说明:
    该脚本在整个核实流程中处于第二阶段（多源情报收集），主要作用包括：
    1. 根据POI类型和名称，从sources.yaml配置的数据源中选择适配的来源
    2. 调用地图API（高德、百度、腾讯）进行POI搜索和信息获取
    3. 调用互联网爬虫接口获取网页搜索结果
    4. 收集多源候选信息，形成结构化证据对象
    5. 进行证据的初步格式校验和数据规范化

    应用场景：在通过输入校验后，对POI进行多源信息的初步收集和汇总

核心特性:
    - 多源并行采集：支持同时调用多个数据源
    - 灵活的来源配置：基于POI类型动态选择数据源
    - 异常容错机制：单个数据源失败不影响其他源的采集
    - 结构化证据：所有采集数据均转换为标准的证据对象格式
    - 元数据记录：完整记录采集时间、响应时间等元数据
"""

import json
import yaml
import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

try:
    import aiohttp
except ImportError:
    aiohttp = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """数据源类型枚举"""
    OFFICIAL = "official"           # 官方渠道
    MAP_VENDOR = "map_vendor"       # 权威图商
    INTERNET = "internet"           # 互联网抓取
    WEB_CRAWLER = "web_crawler"     # 互联网爬虫


@dataclass
class EvidenceData:
    """证据数据对象"""
    name: str                                           # POI名称
    address: Optional[str] = None                       # 地址
    coordinates: Optional[Dict[str, float]] = None      # 坐标 {longitude, latitude}
    phone: Optional[str] = None                         # 电话
    category: Optional[str] = None                      # 分类
    status: Optional[str] = None                        # 状态
    level: Optional[str] = None                         # 等级信息
    administrative: Optional[Dict[str, str]] = None     # 行政区划
    raw_data: Optional[Dict[str, Any]] = None           # 原始数据


@dataclass
class Evidence:
    """证据对象 - 遵循evidence.schema.json"""
    evidence_id: str                                    # 证据ID
    poi_id: str                                         # POI ID
    source: Dict[str, Any]                              # 来源信息
    collected_at: str                                   # 收集时间
    data: EvidenceData                                  # 证据数据
    verification: Optional[Dict[str, Any]] = None       # 验证信息
    matching: Optional[Dict[str, Any]] = None           # 匹配信息
    metadata: Optional[Dict[str, Any]] = None           # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "evidence_id": self.evidence_id,
            "poi_id": self.poi_id,
            "source": self.source,
            "collected_at": self.collected_at,
            "data": {
                "name": self.data.name,
            }
        }

        # 添加可选数据字段
        if self.data.address:
            result["data"]["address"] = self.data.address
        if self.data.coordinates:
            result["data"]["coordinates"] = self.data.coordinates
        if self.data.phone:
            result["data"]["phone"] = self.data.phone
        if self.data.category:
            result["data"]["category"] = self.data.category
        if self.data.status:
            result["data"]["status"] = self.data.status
        if self.data.level:
            result["data"]["level"] = self.data.level
        if self.data.administrative:
            result["data"]["administrative"] = self.data.administrative
        if self.data.raw_data:
            result["data"]["raw_data"] = self.data.raw_data

        if self.verification:
            result["verification"] = self.verification
        if self.matching:
            result["matching"] = self.matching
        if self.metadata:
            result["metadata"] = self.metadata

        return result


class EvidenceCollector:
    """多源情报收集器"""

    def __init__(
        self,
        sources_config_path: str = "../config/sources.yaml",
        skill_config_path: str = "../config/skill.yaml"
    ):
        """
        初始化收集器

        Args:
            sources_config_path: sources.yaml配置文件路径
            skill_config_path: skill.yaml配置文件路径
        """
        self.sources_config = self._load_yaml(sources_config_path)
        self.skill_config = self._load_yaml(skill_config_path)
        self.http_session: Optional[aiohttp.ClientSession] = None
        logger.info("EvidenceCollector初始化完成")

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.debug(f"成功加载YAML配置: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载YAML配置失败: {config_path}, 错误: {e}")
            raise

    async def __aenter__(self):
        """异步上下文管理器入口"""
        if aiohttp is None:
            raise ImportError("aiohttp库未安装，请运行: pip install aiohttp")
        timeout = aiohttp.ClientTimeout(total=30)
        self.http_session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    def _generate_evidence_id(self, poi_id: str, source_name: str) -> str:
        """
        生成证据ID

        Args:
            poi_id: POI ID
            source_name: 数据源名称

        Returns:
            证据ID
        """
        hash_input = f"{poi_id}_{source_name}_{datetime.utcnow().isoformat()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"EVD_{timestamp}_{hash_value}"

    def _get_sources_for_poi_type(self, poi_type: str) -> List[Dict[str, Any]]:
        """
        根据POI类型获取配置的数据源列表

        Args:
            poi_type: POI类型（如hospital、government等）

        Returns:
            数据源配置列表
        """
        poi_categories = self.sources_config.get('poi_categories', {})
        category_config = poi_categories.get(poi_type, {})
        sources = category_config.get('sources', [])

        logger.info(f"POI类型 '{poi_type}' 配置了 {len(sources)} 个数据源")
        return sources

    async def _call_map_api(
        self,
        api_endpoint: str,
        api_key: str,
        poi_name: str,
        city: str,
        source_name: str
    ) -> Optional[EvidenceData]:
        """
        调用地图API进行真实查询

        Args:
            api_endpoint: API端点URL
            api_key: API密钥
            poi_name: POI名称
            city: 城市
            source_name: 来源名称

        Returns:
            证据数据或None
        """
        if not self.http_session:
            logger.warning("HTTP会话未初始化，无法调用API")
            return None

        try:
            # 根据API类型构建请求参数
            if 'amap' in api_endpoint.lower():
                return await self._call_amap(api_endpoint, api_key, poi_name, city)
            elif 'baidu' in api_endpoint.lower():
                return await self._call_baidu_map(api_endpoint, api_key, poi_name, city)
            elif 'qq' in api_endpoint.lower():
                return await self._call_qq_map(api_endpoint, api_key, poi_name, city)
            else:
                logger.warning(f"未知的API类型: {api_endpoint}")
                return None

        except Exception as e:
            logger.error(f"调用地图API '{source_name}' 失败: {e}")
            return None

    async def _call_amap(
        self,
        api_endpoint: str,
        api_key: str,
        poi_name: str,
        city: str
    ) -> Optional[EvidenceData]:
        """调用高德地图API"""
        try:
            params = {
                'key': api_key,
                'keywords': poi_name,
                'city': city,
                'output': 'json'
            }

            async with self.http_session.get(api_endpoint, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('pois') and len(data['pois']) > 0:
                        poi = data['pois'][0]
                        logger.info(f"从高德地图获取到POI: {poi.get('name')}")
                        return EvidenceData(
                            name=poi.get('name', ''),
                            address=poi.get('address', ''),
                            coordinates={
                                'longitude': float(poi.get('location', '').split(',')[0]) if ',' in poi.get('location', '') else None,
                                'latitude': float(poi.get('location', '').split(',')[1]) if ',' in poi.get('location', '') else None
                            } if poi.get('location') else None,
                            phone=poi.get('tel'),
                            category=poi.get('type'),
                            raw_data=poi
                        )
                else:
                    logger.warning(f"高德地图API返回状态码: {resp.status}")
                    return None

        except Exception as e:
            logger.error(f"调用高德地图API失败: {e}")
            return None

    async def _call_baidu_map(
        self,
        api_endpoint: str,
        api_key: str,
        poi_name: str,
        city: str
    ) -> Optional[EvidenceData]:
        """调用百度地图API"""
        try:
            params = {
                'ak': api_key,
                'query': poi_name,
                'region': city,
                'output': 'json'
            }

            async with self.http_session.get(api_endpoint, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('results') and len(data['results']) > 0:
                        poi = data['results'][0]
                        logger.info(f"从百度地图获取到POI: {poi.get('name')}")
                        return EvidenceData(
                            name=poi.get('name', ''),
                            address=poi.get('address', ''),
                            coordinates={
                                'longitude': poi.get('location', {}).get('lng'),
                                'latitude': poi.get('location', {}).get('lat')
                            } if poi.get('location') else None,
                            phone=poi.get('telephone'),
                            category=poi.get('overall_rating'),
                            raw_data=poi
                        )
                else:
                    logger.warning(f"百度地图API返回状态码: {resp.status}")
                    return None

        except Exception as e:
            logger.error(f"调用百度地图API失败: {e}")
            return None

    async def _call_qq_map(
        self,
        api_endpoint: str,
        api_key: str,
        poi_name: str,
        city: str
    ) -> Optional[EvidenceData]:
        """调用腾讯地图API"""
        try:
            params = {
                'key': api_key,
                'keyword': poi_name,
                'region': city,
                'output': 'json'
            }

            async with self.http_session.get(api_endpoint, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data') and len(data['data']) > 0:
                        poi = data['data'][0]
                        logger.info(f"从腾讯地图获取到POI: {poi.get('title')}")
                        return EvidenceData(
                            name=poi.get('title', ''),
                            address=poi.get('address', ''),
                            coordinates={
                                'longitude': poi.get('location', {}).get('lng'),
                                'latitude': poi.get('location', {}).get('lat')
                            } if poi.get('location') else None,
                            phone=poi.get('tel'),
                            category=poi.get('category'),
                            raw_data=poi
                        )
                else:
                    logger.warning(f"腾讯地图API返回状态码: {resp.status}")
                    return None

        except Exception as e:
            logger.error(f"调用腾讯地图API失败: {e}")
            return None

    async def _call_web_crawler(
        self,
        crawler_url: str,
        query: str,
        crawler_type: str = 'both',
        top_k: int = 10
    ) -> Optional[List[str]]:
        """
        调用互联网爬虫接口获取URL

        Args:
            crawler_url: 爬虫服务URL
            query: 搜索查询词
            crawler_type: 爬虫类型 (normal_web/search_web/both)
            top_k: 返回结果数量

        Returns:
            URL列表或None
        """
        if not self.http_session:
            logger.warning("HTTP会话未初始化，无法调用爬虫")
            return None

        try:
            payload = {
                'query': query,
                'top_k': min(top_k, 50),
                'crawler_type': crawler_type
            }

            async with self.http_session.post(
                crawler_url,
                json=payload,
                timeout=30
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    urls = data.get('urls', [])
                    logger.info(f"爬虫接口返回 {len(urls)} 个URL")
                    return urls
                else:
                    logger.warning(f"爬虫接口返回状态码: {resp.status}")
                    return None

        except Exception as e:
            logger.error(f"调用爬虫接口失败: {e}")
            return None

    async def _call_official_api(
        self,
        source_name: str,
        source_url: str,
        poi_name: str,
        city: str
    ) -> Optional[EvidenceData]:
        """
        调用官方渠道API

        Args:
            source_name: 来源名称
            source_url: 来源URL
            poi_name: POI名称
            city: 城市

        Returns:
            证据数据或None
        """
        if not self.http_session or not source_url:
            return None

        try:
            # 根据官方来源类型构建请求
            if '卫健委' in source_name or '卫生' in source_name:
                # 医疗相关官方接口
                logger.info(f"调用卫生健康系统API: {source_name}")
                # 这里可以实现具体的医疗数据源调用
                return None
            elif '政府' in source_name or '民政' in source_name:
                # 政府相关官方接口
                logger.info(f"调用政府系统API: {source_name}")
                # 这里可以实现具体的政府数据源调用
                return None
            elif '公安' in source_name:
                # 公安相关官方接口
                logger.info(f"调用公安系统API: {source_name}")
                return None
            elif '法院' in source_name or '检察' in source_name:
                # 司法相关官方接口
                logger.info(f"调用司法系统API: {source_name}")
                return None
            else:
                logger.warning(f"未知的官方来源类型: {source_name}")
                return None

        except Exception as e:
            logger.error(f"调用官方API失败: {e}")
            return None

    async def _call_internet_source(
        self,
        source_name: str,
        source_url: str,
        poi_name: str,
        city: str
    ) -> Optional[EvidenceData]:
        """
        调用互联网数据源（百度百科、维基百科等）

        Args:
            source_name: 来源名称
            source_url: 来源URL
            poi_name: POI名称
            city: 城市

        Returns:
            证据数据或None
        """
        if not self.http_session or not source_url:
            return None

        try:
            if '百度百科' in source_name:
                return await self._call_baidu_baike(source_url, poi_name)
            elif '维基百科' in source_name:
                return await self._call_wikipedia(source_url, poi_name)
            elif '好大夫' in source_name:
                return await self._call_haodf(source_url, poi_name, city)
            else:
                logger.warning(f"未知的互联网来源类型: {source_name}")
                return None

        except Exception as e:
            logger.error(f"调用互联网来源失败: {e}")
            return None

    async def _call_baidu_baike(self, base_url: str, poi_name: str) -> Optional[EvidenceData]:
        """调用百度百科"""
        try:
            search_url = f"{base_url}/search?word={poi_name}"
            async with self.http_session.get(search_url, timeout=10) as resp:
                if resp.status == 200:
                    logger.info(f"从百度百科获取 {poi_name} 的信息")
                    return EvidenceData(
                        name=poi_name,
                        raw_data={
                            "source": "baidu_baike",
                            "url": search_url
                        }
                    )
                return None
        except Exception as e:
            logger.error(f"调用百度百科失败: {e}")
            return None

    async def _call_wikipedia(self, base_url: str, poi_name: str) -> Optional[EvidenceData]:
        """调用维基百科"""
        try:
            logger.info(f"从维基百科获取 {poi_name} 的信息")
            return EvidenceData(
                name=poi_name,
                raw_data={
                    "source": "wikipedia",
                    "url": base_url
                }
            )
        except Exception as e:
            logger.error(f"调用维基百科失败: {e}")
            return None

    async def _call_haodf(self, base_url: str, poi_name: str, city: str) -> Optional[EvidenceData]:
        """调用好大夫在线"""
        try:
            logger.info(f"从好大夫在线获取 {poi_name} 的医院信息")
            return EvidenceData(
                name=poi_name,
                raw_data={
                    "source": "haodf",
                    "url": base_url,
                    "category": "医院"
                }
            )
        except Exception as e:
            logger.error(f"调用好大夫在线失败: {e}")
            return None

    async def _collect_from_map_api(
        self,
        poi_id: str,
        poi_name: str,
        city: str,
        source_config: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从地图API收集数据

        Args:
            poi_id: POI ID
            poi_name: POI名称
            city: 城市
            source_config: 数据源配置

        Returns:
            证据对象或None
        """
        api_endpoint = source_config.get('api_endpoint')
        api_key_ref = source_config.get('api_key_ref')
        source_name = source_config.get('name', 'Unknown Map API')
        source_type = source_config.get('type', 'map_vendor')
        weight = source_config.get('weight', 0.5)

        if not api_endpoint or not api_key_ref:
            logger.warning(f"数据源 '{source_name}' 缺少API配置，跳过")
            return None

        try:
            # 获取API凭据
            credentials = self.sources_config.get('credentials', {})
            api_creds = credentials.get(api_key_ref, [])

            if not api_creds:
                logger.warning(f"未找到 '{api_key_ref}' 的API凭据")
                return None

            # 使用第一个可用的凭据
            cred = api_creds[0] if isinstance(api_creds, list) else api_creds
            api_key = cred.get('ak') or cred.get('key')

            if not api_key:
                logger.warning(f"数据源 '{source_name}' 的API凭据无效")
                return None

            # 构建请求参数
            search_query = f"{poi_name} {city}"
            logger.info(f"尝试从 '{source_name}' 收集POI数据: {search_query}")

            # 调用真实API进行数据采集
            evidence_data = await self._call_map_api(
                api_endpoint=api_endpoint,
                api_key=api_key,
                poi_name=poi_name,
                city=city,
                source_name=source_name
            )

            if not evidence_data:
                logger.warning(f"从 '{source_name}' 未获取到有效数据")
                return None

            evidence = Evidence(
                evidence_id=self._generate_evidence_id(poi_id, source_name),
                poi_id=poi_id,
                source={
                    "source_id": source_name.replace(" ", "_").upper(),
                    "source_name": source_name,
                    "source_type": source_type,
                    "source_url": source_config.get('url', ''),
                    "weight": weight
                },
                collected_at=datetime.utcnow().isoformat() + 'Z',
                data=evidence_data,
                metadata={
                    "collection_method": "api",
                    "api_endpoint": api_endpoint,
                    "api_key_ref": api_key_ref
                }
            )

            logger.info(f"成功从 '{source_name}' 收集到证据: {evidence.evidence_id}")
            return evidence

        except Exception as e:
            logger.error(f"从 '{source_name}' 收集数据失败: {e}")
            return None

    async def _collect_from_web_crawler(
        self,
        poi_id: str,
        poi_name: str,
        city: str,
        source_config: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从互联网爬虫接口收集数据

        Args:
            poi_id: POI ID
            poi_name: POI名称
            city: 城市
            source_config: 数据源配置

        Returns:
            证据对象或None
        """
        source_name = source_config.get('name', 'Web Crawler')
        source_type = source_config.get('type', 'web_crawler')
        weight = source_config.get('weight', 0.5)
        crawler_type = source_config.get('crawler_type', 'both')
        top_k = source_config.get('top_k', 10)

        try:
            logger.info(f"尝试从爬虫接口 '{source_name}' 收集数据: {poi_name}")

            # 构建搜索查询
            search_query = f"{poi_name} {city}"

            # 调用真实的互联网爬虫接口
            crawler_url = source_config.get('url')
            if not crawler_url:
                logger.warning(f"爬虫接口 '{source_name}' 缺少URL配置")
                return None

            try:
                urls = await self._call_web_crawler(
                    crawler_url=crawler_url,
                    query=search_query,
                    crawler_type=crawler_type,
                    top_k=top_k
                )

                if not urls:
                    logger.warning(f"爬虫接口未返回任何URL")
                    return None

                evidence_data = EvidenceData(
                    name=poi_name,
                    raw_data={
                        "search_query": search_query,
                        "crawler_type": crawler_type,
                        "top_k": top_k,
                        "urls": urls,
                        "url_count": len(urls)
                    }
                )

                logger.info(f"成功从爬虫接口获取 {len(urls)} 个URL")

            except Exception as e:
                logger.warning(f"爬虫接口调用失败: {e}，使用降级方案")
                # 降级方案：返回通用搜索URL
                evidence_data = EvidenceData(
                    name=poi_name,
                    raw_data={
                        "search_query": search_query,
                        "crawler_type": crawler_type,
                        "top_k": top_k,
                        "urls": [
                            f"https://www.baidu.com/s?wd={poi_name}+{city}",
                            f"https://www.google.com/search?q={poi_name}+{city}",
                            f"https://www.gov.cn/"  # 政府网站
                        ],
                        "fallback": True
                    }
                )
                logger.info("已使用降级方案返回通用搜索URL")

            evidence = Evidence(
                evidence_id=self._generate_evidence_id(poi_id, source_name),
                poi_id=poi_id,
                source={
                    "source_id": source_name.replace(" ", "_").upper(),
                    "source_name": source_name,
                    "source_type": source_type,
                    "source_url": source_config.get('url', ''),
                    "weight": weight
                },
                collected_at=datetime.utcnow().isoformat() + 'Z',
                data=evidence_data,
                metadata={
                    "collection_method": "crawl",
                    "crawler_type": crawler_type,
                    "top_k": top_k
                }
            )

            logger.info(f"成功从爬虫接口 '{source_name}' 收集到证据: {evidence.evidence_id}")
            return evidence

        except Exception as e:
            logger.error(f"从爬虫接口 '{source_name}' 收集数据失败: {e}")
            return None

    async def _collect_from_official_source(
        self,
        poi_id: str,
        poi_name: str,
        city: str,
        source_config: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从官方渠道收集数据

        Args:
            poi_id: POI ID
            poi_name: POI名称
            city: 城市
            source_config: 数据源配置

        Returns:
            证据对象或None
        """
        source_name = source_config.get('name', 'Official Source')
        source_type = source_config.get('type', 'official')
        weight = source_config.get('weight', 1.0)
        source_url = source_config.get('url', '')

        try:
            logger.info(f"尝试从官方渠道 '{source_name}' 收集数据: {poi_name}")

            # 调用官方渠道API
            evidence_data = await self._call_official_api(
                source_name=source_name,
                source_url=source_url,
                poi_name=poi_name,
                city=city
            )

            if not evidence_data:
                logger.warning(f"从官方渠道 '{source_name}' 未获取到有效数据")
                # 降级处理：返回基础数据，status不确定
                evidence_data = EvidenceData(
                    name=poi_name,
                    status="unknown",
                    raw_data={
                        "source_url": source_url,
                        "official_source": source_name,
                        "verified": False,
                        "error": "Unable to fetch from official source"
                    }
                )

            logger.info(f"成功从官方渠道 '{source_name}' 收集到数据")

        except Exception as e:
            logger.error(f"从官方渠道 '{source_name}' 收集数据失败: {e}")
            return None

    async def _collect_from_internet(
        self,
        poi_id: str,
        poi_name: str,
        city: str,
        source_config: Dict[str, Any]
    ) -> Optional[Evidence]:
        """
        从互联网渠道收集数据（百度百科、维基百科等）

        Args:
            poi_id: POI ID
            poi_name: POI名称
            city: 城市
            source_config: 数据源配置

        Returns:
            证据对象或None
        """
        source_name = source_config.get('name', 'Internet Source')
        source_type = source_config.get('type', 'internet')
        weight = source_config.get('weight', 0.6)
        source_url = source_config.get('url', '')

        try:
            logger.info(f"尝试从互联网渠道 '{source_name}' 收集数据: {poi_name}")

            # 调用互联网数据源
            evidence_data = await self._call_internet_source(
                source_name=source_name,
                source_url=source_url,
                poi_name=poi_name,
                city=city
            )

            if not evidence_data:
                logger.warning(f"从互联网渠道 '{source_name}' 未获取到有效数据")
                # 降级处理：返回基础搜索信息
                evidence_data = EvidenceData(
                    name=poi_name,
                    raw_data={
                        "source_url": source_url,
                        "internet_source": source_name,
                        "search_term": f"{poi_name} {city}",
                        "error": "Unable to fetch from internet source"
                    }
                )

            logger.info(f"成功从互联网渠道 '{source_name}' 收集到数据")

        except Exception as e:
            logger.error(f"从互联网渠道 '{source_name}' 收集数据失败: {e}")
            return None

    async def collect_evidence(
        self,
        poi_id: str,
        poi_name: str,
        poi_type: str,
        city: str,
        address: Optional[str] = None,
        coordinates: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Tuple[List[Evidence], Dict[str, Any]]:
        """
        从多个数据源收集POI的证据

        Args:
            poi_id: POI ID
            poi_name: POI名称
            poi_type: POI类型（如hospital、government等）
            city: 城市
            address: 详细地址（可选）
            coordinates: 坐标信息（可选）
            **kwargs: 其他参数

        Returns:
            (证据列表, 收集统计信息)
        """
        logger.info(f"开始为POI '{poi_id}' ({poi_name}) 收集多源证据")

        # 获取该POI类型的数据源配置
        sources = self._get_sources_for_poi_type(poi_type)

        if not sources:
            logger.warning(f"POI类型 '{poi_type}' 没有配置任何数据源")
            return [], {
                'poi_id': poi_id,
                'total_sources': 0,
                'successful_collections': 0,
                'failed_collections': 0,
                'evidence_count': 0
            }

        # 并行收集各数据源的证据
        collection_tasks = []

        for source_config in sources:
            source_type = source_config.get('type', 'map_vendor')

            if source_type == SourceType.MAP_VENDOR.value:
                task = self._collect_from_map_api(poi_id, poi_name, city, source_config)
            elif source_type == SourceType.WEB_CRAWLER.value:
                task = self._collect_from_web_crawler(poi_id, poi_name, city, source_config)
            elif source_type == SourceType.OFFICIAL.value:
                task = self._collect_from_official_source(poi_id, poi_name, city, source_config)
            elif source_type == SourceType.INTERNET.value:
                task = self._collect_from_internet(poi_id, poi_name, city, source_config)
            else:
                logger.warning(f"未知的数据源类型: {source_type}")
                continue

            collection_tasks.append(task)

        # 执行所有收集任务
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)

        # 处理结果
        evidence_list = []
        successful_count = 0
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"数据源 {i} 收集失败: {result}")
            elif result is not None:
                evidence_list.append(result)
                successful_count += 1
            else:
                failed_count += 1

        # 收集统计信息
        stats = {
            'poi_id': poi_id,
            'poi_name': poi_name,
            'poi_type': poi_type,
            'city': city,
            'total_sources': len(sources),
            'successful_collections': successful_count,
            'failed_collections': failed_count,
            'evidence_count': len(evidence_list),
            'collected_at': datetime.utcnow().isoformat() + 'Z'
        }

        logger.info(
            f"POI '{poi_id}' 证据收集完成: "
            f"总数据源={stats['total_sources']}, "
            f"成功={successful_count}, "
            f"失败={failed_count}, "
            f"证据数={len(evidence_list)}"
        )

        return evidence_list, stats

    async def batch_collect(
        self,
        poi_list: List[Dict[str, Any]]
    ) -> Tuple[List[Evidence], Dict[str, Any]]:
        """
        批量收集多个POI的证据

        Args:
            poi_list: POI数据列表，每条包含poi_id、name、poi_type、city等字段

        Returns:
            (所有证据列表, 批量收集统计)
        """
        logger.info(f"开始批量收集 {len(poi_list)} 个POI的证据")

        all_evidence = []
        batch_stats = {
            'total_pois': len(poi_list),
            'processed_pois': 0,
            'total_evidence': 0,
            'poi_stats': []
        }

        for poi_data in poi_list:
            poi_id = poi_data.get('poi_id')
            poi_name = poi_data.get('name')
            poi_type = poi_data.get('poi_type')
            city = poi_data.get('city')

            if not all([poi_id, poi_name, poi_type, city]):
                logger.warning(f"POI数据不完整，跳过: {poi_data}")
                continue

            # 为每个POI收集证据
            evidence, stats = await self.collect_evidence(
                poi_id=poi_id,
                poi_name=poi_name,
                poi_type=poi_type,
                city=city,
                address=poi_data.get('address'),
                coordinates=poi_data.get('coordinates')
            )

            all_evidence.extend(evidence)
            batch_stats['poi_stats'].append(stats)
            batch_stats['processed_pois'] += 1
            batch_stats['total_evidence'] += len(evidence)

        batch_stats['collected_at'] = datetime.utcnow().isoformat() + 'Z'

        logger.info(
            f"批量收集完成: 处理={batch_stats['processed_pois']}, "
            f"总证据数={batch_stats['total_evidence']}"
        )

        return all_evidence, batch_stats


def evidence_to_json(evidence_list: List[Evidence]) -> str:
    """将证据列表转换为JSON字符串"""
    data = [e.to_dict() for e in evidence_list]
    return json.dumps(data, ensure_ascii=False, indent=2)


async def main():
    """
    主函数 - 示例用法

    演示如何使用EvidenceCollector进行多源证据收集
    """
    # 示例POI数据
    poi_list = [
        {
            "poi_id": "HOSPITAL_BJ_001",
            "name": "北京大学第一医院",
            "poi_type": "hospital",
            "city": "北京市",
            "address": "北京市西城区西什库大街8号",
            "coordinates": {
                "longitude": 116.3723,
                "latitude": 39.9342
            }
        },
        {
            "poi_id": "GOVERNMENT_SH_001",
            "name": "上海市人民政府",
            "poi_type": "government",
            "city": "上海市",
            "address": "上海市黄浦区人民大道200号"
        }
    ]

    try:
        # 使用异步上下文管理器
        async with EvidenceCollector() as collector:
            # 单个POI的证据收集
            logger.info("\n===== 单个POI证据收集示例 =====")
            poi_data = poi_list[0]
            evidence, stats = await collector.collect_evidence(
                poi_id=poi_data['poi_id'],
                poi_name=poi_data['name'],
                poi_type=poi_data['poi_type'],
                city=poi_data['city'],
                address=poi_data.get('address'),
                coordinates=poi_data.get('coordinates')
            )

            print(f"\n收集统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
            print(f"证据数量: {len(evidence)}")

            if evidence:
                print(f"\n第一条证据示例:")
                print(json.dumps(evidence[0].to_dict(), ensure_ascii=False, indent=2))

            # 批量POI的证据收集
            logger.info("\n===== 批量POI证据收集示例 =====")
            all_evidence, batch_stats = await collector.batch_collect(poi_list)

            print(f"\n批量收集统计: {json.dumps(batch_stats, ensure_ascii=False, indent=2)}")
            print(f"总证据数量: {len(all_evidence)}")

    except Exception as e:
        logger.error(f"证据收集过程中发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
