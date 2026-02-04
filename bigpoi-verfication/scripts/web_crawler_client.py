#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.0
# Author: BigPOI Verification System
# Date: 2024-01-15

互联网网站信息获取客户端模块

功能定义说明:
    本模块封装了两个用于互联网网站信息获取的接口调用功能:
    1. normal_web_playwright: 普通网页抓取接口
    2. search_web_filter: 网页搜索过滤接口

用途说明:
    该模块为大POI核实流程提供互联网情报获取能力，通过调用外部爬虫服务
    获取与目标POI相关的网页URL列表，作为多源情报收集的重要补充来源。
    
    主要功能:
    1. 封装HTTP POST请求，统一处理接口调用
    2. 提供完善的异常处理机制（网络错误、超时、格式异常等）
    3. 支持请求参数的动态构建和验证
    4. 返回结构化的URL数据，便于后续处理

    应用场景: 需要从互联网获取POI相关网页链接、进行多源信息交叉验证
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientConnectorError

# 配置日志
logger = logging.getLogger(__name__)


class CrawlResult:
    """爬虫接口返回结果数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        urls: List[str],
        query: str,
        top_k: int,
        site: Optional[str],
        date_range: Optional[str],
        response_time_ms: float,
        collected_at: str
    ):
        self.urls = urls
        self.query = query
        self.top_k = top_k
        self.site = site
        self.date_range = date_range
        self.response_time_ms = response_time_ms
        self.collected_at = collected_at

    def __repr__(self):
        return (f"CrawlResult(query={self.query!r}, "
                f"urls_count={len(self.urls)}, "
                f"response_time_ms={self.response_time_ms:.2f})")


class WebCrawlerClient:
    """
    互联网网站信息获取客户端
    
    封装两个爬虫接口的调用功能:
    - normal_web_playwright: 普通网页抓取
    - search_web_filter: 网页搜索过滤
    """
    
    # 接口基础URL
    BASE_URL = "http://10.82.122.209:9081"
    
    # 接口路径
    NORMAL_WEB_ENDPOINT = "/crawl/normal_web_playwright"
    SEARCH_WEB_ENDPOINT = "/crawl/search_web_filter"
    
    # 默认配置
    DEFAULT_TIMEOUT = 30  # 默认超时时间（秒）
    DEFAULT_TOP_K = 10    # 默认返回结果数量
    MAX_TOP_K = 50        # 最大返回结果数量
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """
        初始化爬虫客户端
        
        Args:
            timeout: 请求超时时间（秒），默认30秒
        """
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"WebCrawlerClient初始化完成，超时设置: {timeout}s")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        client_timeout = ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=client_timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _build_request_payload(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        site: Optional[str] = None,
        date_range: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        构建请求参数
        
        Args:
            query: 搜索关键词
            top_k: 返回结果数量，最大值为50
            site: 网站约束（例如:"*.gov.cn"）
            date_range: 时间范围（例如："2025-01-06,2026-01-06"）
            
        Returns:
            请求参数字典
            
        Raises:
            ValueError: 当参数验证失败时抛出
        """
        # 参数验证
        if not query or not isinstance(query, str):
            raise ValueError("query参数必须为非空字符串")
        
        # 限制top_k范围
        top_k = min(max(1, top_k), self.MAX_TOP_K)
        
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        # 可选参数
        if site:
            payload["site"] = site
        if date_range:
            payload["date_range"] = date_range
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any]) -> List[str]:
        """
        解析接口返回数据
        
        Args:
            response_data: 接口返回的JSON数据
            
        Returns:
            URL列表
            
        Raises:
            ValueError: 当返回格式异常时抛出
        """
        if not isinstance(response_data, dict):
            raise ValueError(f"返回数据格式错误: 期望dict，实际为{type(response_data)}")
        
        urls_data = response_data.get("urls", [])
        if not isinstance(urls_data, list):
            raise ValueError(f"urls字段格式错误: 期望list，实际为{type(urls_data)}")
        
        # 提取URL字符串
        urls = []
        for item in urls_data:
            if isinstance(item, dict) and "url" in item:
                url = item.get("url")
                if url and isinstance(url, str):
                    urls.append(url)
            elif isinstance(item, str):
                urls.append(item)
        
        return urls
    
    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> List[str]:
        """
        执行HTTP POST请求
        
        Args:
            endpoint: 接口路径
            payload: 请求参数
            
        Returns:
            URL列表
            
        Raises:
            ClientError: 网络请求异常
            ValueError: 响应数据解析异常
            TimeoutError: 请求超时
        """
        if not self.session:
            raise RuntimeError("ClientSession未初始化，请使用async with上下文")
        
        url = f"{self.BASE_URL}{endpoint}"
        start_time = datetime.now()
        
        try:
            logger.debug(f"发送请求到 {url}, 参数: {payload}")
            
            async with self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                # 检查HTTP状态码
                if response.status != 200:
                    error_text = await response.text()
                    raise ClientError(
                        f"HTTP请求失败: status={response.status}, "
                        f"reason={response.reason}, body={error_text[:200]}"
                    )
                
                # 解析JSON响应
                try:
                    response_data = await response.json()
                except json.JSONDecodeError as e:
                    response_text = await response.text()
                    raise ValueError(
                        f"JSON解析失败: {e}, "
                        f"响应内容: {response_text[:200]}"
                    )
                
                # 解析URL列表
                urls = self._parse_response(response_data)
                
                elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(
                    f"请求成功: {url}, "
                    f"获取到 {len(urls)} 个URL, "
                    f"耗时: {elapsed_ms:.2f}ms"
                )
                
                return urls
                
        except ClientConnectorError as e:
            logger.error(f"连接失败: 无法连接到服务器 {self.BASE_URL}, 错误: {e}")
            raise ConnectionError(f"无法连接到爬虫服务器: {e}") from e
        except TimeoutError as e:
            logger.error(f"请求超时: {url}, 超时设置: {self.timeout}s")
            raise TimeoutError(f"请求超时({self.timeout}s): {e}") from e
        except ClientError as e:
            logger.error(f"HTTP请求异常: {e}")
            raise
        except Exception as e:
            logger.error(f"请求过程中发生未知错误: {e}")
            raise
    
    async def crawl_normal_web(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        site: Optional[str] = None,
        date_range: Optional[str] = None
    ) -> CrawlResult:
        """
        调用普通网页抓取接口
        
        Args:
            query: 搜索关键词（必填）
            top_k: 返回结果数量，默认10，最大50
            site: 网站约束，例如"*.gov.cn"表示只搜索gov.cn域名
            date_range: 时间范围，格式为"YYYY-MM-DD,YYYY-MM-DD"
            
        Returns:
            CrawlResult对象，包含URL列表和元数据
            
        Raises:
            ValueError: 参数验证失败
            ConnectionError: 连接服务器失败
            TimeoutError: 请求超时
            ClientError: HTTP请求异常
        """
        logger.info(f"调用normal_web_playwright接口, query: {query}")
        
        payload = self._build_request_payload(query, top_k, site, date_range)
        start_time = datetime.now()
        
        urls = await self._make_request(self.NORMAL_WEB_ENDPOINT, payload)
        
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return CrawlResult(
            urls=urls,
            query=query,
            top_k=payload["top_k"],
            site=site,
            date_range=date_range,
            response_time_ms=elapsed_ms,
            collected_at=datetime.utcnow().isoformat() + 'Z'
        )
    
    async def crawl_search_web(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        site: Optional[str] = None,
        date_range: Optional[str] = None
    ) -> CrawlResult:
        """
        调用网页搜索过滤接口
        
        Args:
            query: 搜索关键词（必填）
            top_k: 返回结果数量，默认10，最大50
            site: 网站约束，例如"*.gov.cn"表示只搜索gov.cn域名
            date_range: 时间范围，格式为"YYYY-MM-DD,YYYY-MM-DD"
            
        Returns:
            CrawlResult对象，包含URL列表和元数据
            
        Raises:
            ValueError: 参数验证失败
            ConnectionError: 连接服务器失败
            TimeoutError: 请求超时
            ClientError: HTTP请求异常
        """
        logger.info(f"调用search_web_filter接口, query: {query}")
        
        payload = self._build_request_payload(query, top_k, site, date_range)
        start_time = datetime.now()
        
        urls = await self._make_request(self.SEARCH_WEB_ENDPOINT, payload)
        
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return CrawlResult(
            urls=urls,
            query=query,
            top_k=payload["top_k"],
            site=site,
            date_range=date_range,
            response_time_ms=elapsed_ms,
            collected_at=datetime.utcnow().isoformat() + 'Z'
        )
    
    async def crawl_both(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        site: Optional[str] = None,
        date_range: Optional[str] = None
    ) -> Dict[str, CrawlResult]:
        """
        同时调用两个接口获取更全面的结果
        
        Args:
            query: 搜索关键词（必填）
            top_k: 返回结果数量，默认10，最大50
            site: 网站约束，例如"*.gov.cn"
            date_range: 时间范围，格式为"YYYY-MM-DD,YYYY-MM-DD"
            
        Returns:
            字典，包含两个接口的返回结果:
            {
                "normal_web": CrawlResult,
                "search_web": CrawlResult
            }
        """
        import asyncio
        
        logger.info(f"同时调用两个爬虫接口, query: {query}")
        
        # 并行调用两个接口
        tasks = [
            self.crawl_normal_web(query, top_k, site, date_range),
            self.crawl_search_web(query, top_k, site, date_range)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {
            "normal_web": None,
            "search_web": None
        }
        
        # 处理结果
        if not isinstance(results[0], Exception):
            output["normal_web"] = results[0]
        else:
            logger.error(f"normal_web_playwright接口调用失败: {results[0]}")
        
        if not isinstance(results[1], Exception):
            output["search_web"] = results[1]
        else:
            logger.error(f"search_web_filter接口调用失败: {results[1]}")
        
        return output


# 便捷函数，用于快速调用
def create_crawler_client(timeout: int = 30) -> WebCrawlerClient:
    """
    创建爬虫客户端实例
    
    Args:
        timeout: 请求超时时间（秒）
        
    Returns:
        WebCrawlerClient实例
    """
    return WebCrawlerClient(timeout=timeout)
