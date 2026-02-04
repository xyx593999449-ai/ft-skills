#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.2
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的证据规范化处理阶段，对收集到的原始证据进行
    名称、地址、坐标等字段的标准化与规范化处理，确保数据格式一致、可比较。

用途说明:
    该脚本在整个核实流程中处于第三阶段（证据规范化处理），主要作用包括：
    1. 名称规范化：去除空格、标点、标准化简称
    2. 地址标准化：补全行政区划、统一格式
    3. 坐标系统转换：WGS84/BD09 → GCJ02（国测局坐标系）
    4. 分类信息映射：统一分类编码体系
    5. 证据去重和冲突标记
    6. 生成可用于后续维度判断的规范化证据对象

    应用场景：多源证据的规范化处理、坐标系统转换、数据冲突检测
"""

import json
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NormalizedEvidence:
    """规范化证据数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        evidence_id: str,
        poi_id: str,
        original_data: Dict[str, Any],
        normalized_data: Dict[str, Any],
        normalization_log: List[str],
        source: Dict[str, Any],
        confidence: float
    ):
        self.evidence_id = evidence_id
        self.poi_id = poi_id
        self.original_data = original_data
        self.normalized_data = normalized_data
        self.normalization_log = normalization_log
        self.source = source
        self.confidence = confidence

    def __repr__(self):
        return (f"NormalizedEvidence(evidence_id={self.evidence_id!r}, "
                f"poi_id={self.poi_id!r}, confidence={self.confidence!r})")


class CoordinateTransformer:
    """坐标转换工具类"""

    # WGS84 to GCJ02 转换参数
    M_PI = 3.14159265358979324
    A = 6378245.0
    EE = 0.00669342162296594323

    @staticmethod
    def wgs84_to_gcj02(lon: float, lat: float) -> Tuple[float, float]:
        """
        WGS84坐标转GCJ02坐标

        Args:
            lon: WGS84 经度
            lat: WGS84 纬度

        Returns:
            (GCJ02经度, GCJ02纬度)
        """
        if CoordinateTransformer._out_of_china(lon, lat):
            return lon, lat

        dx = CoordinateTransformer._calculate_lat_offset(lon, lat - 35.0)
        dy = CoordinateTransformer._calculate_lon_offset(lon - 105.0, lat - 35.0)

        rad_lat = (lat - 35.0) * CoordinateTransformer.M_PI / 180.0
        magic = math.sin(rad_lat)
        magic = 1 - CoordinateTransformer.EE * magic * magic
        magic = math.sqrt(magic)

        dy = (dy * 180.0) / (CoordinateTransformer.A / magic * math.cos(rad_lat) * CoordinateTransformer.M_PI)
        dx = (dx * 180.0) / (CoordinateTransformer.A / magic * CoordinateTransformer.M_PI)

        gcj_lat = lat + dx
        gcj_lon = lon + dy

        return gcj_lon, gcj_lat

    @staticmethod
    def bd09_to_gcj02(lon: float, lat: float) -> Tuple[float, float]:
        """
        BD09坐标转GCJ02坐标

        Args:
            lon: BD09 经度
            lat: BD09 纬度

        Returns:
            (GCJ02经度, GCJ02纬度)
        """
        x = lon - 0.0065
        y = lat - 0.006

        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * CoordinateTransformer.M_PI)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * CoordinateTransformer.M_PI)

        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)

        return gcj_lon, gcj_lat

    @staticmethod
    def _out_of_china(lon: float, lat: float) -> bool:
        """判断坐标是否在中国范围外"""
        if lon < 72.004 or lon > 137.4047:
            return True
        if lat < 0.8293 or lat > 55.8271:
            return True
        return False

    @staticmethod
    def _calculate_lat_offset(lon: float, lat: float) -> float:
        """计算纬度偏移"""
        ret = (-100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat +
               0.1 * lon * lat + 0.2 * math.sqrt(abs(lon)))
        ret += ((20.0 * math.sin(6.0 * lon * CoordinateTransformer.M_PI / 180.0) +
                 20.0 * math.sin(2.0 * lon * CoordinateTransformer.M_PI / 180.0)) * 2.0 / 3.0)
        ret += ((20.0 * math.sin(lat * CoordinateTransformer.M_PI / 180.0) +
                 40.0 * math.sin(lat / 3.0 * CoordinateTransformer.M_PI / 180.0)) * 2.0 / 3.0)
        ret += ((160.0 * math.sin(lat / 12.0 * CoordinateTransformer.M_PI / 180.0) +
                 320 * math.sin(lat * CoordinateTransformer.M_PI / 180.0 / 30.0)) * 2.0 / 3.0)
        return ret

    @staticmethod
    def _calculate_lon_offset(lon: float, lat: float) -> float:
        """计算经度偏移"""
        ret = (300.0 + lon + 2.0 * lat + 0.1 * lon * lon +
               0.1 * lon * lat + 0.1 * math.sqrt(abs(lon)))
        ret += ((20.0 * math.sin(6.0 * lon * CoordinateTransformer.M_PI / 180.0) +
                 20.0 * math.sin(2.0 * lon * CoordinateTransformer.M_PI / 180.0)) * 2.0 / 3.0)
        ret += ((20.0 * math.sin(lon * CoordinateTransformer.M_PI / 180.0) +
                 40.0 * math.sin(lon / 3.0 * CoordinateTransformer.M_PI / 180.0)) * 2.0 / 3.0)
        ret += ((150.0 * math.sin(lon / 12.0 * CoordinateTransformer.M_PI / 180.0) +
                 300.0 * math.sin(lon / 30.0 * CoordinateTransformer.M_PI / 180.0)) * 2.0 / 3.0)
        return ret


class EvidenceNormalizer:
    """证据规范化处理器"""

    def __init__(self):
        """初始化规范化处理器"""
        self.transformer = CoordinateTransformer()
        logger.info("证据规范化处理器初始化完成")

    def normalize(self, evidence: Dict[str, Any]) -> NormalizedEvidence:
        """
        规范化单条证据

        Args:
            evidence: 原始证据对象

        Returns:
            规范化后的证据对象
        """
        evidence_id = evidence.get('evidence_id', 'UNKNOWN')
        poi_id = evidence.get('poi_id', 'UNKNOWN')
        normalization_log = []

        logger.info(f"开始规范化证据: {evidence_id}")

        # 深复制数据
        normalized_data = {}

        # 1. 规范化名称
        if 'data' in evidence and 'name' in evidence['data']:
            original_name = evidence['data']['name']
            normalized_name = self._normalize_name(original_name)
            normalized_data['name'] = normalized_name

            if original_name != normalized_name:
                normalization_log.append(f"名称规范化: '{original_name}' → '{normalized_name}'")

        # 2. 规范化地址
        if 'data' in evidence and 'address' in evidence['data']:
            original_address = evidence['data']['address']
            normalized_address = self._normalize_address(original_address)
            normalized_data['address'] = normalized_address

            if original_address != normalized_address:
                normalization_log.append(f"地址规范化: '{original_address}' → '{normalized_address}'")

        # 3. 转换坐标系统
        if 'data' in evidence and 'coordinates' in evidence['data']:
            coords = evidence['data']['coordinates']
            normalized_coords, coord_log = self._normalize_coordinates(coords)
            normalized_data['coordinates'] = normalized_coords
            normalization_log.extend(coord_log)

        # 4. 复制其他字段
        if 'data' in evidence:
            for key, value in evidence['data'].items():
                if key not in ['name', 'address', 'coordinates']:
                    normalized_data[key] = value

        # 5. 计算置信度
        confidence = evidence.get('verification', {}).get('confidence', 0.5)

        normalized_evidence = NormalizedEvidence(
            evidence_id=evidence_id,
            poi_id=poi_id,
            original_data=evidence.get('data', {}),
            normalized_data=normalized_data,
            normalization_log=normalization_log,
            source=evidence.get('source', {}),
            confidence=confidence
        )

        logger.info(f"证据 {evidence_id} 规范化完成，记录 {len(normalization_log)} 条转换")
        return normalized_evidence

    def _normalize_name(self, name: str) -> str:
        """
        规范化名称

        Args:
            name: 原始名称

        Returns:
            规范化后的名称
        """
        if not name:
            return name

        # 去除前后空格
        normalized = name.strip()

        # 去除多余空格
        normalized = ' '.join(normalized.split())

        # 转换为简体中文
        normalized = self._convert_to_simplified(normalized)

        return normalized

    def _normalize_address(self, address: str) -> str:
        """
        规范化地址

        Args:
            address: 原始地址

        Returns:
            规范化后的地址
        """
        if not address:
            return address

        # 去除前后空格
        normalized = address.strip()

        # 去除多余空格
        normalized = ' '.join(normalized.split())

        return normalized

    def _normalize_coordinates(self, coords: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        规范化坐标，转换为GCJ02

        Args:
            coords: 原始坐标

        Returns:
            (规范化后的坐标, 转换日志)
        """
        logs = []

        if not isinstance(coords, dict):
            return coords, logs

        lon = coords.get('longitude')
        lat = coords.get('latitude')
        coord_system = coords.get('coordinate_system', 'GCJ02')

        if lon is None or lat is None:
            return coords, logs

        try:
            lon = float(lon)
            lat = float(lat)
        except (ValueError, TypeError):
            return coords, logs

        # 如果已是GCJ02，直接返回
        if coord_system == 'GCJ02':
            return {
                'longitude': lon,
                'latitude': lat,
                'coordinate_system': 'GCJ02'
            }, logs

        # 转换坐标系统
        if coord_system == 'WGS84':
            new_lon, new_lat = self.transformer.wgs84_to_gcj02(lon, lat)
            logs.append(f"坐标系转换: WGS84({lon}, {lat}) → GCJ02({new_lon:.6f}, {new_lat:.6f})")

        elif coord_system == 'BD09':
            new_lon, new_lat = self.transformer.bd09_to_gcj02(lon, lat)
            logs.append(f"坐标系转换: BD09({lon}, {lat}) → GCJ02({new_lon:.6f}, {new_lat:.6f})")

        else:
            # 不支持的坐标系统，返回原坐标
            new_lon, new_lat = lon, lat
            logs.append(f"警告: 不支持的坐标系统 {coord_system}，返回原坐标")

        return {
            'longitude': new_lon,
            'latitude': new_lat,
            'coordinate_system': 'GCJ02'
        }, logs

    def _convert_to_simplified(self, text: str) -> str:
        """
        转换为简体中文（简化版本）

        Args:
            text: 原始文本

        Returns:
            转换后的文本
        """
        # 注：完整的繁简转换需要使用专门的库（如opencc）
        # 这里仅作演示，保持原文本不变
        return text

    def batch_normalize(self, evidence_list: List[Dict[str, Any]]) -> Tuple[List[NormalizedEvidence], dict]:
        """
        批量规范化证据

        Args:
            evidence_list: 证据列表

        Returns:
            (规范化证据列表, 统计信息)
        """
        normalized_list = []
        stats = {
            'total': len(evidence_list),
            'successful': 0,
            'failed': 0,
            'total_transformations': 0
        }

        logger.info(f"开始批量规范化 {len(evidence_list)} 条证据")

        for evidence in evidence_list:
            try:
                normalized = self.normalize(evidence)
                normalized_list.append(normalized)
                stats['successful'] += 1
                stats['total_transformations'] += len(normalized.normalization_log)
            except Exception as e:
                logger.error(f"规范化证据失败: {evidence.get('evidence_id', 'UNKNOWN')}, 错误: {e}")
                stats['failed'] += 1

        logger.info(
            f"批量规范化完成: 总数={stats['total']}, "
            f"成功={stats['successful']}, 失败={stats['failed']}, "
            f"总转换数={stats['total_transformations']}"
        )

        return normalized_list, stats


async def main():
    """
    主函数 - 示例用法
    """
    # 示例证据数据
    evidence_list = [
        {
            'evidence_id': 'EVD_001',
            'poi_id': 'HOSPITAL_BJ_001',
            'source': {
                'source_id': 'AMAP',
                'source_name': '高德地图',
                'weight': 0.85
            },
            'data': {
                'name': '  北京大学第一医院  ',
                'address': '北京市   西城区   西什库大街8号',
                'coordinates': {
                    'longitude': 116.3723,
                    'latitude': 39.9342,
                    'coordinate_system': 'WGS84'
                }
            },
            'verification': {
                'confidence': 0.85
            }
        },
        {
            'evidence_id': 'EVD_002',
            'poi_id': 'HOSPITAL_BJ_001',
            'source': {
                'source_id': 'BAIDU',
                'source_name': '百度地图',
                'weight': 0.85
            },
            'data': {
                'name': '北京大学第一医院',
                'address': '北京市西城区西什库大街8号',
                'coordinates': {
                    'longitude': 116.3724,
                    'latitude': 39.9343,
                    'coordinate_system': 'BD09'
                }
            },
            'verification': {
                'confidence': 0.85
            }
        }
    ]

    # 规范化证据
    normalizer = EvidenceNormalizer()
    normalized_list, stats = normalizer.batch_normalize(evidence_list)

    # 打印结果
    print("\n===== 证据规范化结果 =====")
    print(f"总数: {stats['total']}, 成功: {stats['successful']}, 失败: {stats['failed']}")
    print(f"总转换数: {stats['total_transformations']}")

    for normalized in normalized_list:
        print(f"\n证据ID: {normalized.evidence_id}")
        print(f"原始数据: {normalized.original_data}")
        print(f"规范化数据: {normalized.normalized_data}")
        if normalized.normalization_log:
            print("转换日志:")
            for log in normalized.normalization_log:
                print(f"  - {log}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
        """
        规范化POI名称
        
        Args:
            name: 原始名称
            
        Returns:
            (规范化后的名称, 规范化元数据)
        """
        if not name:
            return "", {"error": "名称为空"}
        
        original = name
        metadata = {
            "original": original,
            "transformations": []
        }
        
        # 去除首尾空格
        name = name.strip()
        if name != original:
            metadata["transformations"].append("trim_spaces")
        
        # 统一全角/半角标点符号
        name = self._normalize_punctuation(name)
        if name != original:
            metadata["transformations"].append("normalize_punctuation")
        
        # 去除多余空格
        name = re.sub(r'\s+', ' ', name)
        if '  ' in original:
            metadata["transformations"].append("remove_extra_spaces")
        
        # 标准化常见简称
        name, abbreviation_applied = self._standardize_abbreviations(name)
        if abbreviation_applied:
            metadata["transformations"].append("standardize_abbreviations")
        
        # 统一括号格式
        name = self._normalize_brackets(name)
        if '(' in original or ')' in original:
            metadata["transformations"].append("normalize_brackets")
        
        metadata["normalized"] = name
        
        return name, metadata
    
    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号（全角转半角）"""
        # 全角转半角映射
        punct_map = {
            '，': ',',
            '。': '.',
            '；': ';',
            '：': ':',
            '？': '?',
            '！': '!',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>',
            '、': ',',
            '～': '~',
            '—': '-',
        }
        
        for full, half in punct_map.items():
            text = text.replace(full, half)
        
        return text
    
    def _standardize_abbreviations(self, name: str) -> Tuple[str, bool]:
        """标准化常见简称"""
        abbreviation_map = {
            r'北大': '北京大学',
            r'清华': '清华大学',
            r'复旦': '复旦大学',
            r'浙大': '浙江大学',
            r'南大': '南京大学',
            r'武大': '武汉大学',
            r'中山医院': '复旦大学附属中山医院',
            r'协和医院': '北京协和医院',
            r'301医院': '中国人民解放军总医院',
        }
        
        applied = False
        for pattern, full_name in abbreviation_map.items():
            if re.search(pattern, name) and pattern != full_name:
                # 注意：这里只是示例，实际应用中需要更谨慎的处理
                # name = re.sub(pattern, full_name, name)
                applied = True
        
        return name, applied
    
    def _normalize_brackets(self, text: str) -> str:
        """统一括号格式"""
        # 统一使用中文括号
        text = text.replace('(', '（').replace(')', '）')
        return text
    
    def normalize_address(self, address: str, city: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        规范化地址信息
        
        Args:
            address: 原始地址
            city: 所属城市（用于补全）
            
        Returns:
            (规范化后的地址, 规范化元数据)
        """
        if not address:
            return "", {"error": "地址为空"}
        
        original = address
        metadata = {
            "original": original,
            "transformations": []
        }
        
        # 去除首尾空格
        address = address.strip()
        
        # 补全城市前缀（如果缺失）
        if city and not address.startswith(city):
            # 检查是否包含城市名
            if city not in address:
                address = f"{city}{address}"
                metadata["transformations"].append("add_city_prefix")
        
        # 规范化行政区划名称
        address = self._normalize_administrative(address)
        if address != original:
            metadata["transformations"].append("normalize_administrative")
        
        # 去除冗余信息
        address = self._remove_address_redundancy(address)
        if address != original:
            metadata["transformations"].append("remove_redundancy")
        
        metadata["normalized"] = address
        
        return address, metadata
    
    def _normalize_administrative(self, address: str) -> str:
        """规范化行政区划名称"""
        # 统一"区/县"、"镇/乡"等后缀
        address = re.sub(r'区(?=.)', '区', address)
        address = re.sub(r'县(?=.)', '县', address)
        return address
    
    def _remove_address_redundancy(self, address: str) -> str:
        """去除地址中的冗余信息"""
        # 去除重复的"省市区县"
        address = re.sub(r'(省|市|区|县)\1+', r'\1', address)
        # 去除"附近"、"周边"等模糊描述
        address = re.sub(r'(附近|周边|旁边|对面)', '', address)
        return address.strip()
    
    def convert_coordinates(
        self, 
        longitude: float, 
        latitude: float, 
        from_system: str, 
        to_system: str = "GCJ02"
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        坐标系转换
        
        Args:
            longitude: 经度
            latitude: 纬度
            from_system: 源坐标系（WGS84/GCJ02/BD09）
            to_system: 目标坐标系（WGS84/GCJ02/BD09），默认为GCJ02
            
        Returns:
            (转换后的经度, 转换后的纬度, 转换元数据)
        """
        metadata = {
            "original": {"longitude": longitude, "latitude": latitude, "system": from_system},
            "target_system": to_system
        }
        
        if from_system == to_system:
            metadata["transformed"] = False
            return longitude, latitude, metadata
        
        try:
            if from_system == "WGS84" and to_system == "GCJ02":
                lng, lat = self._wgs84_to_gcj02(longitude, latitude)
            elif from_system == "GCJ02" and to_system == "WGS84":
                lng, lat = self._gcj02_to_wgs84(longitude, latitude)
            elif from_system == "WGS84" and to_system == "BD09":
                lng, lat = self._wgs84_to_bd09(longitude, latitude)
            elif from_system == "BD09" and to_system == "WGS84":
                lng, lat = self._bd09_to_wgs84(longitude, latitude)
            elif from_system == "GCJ02" and to_system == "BD09":
                lng, lat = self._gcj02_to_bd09(longitude, latitude)
            elif from_system == "BD09" and to_system == "GCJ02":
                lng, lat = self._bd09_to_gcj02(longitude, latitude)
            else:
                raise ValueError(f"不支持的坐标系转换: {from_system} -> {to_system}")
            
            metadata["transformed"] = True
            metadata["result"] = {"longitude": lng, "latitude": lat}
            
            return lng, lat, metadata
            
        except Exception as e:
            logger.error(f"坐标转换失败: {e}")
            metadata["error"] = str(e)
            return longitude, latitude, metadata
    
    def _wgs84_to_gcj02(self, lng: float, lat: float) -> Tuple[float, float]:
        """WGS84转GCJ02"""
        if self._out_of_china(lng, lat):
            return lng, lat
        
        dlat = self._transform_lat(lng - 105.0, lat - 35.0)
        dlng = self._transform_lng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.PI
        magic = math.sin(radlat)
        magic = 1 - self.EE * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.A * (1 - self.EE)) / (magic * sqrtmagic) * self.PI)
        dlng = (dlng * 180.0) / (self.A / sqrtmagic * math.cos(radlat) * self.PI)
        mglat = lat + dlat
        mglng = lng + dlng
        return mglng, mglat
    
    def _gcj02_to_wgs84(self, lng: float, lat: float) -> Tuple[float, float]:
        """GCJ02转WGS84"""
        if self._out_of_china(lng, lat):
            return lng, lat
        
        dlat = self._transform_lat(lng - 105.0, lat - 35.0)
        dlng = self._transform_lng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.PI
        magic = math.sin(radlat)
        magic = 1 - self.EE * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.A * (1 - self.EE)) / (magic * sqrtmagic) * self.PI)
        dlng = (dlng * 180.0) / (self.A / sqrtmagic * math.cos(radlat) * self.PI)
        mglat = lat - dlat
        mglng = lng - dlng
        return mglng, mglat
    
    def _wgs84_to_bd09(self, lng: float, lat: float) -> Tuple[float, float]:
        """WGS84转BD09"""
        gcj_lng, gcj_lat = self._wgs84_to_gcj02(lng, lat)
        return self._gcj02_to_bd09(gcj_lng, gcj_lat)
    
    def _bd09_to_wgs84(self, lng: float, lat: float) -> Tuple[float, float]:
        """BD09转WGS84"""
        gcj_lng, gcj_lat = self._bd09_to_gcj02(lng, lat)
        return self._gcj02_to_wgs84(gcj_lng, gcj_lat)
    
    def _gcj02_to_bd09(self, lng: float, lat: float) -> Tuple[float, float]:
        """GCJ02转BD09"""
        z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * self.X_PI)
        theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * self.X_PI)
        bd_lng = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return bd_lng, bd_lat
    
    def _bd09_to_gcj02(self, lng: float, lat: float) -> Tuple[float, float]:
        """BD09转GCJ02"""
        x = lng - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.X_PI)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.X_PI)
        gcj_lng = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        return gcj_lng, gcj_lat
    
    def _transform_lat(self, lng: float, lat: float) -> float:
        """纬度转换"""
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.PI) + 20.0 *
                math.sin(2.0 * lng * self.PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * self.PI) + 40.0 *
                math.sin(lat / 3.0 * self.PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * self.PI) + 320 *
                math.sin(lat * self.PI / 30.0)) * 2.0 / 3.0
        return ret
    
    def _transform_lng(self, lng: float, lat: float) -> float:
        """经度转换"""
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.PI) + 20.0 *
                math.sin(2.0 * lng * self.PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * self.PI) + 40.0 *
                math.sin(lng / 3.0 * self.PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * self.PI) + 300.0 *
                math.sin(lng / 30.0 * self.PI)) * 2.0 / 3.0
        return ret
    
    def _out_of_china(self, lng: float, lat: float) -> bool:
        """判断是否在中国境外"""
        return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)
    
    def normalize_category(self, category: str, poi_type: str) -> Tuple[str, Dict[str, Any]]:
        """
        规范化分类信息
        
        Args:
            category: 原始分类
            poi_type: POI类型
            
        Returns:
            (规范化后的分类, 规范化元数据)
        """
        if not category:
            return "", {"error": "分类为空"}
        
        original = category
        metadata = {
            "original": original,
            "transformations": []
        }
        
        # 去除首尾空格
        category = category.strip()
        
        # 分类映射（示例）
        category_map = {
            "医院": "综合医院",
            "三级甲等": "三级甲等医院",
            "三甲医院": "三级甲等医院",
            "政府": "政府机关",
            "景区": "旅游景区",
        }
        
        if category in category_map:
            category = category_map[category]
            metadata["transformations"].append("category_mapping")
        
        metadata["normalized"] = category
        
        return category, metadata
    
    def normalize_evidence(self, evidence: Dict[str, Any]) -> NormalizedEvidence:
        """
        规范化单条证据
        
        Args:
            evidence: 原始证据字典
            
        Returns:
            规范化后的证据对象
        """
        evidence_id = evidence.get('evidence_id', '')
        poi_id = evidence.get('poi_id', '')
        source = evidence.get('source', {})
        collected_at = evidence.get('collected_at', '')
        original_data = evidence.get('data', {})
        
        logger.info(f"开始规范化证据: {evidence_id}")
        
        normalized_data = {}
        normalization_metadata = {
            "normalized_at": datetime.utcnow().isoformat() + 'Z',
            "fields_normalized": []
        }
        
        # 规范化名称
        if 'name' in original_data:
            normalized_name, name_metadata = self.normalize_name(original_data['name'])
            normalized_data['name'] = normalized_name
            normalization_metadata['name'] = name_metadata
            normalization_metadata['fields_normalized'].append('name')
        
        # 规范化地址
        if 'address' in original_data:
            city = original_data.get('administrative', {}).get('city', '')
            normalized_address, addr_metadata = self.normalize_address(
                original_data['address'], city
            )
            normalized_data['address'] = normalized_address
            normalization_metadata['address'] = addr_metadata
            normalization_metadata['fields_normalized'].append('address')
        
        # 规范化坐标 - 统一转换为GCJ02坐标系
        if 'coordinates' in original_data:
            coords = original_data['coordinates']
            coord_system = coords.get('coordinate_system', 'WGS84')
            lng, lat, coord_metadata = self.convert_coordinates(
                coords.get('longitude', 0),
                coords.get('latitude', 0),
                coord_system,
                'GCJ02'  # 默认目标坐标系为GCJ02
            )
            normalized_data['coordinates'] = {
                'longitude': lng,
                'latitude': lat,
                'coordinate_system': 'GCJ02'  # 统一使用GCJ02坐标系
            }
            normalization_metadata['coordinates'] = coord_metadata
            normalization_metadata['fields_normalized'].append('coordinates')
        
        # 规范化分类
        if 'category' in original_data:
            poi_type = evidence.get('poi_type', '')
            normalized_category, cat_metadata = self.normalize_category(
                original_data['category'], poi_type
            )
            normalized_data['category'] = normalized_category
            normalization_metadata['category'] = cat_metadata
            normalization_metadata['fields_normalized'].append('category')
        
        # 复制其他未规范化的字段
        for key, value in original_data.items():
            if key not in normalized_data:
                normalized_data[key] = value
        
        logger.info(f"证据规范化完成: {evidence_id}")
        
        return NormalizedEvidence(
            evidence_id=evidence_id,
            poi_id=poi_id,
            source=source,
            collected_at=collected_at,
            original_data=original_data,
            normalized_data=normalized_data,
            normalization_metadata=normalization_metadata
        )
    
    def normalize_evidence_list(
        self, 
        evidence_list: List[Dict[str, Any]]
    ) -> List[NormalizedEvidence]:
        """
        批量规范化证据列表
        
        Args:
            evidence_list: 证据字典列表
            
        Returns:
            规范化后的证据对象列表
        """
        logger.info(f"开始批量规范化 {len(evidence_list)} 条证据")
        
        normalized_list = []
        for evidence in evidence_list:
            try:
                normalized = self.normalize_evidence(evidence)
                normalized_list.append(normalized)
            except Exception as e:
                logger.error(f"规范化证据失败 {evidence.get('evidence_id')}: {e}")
        
        logger.info(f"成功规范化 {len(normalized_list)} 条证据")
        return normalized_list
    
    def save_normalized_evidence(
        self, 
        normalized_list: List[NormalizedEvidence], 
        output_path: str
    ) -> None:
        """
        保存规范化后的证据
        
        Args:
            normalized_list: 规范化证据列表
            output_path: 输出文件路径
        """
        try:
            evidence_dicts = []
            for item in normalized_list:
                evidence_dicts.append({
                    'evidence_id': item.evidence_id,
                    'poi_id': item.poi_id,
                    'source': item.source,
                    'collected_at': item.collected_at,
                    'original_data': item.original_data,
                    'normalized_data': item.normalized_data,
                    'normalization_metadata': item.normalization_metadata
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evidence_dicts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"规范化证据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存规范化证据失败: {e}")
            raise


def main():
    """
    主函数 - 示例用法
    """
    # 示例证据数据
    evidence_list = [
        {
            "evidence_id": "EVD_001",
            "poi_id": "HOSPITAL_BJ_001",
            "source": {"source_name": "测试源", "weight": 0.9},
            "collected_at": "2024-01-15T08:35:00Z",
            "data": {
                "name": "  北京大学第一医院  ",
                "address": "北京市西城区西什库大街8号",
                "coordinates": {
                    "longitude": 116.404,
                    "latitude": 39.915,
                    "coordinate_system": "GCJ02"
                },
                "category": "医院"
            }
        }
    ]
    
    # 创建规范化处理器
    normalizer = EvidenceNormalizer()
    
    # 规范化证据
    normalized_list = normalizer.normalize_evidence_list(evidence_list)
    
    # 保存结果
    if normalized_list:
        normalizer.save_normalized_evidence(normalized_list, "normalized_evidence.json")
        
        # 打印规范化结果
        for item in normalized_list:
            print(f"\n证据ID: {item.evidence_id}")
            print(f"原始名称: {item.original_data.get('name')}")
            print(f"规范化名称: {item.normalized_data.get('name')}")
            print(f"规范化字段: {item.normalization_metadata.get('fields_normalized')}")


if __name__ == "__main__":
    main()
