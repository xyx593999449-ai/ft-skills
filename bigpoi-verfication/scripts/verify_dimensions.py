#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.2
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的维度级核实判断阶段，对规范化后的证据数据进行多维度验证检查。
    按照SKILL.md定义的6个核实维度（存在性、名称准确性、空间位置、分类正确性、行政区匹配、时效性风险），
    逐一评估POI数据在各维度上的真实性与准确性，并输出结构化的维度判断结果。

用途说明:
    该脚本在整个核实流程中处于第四阶段（维度级核实判断），主要作用包括：
    1. 存在性验证：判断POI实体是否真实存在（基于证据数量、来源权威性等）
    2. 名称准确性验证：比较输入名称与证据名称的相似度，判断是否一致
    3. 空间位置验证：计算输入坐标与证据坐标的距离，判断是否在合理范围内
    4. 分类正确性验证：判断POI分类是否与实体属性匹配
    5. 行政区匹配验证：判断POI是否落在合理的行政区范围内
    6. 时效性风险评估：检查是否存在已公开的撤销、合并、迁址等变更事实
    7. 为每个维度输出判断结果（pass/fail/uncertain）、置信度和证据引用

    应用场景：多维度数据质量评估、为统一决策提供维度级判断依据
"""

import json
import yaml
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DimensionResult:
    """维度验证结果数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        dimension: str,  # 维度名称
        result: str,  # pass / fail / uncertain
        confidence: float,  # 0.0 ~ 1.0
        score: float,  # 维度评分
        evidence_refs: List[str] = None,
        details: Dict[str, Any] = None
    ):
        self.dimension = dimension
        self.result = result
        self.confidence = confidence
        self.score = score
        self.evidence_refs = evidence_refs if evidence_refs is not None else []
        self.details = details if details is not None else {}

    def __repr__(self):
        return (f"DimensionResult(dimension={self.dimension!r}, "
                f"result={self.result!r}, confidence={self.confidence:.4f})")


class DimensionVerifier:
    """维度验证器类"""
    
    def __init__(self, thresholds_path: str = "../config/thresholds.yaml"):
        """
        初始化维度验证器
        
        Args:
            thresholds_path: 阈值配置文件路径
        """
        self.thresholds = self._load_thresholds(thresholds_path)
        logger.info("维度验证器初始化完成")
    
    def _load_thresholds(self, thresholds_path: str) -> Dict[str, Any]:
        """
        加载阈值配置
        
        Args:
            thresholds_path: 阈值配置文件路径
            
        Returns:
            阈值配置字典
        """
        try:
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds = yaml.safe_load(f)
            logger.info(f"成功加载阈值配置: {thresholds_path}")
            return thresholds
        except Exception as e:
            logger.warning(f"加载阈值配置失败，使用默认配置: {e}")
            return self._get_default_thresholds()
    
    def _get_default_thresholds(self) -> Dict[str, Any]:
        """获取默认阈值配置"""
        return {
            'existence_threshold': {
                'min_evidence_count': 2,
                'min_high_weight_sources': 1,
                'min_overall_confidence': 0.6
            },
            'name_threshold': {
                'exact_match': 1.0,
                'high_similarity': 0.85,
                'medium_similarity': 0.7,
                'low_similarity': 0.5
            },
            'distance_threshold': {
                'exact_match': 0,
                'high_precision': 10,
                'standard': 50,
                'relaxed': 100,
                'wide_range': 500,
                'mismatch': 500
            }
        }
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        计算两个名称的相似度
        
        Args:
            name1: 第一个名称
            name2: 第二个名称
            
        Returns:
            相似度（0.0 ~ 1.0）
        """
        if not name1 or not name2:
            return 0.0
        
        # 使用SequenceMatcher计算相似度
        similarity = SequenceMatcher(None, name1, name2).ratio()
        return round(similarity, 4)
    
    def _calculate_distance(
        self, 
        lng1: float, 
        lat1: float, 
        lng2: float, 
        lat2: float
    ) -> float:
        """
        计算两点间的距离（米）
        
        Args:
            lng1: 第一点经度
            lat1: 第一点纬度
            lng2: 第二点经度
            lat2: 第二点纬度
            
        Returns:
            距离（米）
        """
        # 使用Haversine公式计算球面距离
        R = 6371000  # 地球半径（米）
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = math.sin(delta_lat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return round(distance, 2)
    
    def verify_existence(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> DimensionResult:
        """
        验证存在性维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            存在性验证结果
        """
        logger.info("开始验证存在性维度")
        
        thresholds = self.thresholds.get('existence_threshold', {})
        min_evidence = thresholds.get('min_evidence_count', 2)
        min_high_weight = thresholds.get('min_high_weight_sources', 1)
        min_confidence = thresholds.get('min_overall_confidence', 0.6)
        
        # 统计有效证据
        valid_evidence = [e for e in evidence_list if e.get('verification', {}).get('is_valid', False)]
        evidence_count = len(valid_evidence)
        
        # 统计高权重证据
        high_weight_count = sum(
            1 for e in valid_evidence 
            if e.get('source', {}).get('weight', 0) >= 0.8
        )
        
        # 计算平均置信度
        avg_confidence = 0.0
        if valid_evidence:
            avg_confidence = sum(
                e.get('verification', {}).get('confidence', 0) for e in valid_evidence
            ) / len(valid_evidence)
        
        # 判断结果
        if evidence_count >= min_evidence and high_weight_count >= min_high_weight and avg_confidence >= min_confidence:
            result = "pass"
            confidence = min(avg_confidence * 1.1, 1.0)  # 给予一定奖励
        elif evidence_count >= min_evidence and avg_confidence >= min_confidence * 0.8:
            result = "uncertain"
            confidence = avg_confidence
        else:
            result = "fail"
            confidence = avg_confidence * 0.8
        
        evidence_refs = [e.get('evidence_id') for e in valid_evidence]
        
        return DimensionResult(
            dimension="existence",
            result=result,
            confidence=round(confidence, 4),
            score=round(avg_confidence, 4),
            evidence_refs=evidence_refs,
            details={
                "evidence_count": evidence_count,
                "high_weight_count": high_weight_count,
                "avg_confidence": round(avg_confidence, 4)
            }
        )
    
    def verify_name(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> DimensionResult:
        """
        验证名称准确性维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            名称验证结果
        """
        logger.info("开始验证名称准确性维度")
        
        input_name = poi_data.get('name', '')
        thresholds = self.thresholds.get('name_threshold', {})
        exact_threshold = thresholds.get('exact_match', 1.0)
        high_threshold = thresholds.get('high_similarity', 0.85)
        medium_threshold = thresholds.get('medium_similarity', 0.7)
        
        if not input_name:
            return DimensionResult(
                dimension="name",
                result="fail",
                confidence=0.0,
                score=0.0,
                details={"error": "输入名称为空"}
            )
        
        # 计算与每条证据的相似度
        similarities = []
        evidence_refs = []
        
        for evidence in evidence_list:
            normalized_data = evidence.get('normalized_data', {})
            evidence_name = normalized_data.get('name', '')
            
            if evidence_name:
                similarity = self._calculate_name_similarity(input_name, evidence_name)
                similarities.append(similarity)
                evidence_refs.append(evidence.get('evidence_id'))
        
        if not similarities:
            return DimensionResult(
                dimension="name",
                result="uncertain",
                confidence=0.3,
                score=0.0,
                details={"error": "无有效证据名称"}
            )
        
        # 取最高相似度
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)
        
        # 判断结果
        if max_similarity >= exact_threshold:
            result = "pass"
            confidence = 1.0
        elif max_similarity >= high_threshold:
            result = "pass"
            confidence = max_similarity
        elif max_similarity >= medium_threshold:
            result = "uncertain"
            confidence = max_similarity
        else:
            result = "fail"
            confidence = max_similarity * 0.8
        
        return DimensionResult(
            dimension="name",
            result=result,
            confidence=round(confidence, 4),
            score=round(max_similarity, 4),
            evidence_refs=evidence_refs,
            details={
                "input_name": input_name,
                "max_similarity": round(max_similarity, 4),
                "avg_similarity": round(avg_similarity, 4),
                "matched_name": evidence_list[similarities.index(max_similarity)].get('normalized_data', {}).get('name', '')
            }
        )
    
    def verify_location(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> DimensionResult:
        """
        验证空间位置维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            位置验证结果
        """
        logger.info("开始验证空间位置维度")
        
        input_coords = poi_data.get('coordinates', {})
        input_lng = input_coords.get('longitude')
        input_lat = input_coords.get('latitude')
        
        thresholds = self.thresholds.get('distance_threshold', {})
        high_precision = thresholds.get('high_precision', 10)
        standard = thresholds.get('standard', 50)
        relaxed = thresholds.get('relaxed', 100)
        mismatch = thresholds.get('mismatch', 500)
        
        if input_lng is None or input_lat is None:
            return DimensionResult(
                dimension="location",
                result="uncertain",
                confidence=0.5,
                score=0.5,
                details={"error": "输入坐标缺失"}
            )
        
        # 计算与每条证据的距离
        distances = []
        evidence_refs = []
        
        for evidence in evidence_list:
            normalized_data = evidence.get('normalized_data', {})
            evidence_coords = normalized_data.get('coordinates', {})
            evidence_lng = evidence_coords.get('longitude')
            evidence_lat = evidence_coords.get('latitude')
            
            if evidence_lng is not None and evidence_lat is not None:
                distance = self._calculate_distance(
                    input_lng, input_lat, evidence_lng, evidence_lat
                )
                distances.append(distance)
                evidence_refs.append(evidence.get('evidence_id'))
        
        if not distances:
            return DimensionResult(
                dimension="location",
                result="uncertain",
                confidence=0.3,
                score=0.0,
                details={"error": "无有效证据坐标"}
            )
        
        # 取最小距离
        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)
        
        # 根据距离判断结果
        if min_distance <= high_precision:
            result = "pass"
            confidence = 1.0
        elif min_distance <= standard:
            result = "pass"
            confidence = 0.9
        elif min_distance <= relaxed:
            result = "uncertain"
            confidence = 0.7
        elif min_distance <= mismatch:
            result = "uncertain"
            confidence = 0.4
        else:
            result = "fail"
            confidence = 0.2
        
        return DimensionResult(
            dimension="location",
            result=result,
            confidence=round(confidence, 4),
            score=round(1 - min_distance / mismatch, 4),
            evidence_refs=evidence_refs,
            details={
                "input_coordinates": {"longitude": input_lng, "latitude": input_lat, "coordinate_system": "GCJ02"},
                "min_distance": min_distance,
                "avg_distance": round(avg_distance, 2),
                "coordinate_system": "GCJ02"
            }
        )
    
    def verify_category(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> DimensionResult:
        """
        验证分类正确性维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            分类验证结果
        """
        logger.info("开始验证分类正确性维度")
        
        input_category = poi_data.get('category_code', '')
        poi_type = poi_data.get('poi_type', '')
        
        # 简单的分类匹配逻辑
        category_matches = []
        evidence_refs = []
        
        for evidence in evidence_list:
            normalized_data = evidence.get('normalized_data', {})
            evidence_category = normalized_data.get('category', '')
            
            if evidence_category:
                # 计算分类相似度
                similarity = self._calculate_name_similarity(
                    input_category, evidence_category
                )
                category_matches.append(similarity)
                evidence_refs.append(evidence.get('evidence_id'))
        
        if not category_matches:
            return DimensionResult(
                dimension="category",
                result="uncertain",
                confidence=0.5,
                score=0.5,
                details={"error": "无有效证据分类"}
            )
        
        max_match = max(category_matches)
        
        if max_match >= 0.8:
            result = "pass"
            confidence = max_match
        elif max_match >= 0.5:
            result = "uncertain"
            confidence = max_match
        else:
            result = "fail"
            confidence = max_match * 0.8
        
        return DimensionResult(
            dimension="category",
            result=result,
            confidence=round(confidence, 4),
            score=round(max_match, 4),
            evidence_refs=evidence_refs,
            details={
                "input_category": input_category,
                "max_match": round(max_match, 4)
            }
        )
    
    def verify_administrative(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> DimensionResult:
        """
        验证行政区匹配维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            行政区验证结果
        """
        logger.info("开始验证行政区匹配维度")
        
        input_city = poi_data.get('city', '')
        
        # 检查证据中的城市是否匹配
        city_matches = []
        evidence_refs = []
        
        for evidence in evidence_list:
            normalized_data = evidence.get('normalized_data', {})
            admin_data = normalized_data.get('administrative', {})
            evidence_city = admin_data.get('city', '')
            
            if evidence_city:
                match = 1.0 if input_city in evidence_city or evidence_city in input_city else 0.0
                city_matches.append(match)
                evidence_refs.append(evidence.get('evidence_id'))
        
        if not city_matches:
            return DimensionResult(
                dimension="administrative",
                result="uncertain",
                confidence=0.5,
                score=0.5,
                details={"error": "无有效行政区信息"}
            )
        
        match_rate = sum(city_matches) / len(city_matches)
        
        if match_rate >= 0.8:
            result = "pass"
            confidence = match_rate
        elif match_rate >= 0.5:
            result = "uncertain"
            confidence = match_rate
        else:
            result = "fail"
            confidence = match_rate
        
        return DimensionResult(
            dimension="administrative",
            result=result,
            confidence=round(confidence, 4),
            score=round(match_rate, 4),
            evidence_refs=evidence_refs,
            details={
                "input_city": input_city,
                "match_rate": round(match_rate, 4)
            }
        )
    
    def verify_timeliness(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> DimensionResult:
        """
        验证时效性风险维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            时效性验证结果
        """
        logger.info("开始验证时效性风险维度")
        
        # 检查证据中的状态信息
        status_list = []
        evidence_refs = []
        
        for evidence in evidence_list:
            normalized_data = evidence.get('normalized_data', {})
            status = normalized_data.get('status', 'unknown')
            status_list.append(status)
            evidence_refs.append(evidence.get('evidence_id'))
        
        if not status_list:
            return DimensionResult(
                dimension="timeliness",
                result="uncertain",
                confidence=0.5,
                score=0.5,
                details={"error": "无时效性信息"}
            )
        
        # 统计状态
        active_count = status_list.count('active')
        revoked_count = status_list.count('revoked')
        total_count = len(status_list)
        
        # 计算风险评分
        if revoked_count > 0:
            risk_score = revoked_count / total_count
            result = "fail"
            confidence = 1 - risk_score
        elif active_count == total_count:
            result = "pass"
            confidence = 0.9
        else:
            result = "uncertain"
            confidence = 0.6
        
        return DimensionResult(
            dimension="timeliness",
            result=result,
            confidence=round(confidence, 4),
            score=round(confidence, 4),
            evidence_refs=evidence_refs,
            details={
                "status_distribution": {
                    "active": active_count,
                    "revoked": revoked_count,
                    "other": total_count - active_count - revoked_count
                }
            }
        )
    
    def verify_all_dimensions(
        self, 
        poi_data: Dict[str, Any], 
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, DimensionResult]:
        """
        验证所有维度
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            
        Returns:
            各维度验证结果字典
        """
        logger.info(f"开始验证所有维度: {poi_data.get('poi_id')}")
        
        results = {
            "existence": self.verify_existence(poi_data, evidence_list),
            "name": self.verify_name(poi_data, evidence_list),
            "location": self.verify_location(poi_data, evidence_list),
            "category": self.verify_category(poi_data, evidence_list),
            "administrative": self.verify_administrative(poi_data, evidence_list),
            "timeliness": self.verify_timeliness(poi_data, evidence_list)
        }
        
        logger.info("所有维度验证完成")
        return results
    
    def dimension_result_to_dict(self, result: DimensionResult) -> Dict[str, Any]:
        """
        将维度结果转换为字典
        
        Args:
            result: 维度结果对象
            
        Returns:
            结果字典
        """
        return {
            "dimension": result.dimension,
            "result": result.result,
            "confidence": result.confidence,
            "score": result.score,
            "evidence_refs": result.evidence_refs,
            "details": result.details
        }
    
    def save_dimension_results(
        self, 
        results: Dict[str, DimensionResult], 
        output_path: str
    ) -> None:
        """
        保存维度验证结果
        
        Args:
            results: 维度结果字典
            output_path: 输出文件路径
        """
        try:
            results_dict = {
                dim: self.dimension_result_to_dict(result)
                for dim, result in results.items()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"维度验证结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存维度验证结果失败: {e}")
            raise


def main():
    """
    主函数 - 示例用法
    """
    # 示例POI数据
    poi_data = {
        "poi_id": "HOSPITAL_BJ_001",
        "name": "北京大学第一医院",
        "poi_type": "hospital",
        "city": "北京市",
        "coordinates": {
            "longitude": 116.3723,
            "latitude": 39.9342
        },
        "category_code": "090100"
    }
    
    # 示例证据数据
    evidence_list = [
        {
            "evidence_id": "EVD_001",
            "normalized_data": {
                "name": "北京大学第一医院",
                "coordinates": {"longitude": 116.3725, "latitude": 39.9340},
                "category": "综合医院",
                "administrative": {"city": "北京市"},
                "status": "active"
            },
            "verification": {"is_valid": True, "confidence": 0.95},
            "source": {"weight": 1.0}
        },
        {
            "evidence_id": "EVD_002",
            "normalized_data": {
                "name": "北大第一医院",
                "coordinates": {"longitude": 116.3720, "latitude": 39.9345},
                "category": "医院",
                "administrative": {"city": "北京市"},
                "status": "active"
            },
            "verification": {"is_valid": True, "confidence": 0.85},
            "source": {"weight": 0.8}
        }
    ]
    
    # 创建验证器
    verifier = DimensionVerifier()
    
    # 验证所有维度
    results = verifier.verify_all_dimensions(poi_data, evidence_list)
    
    # 保存结果
    verifier.save_dimension_results(results, "dimension_results.json")
    
    # 打印结果
    for dim, result in results.items():
        print(f"\n维度: {dim}")
        print(f"  结果: {result.result}")
        print(f"  置信度: {result.confidence}")
        print(f"  评分: {result.score}")


if __name__ == "__main__":
    main()
