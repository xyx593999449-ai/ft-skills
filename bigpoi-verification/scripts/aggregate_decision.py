#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.0
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的统一决策与降级判断阶段，汇总各维度验证结果并生成最终的核实决策。
    基于维度级判断结果，结合降级策略配置，评估是否触发降级条件，计算整体置信度，
    并生成符合schema/decision.schema.json标准的结构化决策结果。

用途说明:
    该脚本在整个核实流程中处于第五阶段（统一决策与降级判断），主要作用包括：
    1. 汇总各维度验证结果，计算整体置信度
    2. 根据config/downgrade.yaml配置的降级策略，判断是否触发降级条件
    3. 评估多源信息冲突情况，决定是否需要人工复核
    4. 生成建议的修正内容（名称、地址、坐标等）
    5. 确定最终核实状态（accepted/downgraded/manual_review/rejected）
    6. 生成决策摘要和后续处理建议
    7. 输出符合标准的决策结果，支持审计和责任追溯

    应用场景：批量POI数据核实决策、降级策略执行、人工复核任务分配
"""

import json
import yaml
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecisionResult:
    """决策结果数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        decision_id: str,
        poi_id: str,
        overall: Dict[str, Any],
        dimensions: Dict[str, Any],
        evidence_summary: Dict[str, Any],
        downgrade_info: Optional[Dict[str, Any]],
        corrections: Optional[Dict[str, Any]],
        created_at: str,
        processed_at: str,
        processing_duration_ms: int,
        version: str = "1.0.0"
    ):
        self.decision_id = decision_id
        self.poi_id = poi_id
        self.overall = overall
        self.dimensions = dimensions
        self.evidence_summary = evidence_summary
        self.downgrade_info = downgrade_info
        self.corrections = corrections
        self.created_at = created_at
        self.processed_at = processed_at
        self.processing_duration_ms = processing_duration_ms
        self.version = version

    def __repr__(self):
        return (f"DecisionResult(decision_id={self.decision_id!r}, "
                f"poi_id={self.poi_id!r}, version={self.version!r})")


class DecisionAggregator:
    """决策聚合器类"""
    
    def __init__(
        self, 
        downgrade_config_path: str = "../config/downgrade.yaml"
    ):
        """
        初始化决策聚合器
        
        Args:
            downgrade_config_path: 降级策略配置文件路径
        """
        self.downgrade_config = self._load_downgrade_config(downgrade_config_path)
        logger.info("决策聚合器初始化完成")
    
    def _load_downgrade_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载降级策略配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            降级策略配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载降级策略配置: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载降级策略配置失败，使用默认配置: {e}")
            return self._get_default_downgrade_config()
    
    def _get_default_downgrade_config(self) -> Dict[str, Any]:
        """获取默认降级策略配置"""
        return {
            'global': {
                'enabled': True,
                'default_action': 'manual_review'
            },
            'triggers': [
                {
                    'id': 'LOW_OVERALL_CONFIDENCE',
                    'enabled': True,
                    'criteria': {
                        'metric': 'overall_confidence',
                        'operator': '<',
                        'threshold': 0.6
                    }
                }
            ]
        }
    
    def _generate_decision_id(self, poi_id: str) -> str:
        """
        生成决策唯一标识符
        
        Args:
            poi_id: POI标识符
            
        Returns:
            决策ID
        """
        import hashlib
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        base = f"{poi_id}_{timestamp}"
        return f"DEC_{hashlib.md5(base.encode()).hexdigest()[:16].upper()}"
    
    def _calculate_overall_confidence(
        self, 
        dimension_results: Dict[str, Any]
    ) -> float:
        """
        计算整体置信度
        
        Args:
            dimension_results: 维度结果字典
            
        Returns:
            整体置信度（0.0 ~ 1.0）
        """
        # 定义维度权重
        dimension_weights = {
            "existence": 0.25,
            "name": 0.25,
            "location": 0.20,
            "category": 0.15,
            "administrative": 0.10,
            "timeliness": 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dim, result in dimension_results.items():
            weight = dimension_weights.get(dim, 0.1)
            confidence = result.get('confidence', 0.0)
            weighted_sum += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        overall_confidence = weighted_sum / total_weight
        return round(min(overall_confidence, 1.0), 4)
    
    def _check_downgrade_triggers(
        self, 
        dimension_results: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        overall_confidence: float
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        检查降级触发条件
        
        Args:
            dimension_results: 维度结果字典
            evidence_list: 证据列表
            overall_confidence: 整体置信度
            
        Returns:
            (是否降级, 触发的条件列表, 降级原因描述)
        """
        global_config = self.downgrade_config.get('global', {})
        
        # 检查全局降级开关
        if not global_config.get('enabled', True):
            return False, [], "降级策略已关闭"
        
        triggers = self.downgrade_config.get('triggers', [])
        triggered_conditions = []
        
        for trigger in triggers:
            if not trigger.get('enabled', True):
                continue
            
            trigger_id = trigger.get('id', '')
            criteria = trigger.get('criteria', {})
            
            # 检查触发条件
            is_triggered = self._evaluate_trigger_criteria(
                trigger_id, criteria, dimension_results, 
                evidence_list, overall_confidence
            )
            
            if is_triggered:
                triggered_conditions.append({
                    'id': trigger_id,
                    'name': trigger.get('name', ''),
                    'description': trigger.get('description', ''),
                    'severity': trigger.get('severity', 'medium')
                })
        
        is_downgraded = len(triggered_conditions) > 0
        
        # 生成降级原因描述
        if is_downgraded:
            reason = "; ".join([
                f"{c['name']}({c['id']})" 
                for c in triggered_conditions[:3]  # 最多显示3个原因
            ])
        else:
            reason = ""
        
        return is_downgraded, triggered_conditions, reason
    
    def _evaluate_trigger_criteria(
        self,
        trigger_id: str,
        criteria: Dict[str, Any],
        dimension_results: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        overall_confidence: float
    ) -> bool:
        """
        评估单个触发条件
        
        Args:
            trigger_id: 触发条件ID
            criteria: 触发条件配置
            dimension_results: 维度结果
            evidence_list: 证据列表
            overall_confidence: 整体置信度
            
        Returns:
            是否触发
        """
        metric = criteria.get('metric', '')
        operator = criteria.get('operator', '')
        threshold = criteria.get('threshold', 0)
        
        # 获取指标值
        if metric == 'overall_confidence':
            value = overall_confidence
        elif metric.startswith('dimension_'):
            dim_name = metric.replace('dimension_', '')
            dim_result = dimension_results.get(dim_name, {})
            value = dim_result.get('confidence', 0)
        elif metric == 'evidence_count':
            value = len(evidence_list)
        elif metric == 'conflict_score':
            value = self._calculate_conflict_score(dimension_results)
        else:
            return False
        
        # 比较操作
        if operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        
        return False
    
    def _calculate_conflict_score(
        self, 
        dimension_results: Dict[str, Any]
    ) -> float:
        """
        计算多源冲突评分
        
        Args:
            dimension_results: 维度结果
            
        Returns:
            冲突评分（0.0 ~ 1.0，越高表示冲突越严重）
        """
        fail_count = sum(
            1 for r in dimension_results.values() 
            if r.get('result') == 'fail'
        )
        uncertain_count = sum(
            1 for r in dimension_results.values() 
            if r.get('result') == 'uncertain'
        )
        total_count = len(dimension_results)
        
        if total_count == 0:
            return 0.0
        
        # 失败和不确定都视为冲突
        conflict_score = (fail_count * 1.0 + uncertain_count * 0.5) / total_count
        return round(conflict_score, 4)
    
    def _determine_action(self, status: str) -> str:
        """
        根据状态确定建议动作
        
        Args:
            status: 核实状态
            
        Returns:
            建议动作
        """
        action_map = {
            'accepted': 'adopt',
            'downgraded': 'manual_review',
            'manual_review': 'manual_review',
            'rejected': 'reject'
        }
        return action_map.get(status, 'manual_review')
    
    def _generate_corrections(
        self, 
        poi_data: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        dimension_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成建议的修正内容
        
        Args:
            poi_data: POI输入数据
            evidence_list: 证据列表
            dimension_results: 维度结果
            
        Returns:
            修正建议字典
        """
        corrections = {}
        
        # 名称修正建议
        name_result = dimension_results.get('name', {})
        if name_result.get('result') in ['fail', 'uncertain']:
            name_details = name_result.get('details', {})
            matched_name = name_details.get('matched_name', '')
            if matched_name:
                corrections['name'] = {
                    'original': poi_data.get('name', ''),
                    'suggested': matched_name,
                    'confidence': name_result.get('confidence', 0)
                }
        
        # 地址修正建议
        # 基于证据中的地址信息生成建议
        if evidence_list:
            best_evidence = max(
                evidence_list, 
                key=lambda e: e.get('source', {}).get('weight', 0)
            )
            normalized_data = best_evidence.get('normalized_data', {})
            suggested_address = normalized_data.get('address', '')
            
            if suggested_address and suggested_address != poi_data.get('address', ''):
                corrections['address'] = {
                    'original': poi_data.get('address', ''),
                    'suggested': suggested_address,
                    'confidence': best_evidence.get('source', {}).get('weight', 0)
                }
        
        # 坐标修正建议
        location_result = dimension_results.get('location', {})
        if location_result.get('result') in ['fail', 'uncertain']:
            # 选择最可信的证据坐标
            if evidence_list:
                best_evidence = max(
                    evidence_list,
                    key=lambda e: e.get('source', {}).get('weight', 0)
                )
                normalized_data = best_evidence.get('normalized_data', {})
                suggested_coords = normalized_data.get('coordinates', {})
                
                if suggested_coords:
                    # 确保建议的坐标包含GCJ02坐标系信息
                    suggested_coords_with_system = dict(suggested_coords)
                    if 'coordinate_system' not in suggested_coords_with_system:
                        suggested_coords_with_system['coordinate_system'] = 'GCJ02'
                    corrections['coordinates'] = {
                        'original': poi_data.get('coordinates', {}),
                        'suggested': suggested_coords_with_system,
                        'confidence': best_evidence.get('source', {}).get('weight', 0)
                    }
        
        # 分类修正建议
        category_result = dimension_results.get('category', {})
        if category_result.get('result') in ['fail', 'uncertain']:
            if evidence_list:
                best_evidence = max(
                    evidence_list,
                    key=lambda e: e.get('source', {}).get('weight', 0)
                )
                normalized_data = best_evidence.get('normalized_data', {})
                suggested_category = normalized_data.get('category', '')
                
                if suggested_category:
                    corrections['category'] = {
                        'original': poi_data.get('category_code', ''),
                        'suggested': suggested_category,
                        'confidence': category_result.get('confidence', 0)
                    }
        
        return corrections
    
    def _generate_evidence_summary(
        self, 
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成证据汇总信息
        
        Args:
            evidence_list: 证据列表
            
        Returns:
            证据汇总字典
        """
        total_count = len(evidence_list)
        valid_count = sum(
            1 for e in evidence_list 
            if e.get('verification', {}).get('is_valid', False)
        )
        high_weight_count = sum(
            1 for e in evidence_list 
            if e.get('source', {}).get('weight', 0) >= 0.8
        )
        
        # 来源分布
        source_distribution = {"official": 0, "map_vendor": 0, "internet": 0}
        for evidence in evidence_list:
            source_type = evidence.get('source', {}).get('source_type', 'other')
            if source_type in source_distribution:
                source_distribution[source_type] += 1
        
        return {
            "total_count": total_count,
            "valid_count": valid_count,
            "high_weight_count": high_weight_count,
            "source_distribution": source_distribution
        }
    
    def aggregate_decision(
        self,
        poi_data: Dict[str, Any],
        dimension_results: Dict[str, Any],
        evidence_list: List[Dict[str, Any]]
    ) -> DecisionResult:
        """
        聚合决策
        
        Args:
            poi_data: POI输入数据
            dimension_results: 维度验证结果
            evidence_list: 证据列表
            
        Returns:
            决策结果对象
        """
        start_time = datetime.utcnow()
        poi_id = poi_data.get('poi_id', 'unknown')
        
        logger.info(f"开始聚合决策: {poi_id}")
        
        # 生成决策ID
        decision_id = self._generate_decision_id(poi_id)
        
        # 计算整体置信度
        overall_confidence = self._calculate_overall_confidence(dimension_results)
        
        # 检查降级触发条件
        is_downgraded, triggered_conditions, downgrade_reason = self._check_downgrade_triggers(
            dimension_results, evidence_list, overall_confidence
        )
        
        # 确定核实状态
        if is_downgraded:
            # 根据触发条件的严重程度确定状态
            has_critical = any(
                c.get('severity') == 'critical' 
                for c in triggered_conditions
            )
            if has_critical:
                status = "rejected"
            else:
                status = "manual_review"
        else:
            # 检查是否有失败的维度
            has_fail = any(
                r.get('result') == 'fail' 
                for r in dimension_results.values()
            )
            if has_fail:
                status = "downgraded"
            else:
                status = "accepted"
        
        # 确定建议动作
        action = self._determine_action(status)
        
        # 生成决策摘要
        if status == "accepted":
            summary = f"高置信度通过，整体置信度{overall_confidence}，所有关键维度均符合要求"
        elif status == "downgraded":
            summary = f"部分维度未通过验证，整体置信度{overall_confidence}，建议降级处理"
        elif status == "manual_review":
            summary = f"触发降级条件: {downgrade_reason}，建议转人工复核"
        else:
            summary = f"存在严重问题，整体置信度{overall_confidence}，建议拒绝"
        
        # 构建整体决策
        overall = {
            "status": status,
            "confidence": overall_confidence,
            "action": action,
            "summary": summary
        }
        
        # 构建降级信息
        downgrade_info = None
        if is_downgraded:
            downgrade_info = {
                "is_downgraded": True,
                "reason_code": triggered_conditions[0].get('id', 'UNKNOWN') if triggered_conditions else 'UNKNOWN',
                "reason_description": downgrade_reason,
                "trigger_conditions": [c.get('id') for c in triggered_conditions],
                "recommendation": "建议人工复核后确定最终处理方案"
            }
        
        # 生成修正建议
        corrections = self._generate_corrections(
            poi_data, evidence_list, dimension_results
        ) if status != "accepted" else None
        
        # 生成证据汇总
        evidence_summary = self._generate_evidence_summary(evidence_list)
        
        # 计算处理耗时
        processed_at = datetime.utcnow()
        duration_ms = int((processed_at - start_time).total_seconds() * 1000)
        
        logger.info(f"决策聚合完成: {decision_id}, 状态: {status}")
        
        return DecisionResult(
            decision_id=decision_id,
            poi_id=poi_id,
            overall=overall,
            dimensions=dimension_results,
            evidence_summary=evidence_summary,
            downgrade_info=downgrade_info,
            corrections=corrections if corrections else None,
            created_at=start_time.isoformat() + 'Z',
            processed_at=processed_at.isoformat() + 'Z',
            processing_duration_ms=duration_ms
        )
    
    def decision_to_dict(self, decision: DecisionResult) -> Dict[str, Any]:
        """
        将决策结果转换为字典
        
        Args:
            decision: 决策结果对象
            
        Returns:
            决策字典
        """
        return {
            "decision_id": decision.decision_id,
            "poi_id": decision.poi_id,
            "overall": decision.overall,
            "dimensions": decision.dimensions,
            "evidence_summary": decision.evidence_summary,
            "downgrade_info": decision.downgrade_info,
            "corrections": decision.corrections,
            "created_at": decision.created_at,
            "processed_at": decision.processed_at,
            "processing_duration_ms": decision.processing_duration_ms,
            "version": decision.version
        }
    
    def save_decision(
        self, 
        decision: DecisionResult, 
        output_path: str
    ) -> None:
        """
        保存决策结果
        
        Args:
            decision: 决策结果对象
            output_path: 输出文件路径
        """
        try:
            decision_dict = self.decision_to_dict(decision)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(decision_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"决策结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存决策结果失败: {e}")
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
        "address": "北京市西城区西什库大街8号",
        "coordinates": {
            "longitude": 116.3723,
            "latitude": 39.9342
        },
        "category_code": "090100"
    }
    
    # 示例维度结果
    dimension_results = {
        "existence": {
            "result": "pass",
            "confidence": 0.95,
            "score": 0.95,
            "evidence_refs": ["EVD_001", "EVD_002"]
        },
        "name": {
            "result": "pass",
            "confidence": 1.0,
            "score": 1.0,
            "evidence_refs": ["EVD_001"]
        },
        "location": {
            "result": "pass",
            "confidence": 0.90,
            "score": 0.90,
            "evidence_refs": ["EVD_001", "EVD_002"]
        },
        "category": {
            "result": "pass",
            "confidence": 0.95,
            "score": 0.95,
            "evidence_refs": ["EVD_001"]
        },
        "administrative": {
            "result": "pass",
            "confidence": 1.0,
            "score": 1.0,
            "evidence_refs": ["EVD_001", "EVD_002"]
        },
        "timeliness": {
            "result": "pass",
            "confidence": 0.90,
            "score": 0.90,
            "evidence_refs": ["EVD_001", "EVD_002"]
        }
    }
    
    # 示例证据列表
    evidence_list = [
        {
            "evidence_id": "EVD_001",
            "source": {"weight": 1.0, "source_type": "official"},
            "normalized_data": {
                "name": "北京大学第一医院",
                "address": "北京市西城区西什库大街8号",
                "coordinates": {"longitude": 116.3725, "latitude": 39.9340},
                "category": "综合医院"
            },
            "verification": {"is_valid": True, "confidence": 0.95}
        },
        {
            "evidence_id": "EVD_002",
            "source": {"weight": 0.8, "source_type": "map_vendor"},
            "normalized_data": {
                "name": "北大第一医院",
                "address": "北京市西城区西什库大街8号",
                "coordinates": {"longitude": 116.3720, "latitude": 39.9345},
                "category": "医院"
            },
            "verification": {"is_valid": True, "confidence": 0.85}
        }
    ]
    
    # 创建决策聚合器
    aggregator = DecisionAggregator()
    
    # 聚合决策
    decision = aggregator.aggregate_decision(poi_data, dimension_results, evidence_list)
    
    # 保存决策
    aggregator.save_decision(decision, "decision_result.json")
    
    # 打印决策结果
    print(f"\n决策ID: {decision.decision_id}")
    print(f"POI ID: {decision.poi_id}")
    print(f"整体状态: {decision.overall['status']}")
    print(f"整体置信度: {decision.overall['confidence']}")
    print(f"建议动作: {decision.overall['action']}")
    print(f"决策摘要: {decision.overall['summary']}")
    
    if decision.downgrade_info:
        print(f"\n降级信息:")
        print(f"  是否降级: {decision.downgrade_info['is_downgraded']}")
        print(f"  降级原因: {decision.downgrade_info['reason_description']}")


if __name__ == "__main__":
    main()
