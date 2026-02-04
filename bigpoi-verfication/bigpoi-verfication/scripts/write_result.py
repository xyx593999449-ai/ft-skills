#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.2
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的结果输出与记录阶段，格式化输出最终核实结果并生成可入库的记录文档。
    将决策结果转换为符合schema/record.schema.json标准的记录格式，包含审计追踪信息、质量指标、
    标记信息等，确保所有输出结果支持后续审计、抽检与责任追溯。

用途说明:
    该脚本在整个核实流程中处于第六阶段（结果输出与记录），主要作用包括：
    1. 将决策结果转换为标准记录格式
    2. 生成完整的审计追踪信息（创建者、审核者、版本历史）
    3. 计算并记录质量指标（各维度评分、证据质量、来源多样性）
    4. 设置标记信息（敏感标记、争议标记、定期复核标记）
    5. 生成处理元数据（技能版本、处理时间、数据源、应用规则）
    6. 支持多种输出格式（JSON、CSV、数据库写入）
    7. 生成人类可读的核实报告

    应用场景：批量数据入库、质量抽检、审计追溯、报告生成
"""

import json
import csv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VerificationRecord:
    """核实记录数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        record_id: str,
        poi_id: str,
        input_data: Dict[str, Any],
        verification_result: Dict[str, Any],
        decision_ref: str,
        evidence_refs: List[str],
        audit_trail: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        flags: Dict[str, Any],
        metadata: Dict[str, Any],
        created_at: str,
        updated_at: str,
        expires_at: Optional[str] = None
    ):
        self.record_id = record_id
        self.poi_id = poi_id
        self.input_data = input_data
        self.verification_result = verification_result
        self.decision_ref = decision_ref
        self.evidence_refs = evidence_refs
        self.audit_trail = audit_trail
        self.quality_metrics = quality_metrics
        self.flags = flags
        self.metadata = metadata
        self.created_at = created_at
        self.updated_at = updated_at
        self.expires_at = expires_at

    def __repr__(self):
        return (f"VerificationRecord(record_id={self.record_id!r}, "
                f"poi_id={self.poi_id!r}, created_at={self.created_at!r})")


class ResultWriter:
    """结果写入器类"""
    
    def __init__(self, skill_version: str = "1.0.0"):
        """
        初始化结果写入器
        
        Args:
            skill_version: 技能版本号
        """
        self.skill_version = skill_version
        logger.info(f"结果写入器初始化完成，技能版本: {skill_version}")
    
    def _generate_record_id(self, poi_id: str) -> str:
        """
        生成记录唯一标识符
        
        Args:
            poi_id: POI标识符
            
        Returns:
            记录ID
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        base = f"{poi_id}_{timestamp}"
        return f"REC_{hashlib.md5(base.encode()).hexdigest()[:16].upper()}"
    
    def _extract_input_data(self, poi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取输入数据
        
        Args:
            poi_data: POI原始数据
            
        Returns:
            输入数据字典
        """
        return {
            "name": poi_data.get('name', ''),
            "poi_type": poi_data.get('poi_type', ''),
            "city": poi_data.get('city', ''),
            "address": poi_data.get('address', ''),
            "coordinates": poi_data.get('coordinates', {}),
            "source": poi_data.get('source', '')
        }
    
    def _build_verification_result(
        self,
        poi_data: Dict[str, Any],
        decision: Dict[str, Any],
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建核实结果
        
        Args:
            poi_data: POI原始数据
            decision: 决策结果
            evidence_list: 证据列表
            
        Returns:
            核实结果字典
        """
        overall = decision.get('overall', {})
        corrections = decision.get('corrections', {}) or {}
        dimensions = decision.get('dimensions', {})
        
        # 确定核实状态
        status_map = {
            'accepted': 'verified',
            'downgraded': 'modified',
            'manual_review': 'manual_review_pending',
            'rejected': 'rejected'
        }
        status = status_map.get(overall.get('status'), 'manual_review_pending')
        
        # 构建最终值
        # 确保坐标包含GCJ02坐标系信息
        final_coordinates = corrections.get('coordinates', {}).get('suggested', poi_data.get('coordinates', {}))
        if final_coordinates and 'coordinate_system' not in final_coordinates:
            final_coordinates = dict(final_coordinates)
            final_coordinates['coordinate_system'] = 'GCJ02'
        
        final_values = {
            "name": corrections.get('name', {}).get('suggested', poi_data.get('name', '')),
            "name_confidence": dimensions.get('name', {}).get('confidence', 0),
            "address": corrections.get('address', {}).get('suggested', poi_data.get('address', '')),
            "address_confidence": 0.9,  # 默认地址置信度
            "coordinates": final_coordinates,
            "coordinates_confidence": dimensions.get('location', {}).get('confidence', 0),
            "category": corrections.get('category', {}).get('suggested', poi_data.get('category_code', '')),
            "category_confidence": dimensions.get('category', {}).get('confidence', 0),
            "city": poi_data.get('city', ''),
            "city_confidence": dimensions.get('administrative', {}).get('confidence', 0)
        }
        
        # 构建变更记录
        changes = []
        for field, correction in corrections.items():
            if field == 'coordinates':
                original = correction.get('original', {})
                suggested = correction.get('suggested', {})
                if original != suggested:
                    changes.append({
                        "field": field,
                        "old_value": f"({original.get('longitude', 0)}, {original.get('latitude', 0)}) [{original.get('coordinate_system', 'unknown')}]",
                        "new_value": f"({suggested.get('longitude', 0)}, {suggested.get('latitude', 0)}) [GCJ02]",
                        "reason": f"{field}维度验证结果: {dimensions.get(field, {}).get('result', 'unknown')}"
                    })
            else:
                old_value = correction.get('original', '')
                new_value = correction.get('suggested', '')
                if old_value != new_value:
                    changes.append({
                        "field": field,
                        "old_value": str(old_value),
                        "new_value": str(new_value),
                        "reason": f"{field}维度验证结果: {dimensions.get(field, {}).get('result', 'unknown')}"
                    })
        
        return {
            "status": status,
            "confidence": overall.get('confidence', 0),
            "final_values": final_values,
            "changes": changes
        }
    
    def _build_audit_trail(
        self,
        decision: Dict[str, Any],
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """
        构建审计追踪信息
        
        Args:
            decision: 决策结果
            created_by: 创建者
            
        Returns:
            审计追踪字典
        """
        created_at = decision.get('created_at', datetime.utcnow().isoformat() + 'Z')
        
        audit_trail = {
            "created_by": created_by,
            "created_at": created_at,
            "version_history": [
                {
                    "version": self.skill_version,
                    "timestamp": created_at,
                    "operator": created_by,
                    "action": "auto_verified" if decision.get('overall', {}).get('status') == 'accepted' else "auto_downgraded"
                }
            ]
        }
        
        # 如果需要人工审核，添加审核信息占位
        if decision.get('overall', {}).get('status') in ['manual_review', 'downgraded']:
            audit_trail["verification_notes"] = decision.get('downgrade_info', {}).get('recommendation', '')
        
        return audit_trail
    
    def _calculate_quality_metrics(
        self,
        decision: Dict[str, Any],
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算质量指标
        
        Args:
            decision: 决策结果
            evidence_list: 证据列表
            
        Returns:
            质量指标字典
        """
        dimensions = decision.get('dimensions', {})
        
        # 各维度评分
        dimension_scores = {}
        for dim_name in ['existence', 'name', 'location', 'category', 'administrative', 'timeliness']:
            dim_result = dimensions.get(dim_name, {})
            dimension_scores[dim_name] = dim_result.get('score', 0)
        
        # 证据质量评分
        if evidence_list:
            valid_evidence = [e for e in evidence_list if e.get('verification', {}).get('is_valid', False)]
            evidence_quality = sum(
                e.get('verification', {}).get('confidence', 0) for e in valid_evidence
            ) / len(valid_evidence) if valid_evidence else 0
        else:
            evidence_quality = 0
        
        # 来源多样性评分
        source_types = set()
        for evidence in evidence_list:
            source_type = evidence.get('source', {}).get('source_type', 'unknown')
            source_types.add(source_type)
        
        # 来源类型越多，多样性评分越高
        source_diversity = min(len(source_types) / 3, 1.0)  # 最多3种类型得满分
        
        return {
            "dimension_scores": dimension_scores,
            "evidence_quality": round(evidence_quality, 4),
            "source_diversity": round(source_diversity, 4)
        }
    
    def _build_flags(
        self,
        poi_data: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建标记信息
        
        Args:
            poi_data: POI原始数据
            decision: 决策结果
            
        Returns:
            标记信息字典
        """
        # 判断是否敏感（示例：政府机关、公安机关等）
        poi_type = poi_data.get('poi_type', '')
        sensitive_types = ['government', 'police', 'court', 'procuratorate']
        is_sensitive = poi_type in sensitive_types
        
        # 判断是否存在争议
        dimensions = decision.get('dimensions', {})
        is_disputed = any(
            d.get('result') == 'uncertain' for d in dimensions.values()
        )
        
        # 确定是否需要定期复核
        # 敏感数据或降级数据需要定期复核
        requires_periodic_review = is_sensitive or decision.get('downgrade_info') is not None
        
        # 设置复核周期
        review_period_days = 365  # 默认一年
        if is_sensitive:
            review_period_days = 180  # 敏感数据半年复核
        
        # 生成标签
        tags = []
        if is_sensitive:
            tags.append("敏感数据")
        if decision.get('overall', {}).get('status') == 'accepted':
            tags.append("高置信度")
        elif decision.get('overall', {}).get('status') == 'downgraded':
            tags.append("降级处理")
        
        # 根据POI类型添加标签
        poi_type_tags = {
            'hospital': '医疗机构',
            'government': '政府机构',
            'scenic': '旅游景区',
            'university': '高等院校'
        }
        if poi_type in poi_type_tags:
            tags.append(poi_type_tags[poi_type])
        
        return {
            "is_sensitive": is_sensitive,
            "is_disputed": is_disputed,
            "requires_periodic_review": requires_periodic_review,
            "review_period_days": review_period_days,
            "tags": tags
        }
    
    def _build_metadata(
        self,
        decision: Dict[str, Any],
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建元数据
        
        Args:
            decision: 决策结果
            evidence_list: 证据列表
            
        Returns:
            元数据字典
        """
        # 提取使用的数据源
        data_sources = list(set(
            e.get('source', {}).get('source_id', 'unknown')
            for e in evidence_list
        ))
        
        # 提取应用的规则
        rules_applied = []
        dimensions = decision.get('dimensions', {})
        for dim_name in dimensions.keys():
            rules_applied.append(f"{dim_name}_verify")
        
        return {
            "skill_version": self.skill_version,
            "processing_time_ms": decision.get('processing_duration_ms', 0),
            "data_sources": data_sources,
            "rules_applied": rules_applied
        }
    
    def create_record(
        self,
        poi_data: Dict[str, Any],
        decision: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        created_by: str = "system"
    ) -> VerificationRecord:
        """
        创建核实记录
        
        Args:
            poi_data: POI原始数据
            decision: 决策结果
            evidence_list: 证据列表
            created_by: 创建者
            
        Returns:
            核实记录对象
        """
        poi_id = poi_data.get('poi_id', 'unknown')
        
        logger.info(f"开始创建核实记录: {poi_id}")
        
        # 生成记录ID
        record_id = self._generate_record_id(poi_id)
        
        # 提取输入数据
        input_data = self._extract_input_data(poi_data)
        
        # 构建核实结果
        verification_result = self._build_verification_result(
            poi_data, decision, evidence_list
        )
        
        # 构建审计追踪
        audit_trail = self._build_audit_trail(decision, created_by)
        
        # 计算质量指标
        quality_metrics = self._calculate_quality_metrics(decision, evidence_list)
        
        # 构建标记信息
        flags = self._build_flags(poi_data, decision)
        
        # 构建元数据
        metadata = self._build_metadata(decision, evidence_list)
        
        # 时间戳
        created_at = decision.get('created_at', datetime.utcnow().isoformat() + 'Z')
        updated_at = created_at
        
        # 计算过期时间
        expires_at = None
        if flags.get('requires_periodic_review'):
            review_days = flags.get('review_period_days', 365)
            expires_dt = datetime.utcnow() + timedelta(days=review_days)
            expires_at = expires_dt.isoformat() + 'Z'
        
        logger.info(f"核实记录创建完成: {record_id}")
        
        return VerificationRecord(
            record_id=record_id,
            poi_id=poi_id,
            input_data=input_data,
            verification_result=verification_result,
            decision_ref=decision.get('decision_id', ''),
            evidence_refs=[e.get('evidence_id') for e in evidence_list],
            audit_trail=audit_trail,
            quality_metrics=quality_metrics,
            flags=flags,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at
        )
    
    def record_to_dict(self, record: VerificationRecord) -> Dict[str, Any]:
        """
        将记录对象转换为字典
        
        Args:
            record: 核实记录对象
            
        Returns:
            记录字典
        """
        return {
            "record_id": record.record_id,
            "poi_id": record.poi_id,
            "input_data": record.input_data,
            "verification_result": record.verification_result,
            "decision_ref": record.decision_ref,
            "evidence_refs": record.evidence_refs,
            "audit_trail": record.audit_trail,
            "quality_metrics": record.quality_metrics,
            "flags": record.flags,
            "metadata": record.metadata,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "expires_at": record.expires_at
        }
    
    def save_record_json(
        self, 
        record: VerificationRecord, 
        output_path: str
    ) -> None:
        """
        保存记录为JSON格式
        
        Args:
            record: 核实记录对象
            output_path: 输出文件路径
        """
        try:
            record_dict = self.record_to_dict(record)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(record_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"记录已保存为JSON: {output_path}")
        except Exception as e:
            logger.error(f"保存JSON记录失败: {e}")
            raise
    
    def save_record_csv(
        self, 
        records: List[VerificationRecord], 
        output_path: str
    ) -> None:
        """
        批量保存记录为CSV格式
        
        Args:
            records: 核实记录列表
            output_path: 输出文件路径
        """
        try:
            if not records:
                logger.warning("记录列表为空，不生成CSV")
                return
            
            # 定义CSV字段
            fieldnames = [
                'record_id', 'poi_id', 'input_name', 'input_city',
                'status', 'confidence', 'final_name', 'final_address',
                'created_at', 'is_sensitive', 'tags'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in records:
                    row = {
                        'record_id': record.record_id,
                        'poi_id': record.poi_id,
                        'input_name': record.input_data.get('name', ''),
                        'input_city': record.input_data.get('city', ''),
                        'status': record.verification_result.get('status', ''),
                        'confidence': record.verification_result.get('confidence', 0),
                        'final_name': record.verification_result.get('final_values', {}).get('name', ''),
                        'final_address': record.verification_result.get('final_values', {}).get('address', ''),
                        'created_at': record.created_at,
                        'is_sensitive': record.flags.get('is_sensitive', False),
                        'tags': ','.join(record.flags.get('tags', []))
                    }
                    writer.writerow(row)
            
            logger.info(f"记录已保存为CSV: {output_path}")
        except Exception as e:
            logger.error(f"保存CSV记录失败: {e}")
            raise
    
    def generate_report(
        self, 
        record: VerificationRecord,
        output_path: str
    ) -> None:
        """
        生成人类可读的核实报告
        
        Args:
            record: 核实记录对象
            output_path: 输出文件路径
        """
        try:
            lines = []
            lines.append("=" * 60)
            lines.append("POI核实报告")
            lines.append("=" * 60)
            lines.append("")
            
            # 基本信息
            lines.append("【基本信息】")
            lines.append(f"记录ID: {record.record_id}")
            lines.append(f"POI ID: {record.poi_id}")
            lines.append(f"核实时间: {record.created_at}")
            lines.append(f"技能版本: {record.metadata.get('skill_version', 'unknown')}")
            lines.append("")
            
            # 输入信息
            lines.append("【输入信息】")
            lines.append(f"名称: {record.input_data.get('name', '')}")
            lines.append(f"类型: {record.input_data.get('poi_type', '')}")
            lines.append(f"城市: {record.input_data.get('city', '')}")
            lines.append(f"地址: {record.input_data.get('address', '')}")
            lines.append("")
            
            # 核实结果
            lines.append("【核实结果】")
            result = record.verification_result
            lines.append(f"状态: {result.get('status', '')}")
            lines.append(f"置信度: {result.get('confidence', 0)}")
            lines.append("")
            
            # 最终值
            lines.append("【核实后信息】")
            final_values = result.get('final_values', {})
            lines.append(f"名称: {final_values.get('name', '')} (置信度: {final_values.get('name_confidence', 0)})")
            lines.append(f"地址: {final_values.get('address', '')} (置信度: {final_values.get('address_confidence', 0)})")
            lines.append("")
            
            # 变更记录
            changes = result.get('changes', [])
            if changes:
                lines.append("【变更记录】")
                for change in changes:
                    lines.append(f"  字段: {change.get('field', '')}")
                    lines.append(f"  原值: {change.get('old_value', '')}")
                    lines.append(f"  新值: {change.get('new_value', '')}")
                    lines.append(f"  原因: {change.get('reason', '')}")
                    lines.append("")
            
            # 质量指标
            lines.append("【质量指标】")
            metrics = record.quality_metrics
            lines.append(f"证据质量: {metrics.get('evidence_quality', 0)}")
            lines.append(f"来源多样性: {metrics.get('source_diversity', 0)}")
            lines.append("各维度评分:")
            for dim, score in metrics.get('dimension_scores', {}).items():
                lines.append(f"  {dim}: {score}")
            lines.append("")
            
            # 标记信息
            lines.append("【标记信息】")
            flags = record.flags
            lines.append(f"敏感数据: {'是' if flags.get('is_sensitive') else '否'}")
            lines.append(f"存在争议: {'是' if flags.get('is_disputed') else '否'}")
            lines.append(f"需要定期复核: {'是' if flags.get('requires_periodic_review') else '否'}")
            if flags.get('requires_periodic_review'):
                lines.append(f"复核周期: {flags.get('review_period_days', 365)}天")
            lines.append(f"标签: {', '.join(flags.get('tags', []))}")
            lines.append("")
            
            # 审计信息
            lines.append("【审计信息】")
            audit = record.audit_trail
            lines.append(f"创建者: {audit.get('created_by', '')}")
            lines.append(f"创建时间: {audit.get('created_at', '')}")
            lines.append("")
            
            lines.append("=" * 60)
            lines.append("报告结束")
            lines.append("=" * 60)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"核实报告已生成: {output_path}")
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
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
        "category_code": "090100",
        "source": "官方导入"
    }
    
    # 示例决策结果
    decision = {
        "decision_id": "DEC_202401150001",
        "overall": {
            "status": "accepted",
            "confidence": 0.92,
            "action": "adopt",
            "summary": "高置信度通过，所有维度均符合要求"
        },
        "dimensions": {
            "existence": {"result": "pass", "confidence": 0.95, "score": 0.95},
            "name": {"result": "pass", "confidence": 1.0, "score": 1.0},
            "location": {"result": "pass", "confidence": 0.90, "score": 0.90},
            "category": {"result": "pass", "confidence": 0.95, "score": 0.95},
            "administrative": {"result": "pass", "confidence": 1.0, "score": 1.0},
            "timeliness": {"result": "pass", "confidence": 0.90, "score": 0.90}
        },
        "corrections": None,
        "created_at": "2024-01-15T08:40:00Z",
        "processing_duration_ms": 5000
    }
    
    # 示例证据列表
    evidence_list = [
        {
            "evidence_id": "EVD_001",
            "source": {"source_id": "NHC", "weight": 1.0},
            "verification": {"is_valid": True, "confidence": 0.95}
        },
        {
            "evidence_id": "EVD_002",
            "source": {"source_id": "AMAP", "weight": 0.8},
            "verification": {"is_valid": True, "confidence": 0.85}
        }
    ]
    
    # 创建结果写入器
    writer = ResultWriter(skill_version="1.0.0")
    
    # 创建核实记录
    record = writer.create_record(poi_data, decision, evidence_list)
    
    # 保存为JSON
    writer.save_record_json(record, "verification_record.json")
    
    # 生成报告
    writer.generate_report(record, "verification_report.txt")
    
    # 打印记录摘要
    print(f"\n记录ID: {record.record_id}")
    print(f"POI ID: {record.poi_id}")
    print(f"核实状态: {record.verification_result['status']}")
    print(f"置信度: {record.verification_result['confidence']}")
    print(f"是否敏感: {record.flags['is_sensitive']}")
    print(f"标签: {', '.join(record.flags['tags'])}")


if __name__ == "__main__":
    main()
