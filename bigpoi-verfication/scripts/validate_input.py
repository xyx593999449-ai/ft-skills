#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Version: 1.0.2
# Author: BigPOI Verification System
# Date: 2024-01-15

功能定义说明:
    本脚本负责大POI核实流程中的输入校验阶段，对输入的POI数据进行格式校验、
    字段完整性检查、POI类型白名单验证等操作，确保数据质量和系统稳定性。

用途说明:
    该脚本在整个核实流程中处于第一阶段（输入校验与范围确认），主要作用包括：
    1. 按照input.schema.json验证输入数据的格式和结构
    2. 检查必需字段完整性
    3. 验证POI类型是否在支持的白名单中
    4. 验证城市和地理坐标信息的有效性
    5. 对不满足要求的输入直接输出拒绝或降级结果

    应用场景：批量核实数据的预处理、数据质量控制、拒绝非法或不支持的输入
"""

import json
import yaml
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """验证结果数据类 - 兼容Python 3.6"""

    def __init__(
        self,
        is_valid: bool,
        poi_id: str,
        errors: List[str],
        warnings: List[str],
        validated_data: Optional[Dict[str, Any]] = None
    ):
        self.is_valid = is_valid
        self.poi_id = poi_id
        self.errors = errors
        self.warnings = warnings
        self.validated_data = validated_data

    def __repr__(self):
        return (f"ValidationResult(is_valid={self.is_valid!r}, "
                f"poi_id={self.poi_id!r}, errors={len(self.errors)} errors)")


class InputValidator:
    """输入数据校验器"""

    def __init__(
        self,
        skill_config_path: str = "../config/skill.yaml",
        input_schema_path: str = "../schema/input.schema.json"
    ):
        """
        初始化验证器

        Args:
            skill_config_path: skill配置文件路径
            input_schema_path: 输入schema定义路径
        """
        self.skill_config = self._load_yaml(skill_config_path)
        self.input_schema = self._load_json(input_schema_path)
        self.supported_poi_types = self._extract_supported_poi_types()
        logger.info("输入校验器初始化完成")

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.debug(f"成功加载YAML配置: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载YAML配置失败: {e}")
            raise

    def _load_json(self, schema_path: str) -> Dict[str, Any]:
        """加载JSON Schema文件"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.debug(f"成功加载JSON Schema: {schema_path}")
            return schema
        except Exception as e:
            logger.error(f"加载JSON Schema失败: {e}")
            raise

    def _extract_supported_poi_types(self) -> set:
        """从skill配置中提取支持的POI类型"""
        poi_types = set()
        if 'skill' in self.skill_config:
            for poi_type_config in self.skill_config['skill'].get('supported_poi_types', []):
                if poi_type_config.get('enabled', True):
                    poi_types.add(poi_type_config.get('type'))
        return poi_types

    def validate(self, poi_data: Dict[str, Any]) -> ValidationResult:
        """
        验证POI数据

        Args:
            poi_data: POI输入数据

        Returns:
            验证结果
        """
        errors = []
        warnings = []
        poi_id = poi_data.get('poi_id', 'UNKNOWN')

        logger.info(f"开始验证POI数据: {poi_id}")

        # 1. 检查必需字段
        required_fields = self.skill_config.get('execution', {}).get('required_input_fields', [])
        if not required_fields:
            required_fields = ['poi_id', 'name', 'poi_type', 'city']

        for field in required_fields:
            if field not in poi_data or not poi_data[field]:
                errors.append(f"缺失必需字段: {field}")

        # 如果有必需字段缺失，直接返回失败
        if errors:
            logger.warning(f"POI {poi_id} 必需字段验证失败: {errors}")
            return ValidationResult(
                is_valid=False,
                poi_id=poi_id,
                errors=errors,
                warnings=warnings
            )

        # 2. 验证POI类型
        poi_type = poi_data.get('poi_type', '')
        if poi_type not in self.supported_poi_types:
            errors.append(
                f"POI类型 '{poi_type}' 不在支持的白名单中。"
                f"支持的类型: {', '.join(sorted(self.supported_poi_types))}"
            )

        # 3. 验证城市信息
        city = poi_data.get('city', '')
        if not city or not isinstance(city, str):
            errors.append("城市信息无效")

        # 4. 验证坐标（如果提供）
        if 'coordinates' in poi_data:
            coords = poi_data['coordinates']
            coord_errors = self._validate_coordinates(coords)
            errors.extend(coord_errors)

        # 5. 验证名称格式
        name = poi_data.get('name', '')
        if not name or not isinstance(name, str) or len(name) > 200:
            errors.append("POI名称无效（应为1-200字符的字符串）")

        # 6. 数据类型检查
        type_checks = [
            ('poi_id', str),
            ('name', str),
            ('poi_type', str),
            ('city', str),
            ('address', (str, type(None))),
            ('phone', (str, type(None))),
            ('website', (str, type(None))),
            ('postcode', (str, type(None))),
        ]

        for field, expected_type in type_checks:
            if field in poi_data:
                value = poi_data[field]
                if not isinstance(value, expected_type):
                    errors.append(f"字段 '{field}' 类型错误，期望 {expected_type}，实际 {type(value)}")

        # 7. 可选字段警告
        optional_fields = self.skill_config.get('execution', {}).get('optional_input_fields', [])
        if not optional_fields:
            optional_fields = ['address', 'coordinates', 'phone', 'website', 'postcode']

        unknown_fields = set(poi_data.keys()) - set(required_fields) - set(optional_fields)
        if unknown_fields:
            warnings.append(f"存在未知字段将被忽略: {', '.join(unknown_fields)}")

        # 确定验证结果
        is_valid = len(errors) == 0

        if is_valid:
            # 清理数据：只保留定义的字段
            allowed_fields = set(required_fields) | set(optional_fields)
            validated_data = {k: v for k, v in poi_data.items() if k in allowed_fields}
            logger.info(f"POI {poi_id} 验证通过")
        else:
            validated_data = None
            logger.warning(f"POI {poi_id} 验证失败: {errors}")

        return ValidationResult(
            is_valid=is_valid,
            poi_id=poi_id,
            errors=errors,
            warnings=warnings,
            validated_data=validated_data
        )

    def _validate_coordinates(self, coords: Any) -> List[str]:
        """
        验证坐标信息

        Args:
            coords: 坐标数据

        Returns:
            错误列表
        """
        errors = []

        if not isinstance(coords, dict):
            errors.append("坐标必须为字典类型")
            return errors

        # 检查必需的坐标字段
        if 'longitude' not in coords or 'latitude' not in coords:
            errors.append("坐标缺失 longitude 或 latitude 字段")
            return errors

        try:
            lon = float(coords['longitude'])
            lat = float(coords['latitude'])
        except (ValueError, TypeError):
            errors.append("经度或纬度无法转换为浮点数")
            return errors

        # 验证坐标范围（中国境内）
        if not (73.5 <= lon <= 135.1):
            errors.append(f"经度 {lon} 超出有效范围 [73.5, 135.1]")

        if not (3.5 <= lat <= 53.6):
            errors.append(f"纬度 {lat} 超出有效范围 [3.5, 53.6]")

        # 验证坐标系统（如果提供）
        if 'coordinate_system' in coords:
            supported_systems = ['WGS84', 'GCJ02', 'BD09']
            if coords['coordinate_system'] not in supported_systems:
                errors.append(
                    f"坐标系统 '{coords['coordinate_system']}' 不支持。"
                    f"支持的系统: {', '.join(supported_systems)}"
                )

        return errors

    def batch_validate(self, poi_data_list: List[Dict[str, Any]]) -> Tuple[List[ValidationResult], dict]:
        """
        批量验证POI数据

        Args:
            poi_data_list: POI数据列表

        Returns:
            (验证结果列表, 统计信息)
        """
        results = []
        stats = {
            'total': len(poi_data_list),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }

        logger.info(f"开始批量验证 {len(poi_data_list)} 条POI数据")

        for poi_data in poi_data_list:
            result = self.validate(poi_data)
            results.append(result)

            if result.is_valid:
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
                stats['errors'].append({
                    'poi_id': result.poi_id,
                    'errors': result.errors
                })

        logger.info(
            f"批量验证完成: 总数={stats['total']}, "
            f"有效={stats['valid']}, 无效={stats['invalid']}"
        )

        return results, stats


async def main():
    """
    主函数 - 示例用法
    """
    # 示例POI数据列表
    poi_data_list = [
        {
            "poi_id": "HOSPITAL_BJ_001",
            "name": "北京大学第一医院",
            "poi_type": "hospital",
            "city": "北京市",
            "address": "北京市西城区西什库大街8号",
            "coordinates": {
                "longitude": 116.3723,
                "latitude": 39.9342,
                "coordinate_system": "GCJ02"
            }
        },
        {
            "poi_id": "INVALID_001",
            "name": "测试POI",
            # 缺失 poi_type 和 city
        },
        {
            "poi_id": "UNSUPPORTED_001",
            "name": "不支持的类型",
            "poi_type": "unknown_type",
            "city": "北京市"
        }
    ]

    # 验证数据
    validator = InputValidator()
    results, stats = validator.batch_validate(poi_data_list)

    # 打印结果
    print("\n===== 批量验证结果 =====")
    print(f"总数: {stats['total']}, 有效: {stats['valid']}, 无效: {stats['invalid']}")

    for result in results:
        status = "✓" if result.is_valid else "✗"
        print(f"\n{status} {result.poi_id}")

        if result.errors:
            print("  错误:")
            for error in result.errors:
                print(f"    - {error}")

        if result.warnings:
            print("  警告:")
            for warning in result.warnings:
                print(f"    - {warning}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
