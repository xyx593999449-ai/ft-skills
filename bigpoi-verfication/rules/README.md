# POI核实规则引擎目录

本目录包含了大POI核实技能的规则定义，按维度进行组织。

## 目录结构

```
rules/
├── existence/          # 存在性维度规则
│   └── README.md      # 存在性规则说明
├── name/              # 名称准确性维度规则
│   └── README.md      # 名称规则说明
├── location/          # 空间位置维度规则
│   └── README.md      # 位置规则说明
├── category/          # 分类正确性维度规则
│   └── README.md      # 分类规则说明
└── README.md          # 本文件
```

## 规则文件格式

每个维度的规则文件应遵循以下格式：

### YAML格式规则文件示例

```yaml
# 规则名称
name: "规则名称"
# 规则描述
description: "规则详细说明"
# 规则类型：condition/aggregation/transformation
type: "condition"
# 应用维度
dimension: "existence"
# 优先级（高优先级规则优先应用）
priority: 1
# 规则条件
condition:
  # 条件表达式
  expression: "evidence_count >= 2 && high_weight_source > 0"
  # 对应的置信度调整
  confidence_adjustment: 0.1
# 规则输出
output:
  result: "pass"  # pass/fail/uncertain
  confidence: 0.9
```

## 规则使用原则

1. **维度隔离**：每个维度只能使用自己目录下的规则，禁止跨维度调用
2. **优先级顺序**：按优先级从高到低依次应用规则
3. **证据来源**：所有规则的输入必须来自规范化后的证据
4. **可追溯性**：规则必须记录应用过程，支持审计追溯

## 规则编写指南

### 存在性规则 (rules/existence/)

- 判断POI实体是否真实存在
- 输入：规范化后的证据列表、证据权重
- 输出：pass/fail/uncertain 结果与置信度

### 名称准确性规则 (rules/name/)

- 验证输入名称与证据名称是否一致
- 输入：规范化名称、名称相似度计算结果
- 输出：名称匹配结果与相似度

### 空间位置规则 (rules/location/)

- 验证坐标是否与参考位置一致
- 输入：规范化后的GCJ02坐标、距离计算结果
- 输出：位置偏差结果与距离

### 分类正确性规则 (rules/category/)

- 验证POI分类是否与实体属性匹配
- 输入：POI分类编码、证据分类信息
- 输出：分类匹配结果与匹配度

## 规则执行流程

```
输入证据 → 按维度筛选规则 → 按优先级排序 → 逐一应用 → 收集结果 → 计算最终结果
```

## 规则管理

- 规则版本应与技能版本保持同步
- 规则更新应记录在 SKILL.md 的版本日志中
- 规则修改前应进行充分的测试验证
- 建议定期评估规则的准确率和覆盖率

## 扩展规则

如需添加新规则，请按以下步骤操作：

1. 确定规则所属维度
2. 在对应维度目录下创建规则文件
3. 遵循YAML格式标准
4. 编写详细的规则说明和应用场景
5. 进行单元测试验证
6. 更新 CHANGELOG.md 记录变更

## 相关文档

- [技能文档](../SKILL.md)：技能全面说明
- [阈值配置](../config/thresholds.yaml)：量化阈值标准
- [降级策略](../config/downgrade.yaml)：降级处理规则
