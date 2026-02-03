# 更新履历 (CHANGELOG)

本文件用于记录 `bigpoi-verification` skill 包的所有重大变更、修复与改进。

## [1.0.1] - 2026-02-03

### 修复 (Fixed)
- **配置文件命名**: 将 `config/skill.ymal` 重命名为正确的 `config/skill.yaml`，修复由于拼写错误导致的 Skill 无法识别问题。

### 变更 (Changed)
- **数据源配置增强**: 在 `config/sources.yaml` 中引入了 `credentials` 全局凭据资源池结构。
- **API Key 资源池化**: 支持为百度、高德、腾讯地图配置多个 API Key 列表。
- **Referer 绑定支持**: 升级凭据存储结构，支持为每个 API Key 单独配置可选的 `referer` 字段。
- **字段名适配**: 适配厂商命名习惯，高德/百度使用 `ak` 字段，腾讯保持使用 `key` 字段。
- **引用机制优化**: 数据源通过 `api_key_ref` 标识符关联至对应的凭据池。

### 评审 (Review)
- 完成了对整个 Skill 包的深度技术评审，确认了其纯规则引擎（零 Token 消耗）的特性，并提出了引入 LLM 混合模式处理复杂语义别名的优化建议。
