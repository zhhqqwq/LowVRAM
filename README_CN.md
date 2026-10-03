# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

当前仓库只实现 **P0 — Foundation**：固定数据标准、验证逻辑、测试和 CI，不运行任何大模型。

## P0 已实现

- Python 3.11+ 项目与 Typer CLI
- Hardware / Model / Benchmark / Recipe 四类严格 Pydantic v2 模型
- 对应 JSON Schema Draft 2020-12
- `lowvram validate <file>`
- 至少 5 个合法与 5 个非法 Benchmark fixtures
- pytest、ruff、mypy、GitHub Actions
- 数据格式、架构与 Benchmark 方法文档

P0 明确不实现 Benchmark Runner、llama.cpp 集成、GPU 监控、推荐器、聚合、Web UI 或模型下载。

## 固定单位

- 内存：MB
- 时间：seconds
- 吞吐：tokens/sec
