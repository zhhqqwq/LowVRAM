# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成。P1 — Benchmark Runner 的实现已完成；最终的开发者自有消费级硬件验收仍待执行。**

## 当前能力

P1-01～P1-04 已实现系统硬件采集、RAM Monitor 与 VRAM Monitor。P1-05 提供外部进程
Runtime Adapter；P1-06 完成 llama.cpp 发现与版本探测；P1-07 构建确定性的 llama.cpp
argv；P1-08 提供固定、带完整性校验的 Standard Prompt；P1-09 支持详细 timing block
以及当前 llama.cpp 的紧凑 Prompt/Generation throughput 输出。

P1-10 已实现基于真实 PID 的 Benchmark Orchestrator；P1-11 提供确定性的失败分类；
P1-12 提供不执行模型的 `lowvram doctor`；P1-13 提供与真实运行共享准备路径的
`lowvram benchmark --dry-run`；P1-14 将每次运行持久化为：

```text
runs/<run_id>/
├── benchmark.json
├── stdout.log
└── stderr.log
```

P1-15 已加入固定 provenance 的真实 seed 验证。当前技术 seed 使用真实 llama.cpp 与
真实 GGUF，在 GitHub-hosted CPU runner 上执行 10 次尝试，覆盖两个模型、两个 context，
并包含一次故意 timeout 失败。公开 BenchmarkRun 已升级为 1.1.0，记录
`prompt_version`，不泄露本地 executable/model 路径；当所选 llama.cpp timing 格式
没有直接提供模型 load time 时，`load_time_seconds` 保持为 `null`，不会用进程总耗时
冒充模型加载时间。

GitHub-hosted real-seed 属于真实运行证据，但不是开发者自有消费级硬件。因此 P1 的整体
Final Gate 仍需在开发者可控的消费级机器上运行同一 acceptance matrix 后才能正式通过。

## 已实现

- Python 3.11+ package 与 Typer CLI
- strict Pydantic v2 models 与 JSON Schema Draft 2020-12 contracts
- `lowvram validate <file>`
- `lowvram system`
- 硬件/NVIDIA 采集与 RAM/VRAM monitoring
- shell-free Runtime Adapter
- llama.cpp 发现与版本检测
- 确定性 llama.cpp Command Builder
- 固定且带 SHA-256 完整性校验的 Standard Prompt
- detailed / compact llama.cpp timing parser
- live-PID Benchmark Orchestrator
- Failure Classification
- `lowvram doctor`
- `lowvram benchmark --dry-run`
- `runs/<run_id>/` 原子化三文件日志
- 固定版本真实 llama.cpp/GGUF CPU seed 验证
- 带 prompt provenance 且去除私人本地路径的 BenchmarkRun 1.1.0 export
- Ruff、mypy、pytest 与 GitHub Actions CI

## P1 最终剩余 Gate

在开发者自有消费级硬件上执行 P1-15 acceptance matrix，并审计生成的 evidence；通过后
才能把整个 P1 阶段标记为完成。

P2 Open Benchmark Database 与 P3 “Can I Run It?” 属于后续阶段。

详见 `docs/p1-15-real-seed.md`。
