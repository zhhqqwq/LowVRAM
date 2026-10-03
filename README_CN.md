# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前进入 P1 — Benchmark Runner。**

P1-01 新增本机硬件采集：

```bash
lowvram system
```

命令输出结构化 JSON，包含：

- OS
- CPU
- RAM
- NVIDIA GPU 与总 VRAM
- NVIDIA Driver
- `nvidia-smi` 报告的 CUDA 兼容版本
- Python Version

没有 NVIDIA GPU、没有安装 `nvidia-smi` 或 NVIDIA 查询失败时，命令仍正常输出，并使用 `gpu: []` 表示 CPU-only 机器。

当前仍未实现 llama.cpp 调用、运行时 RAM/VRAM 监控、Benchmark 编排、推荐器、数据库聚合和 Web UI。
