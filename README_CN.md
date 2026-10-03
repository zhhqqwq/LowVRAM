# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前处于 P1 — Benchmark Runner。**

P1-01 已提供：

```bash
lowvram system
```

用于输出 OS、CPU、RAM、NVIDIA GPU/总 VRAM、Driver、CUDA 兼容版本和 Python Version。

P1-02 新增一次性 NVIDIA 状态快照接口：

```python
from lowvram.collectors import collect_nvidia_snapshot

snapshot = collect_nvidia_snapshot()
```

快照包含 GPU index、名称、总 VRAM、已用 VRAM、GPU 利用率、NVIDIA Driver 和 CUDA 兼容版本。没有 NVIDIA 或 `nvidia-smi` 查询失败时返回空 GPU 列表；不伪造测量值。

P1-02 不进行循环采样，不计算 baseline、peak 或 delta VRAM；这些属于后续 P1-04 VRAM Monitor。

当前仍未实现 llama.cpp 调用、持续 RAM/VRAM 监控、Benchmark 编排、推荐器、数据库聚合和 Web UI。
