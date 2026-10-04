# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前处于 P1 — Benchmark Runner。**

当前 P1 已实现：

- P1-01：`lowvram system` 本机硬件采集
- P1-02：一次性 NVIDIA GPU 状态快照
- P1-03：目标进程及递归子进程 RAM Monitor
- P1-04：带 baseline 的多 GPU VRAM Monitor

P1-04 使用：

```python
from lowvram.collectors import VramMonitor

monitor = VramMonitor(pid)
monitor.start()
result = monitor.stop()
```

第一版固定每 100ms 采样一次。结果保留总量和每 GPU 的
baseline/current/peak/delta。整机在 Benchmark 开始前已经占用的显存作为 baseline，
不会被计入新增显存 delta。

多 GPU 的总 peak 使用同一采样时刻的显存总和取最大值，不把不同时间发生的各卡
独立 peak 直接相加。无 NVIDIA 时返回合法的空 GPU / 0 值结果。

当前仍未实现 llama.cpp 调用、Runtime Adapter、性能解析和 Benchmark 编排。
