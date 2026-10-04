# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前处于 P1 — Benchmark Runner。**

当前 P1 已完成硬件采集、RAM/VRAM Monitor、Runtime Adapter、llama.cpp Detector、
Command Builder，并在 P1-08 新增固定 Standard Prompt：

```python
from lowvram.prompts import load_benchmark_prompt

prompt = load_benchmark_prompt()
```

标准文件是 `benchmark_prompts/v1.txt`，版本固定为 `v1`，并登记 SHA-256。
loader 会拒绝文件缺失、未知版本或内容被修改的情况，避免不同 Benchmark 在不知情的
情况下使用不同输入负载。

Prompt 内容完全由仓库提供，不依赖网络、外部知识、私人信息或机器特定信息。

P1-08 不负责执行 llama.cpp。下一项 P1-09 将实现 llama.cpp Output Parser。
