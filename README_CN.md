# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前处于 P1 — Benchmark Runner。**

当前 P1 已完成硬件采集、RAM/VRAM Monitor、Runtime Adapter，并在 P1-06 新增
llama.cpp Detector：

```python
from lowvram.runtime import detect_llama_cpp

result = detect_llama_cpp()
```

P1-06 支持显式 executable 路径或 PATH 搜索，优先现代 `llama-cli`，同时兼容旧版
`main`，并覆盖 Windows/Linux 候选名称。版本探测依次尝试 `--version` 和
`version`。

结构化结果记录 executable、发现来源、候选名称、是否可运行、版本、实际成功的版本
探测命令，以及 `verified_eligible`。

如果 executable 可以运行但版本无法识别，仍报告为 found，但
`verified_eligible = false`，因此不能生成 Verified Benchmark。

P1-06 不负责 Recipe 到 llama.cpp 参数转换；该职责留给 P1-07。
