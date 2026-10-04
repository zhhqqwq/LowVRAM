# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前处于 P1 — Benchmark Runner。**

当前 P1 已完成硬件采集、NVIDIA 快照、RAM Monitor、VRAM Monitor，并在 P1-05
新增统一 Runtime Adapter。

核心接口：

```python
from lowvram.models import RuntimeExecutionRequest
from lowvram.runtime import LlamaCppRuntimeAdapter

adapter = LlamaCppRuntimeAdapter()
result = adapter.execute(
    RuntimeExecutionRequest(
        executable="/path/to/llama-cli",
        arguments=["--version"],
        timeout_seconds=10,
    )
)
```

P1-05 负责：

- shell-free argv 构造
- 外部进程启动与等待
- stdout / stderr 捕获
- exit code
- timeout
- 超时进程树终止
- `runtime_not_found`
- `process_crash`
- 结构化执行结果

P1-05 不自动寻找 llama.cpp；可执行文件发现属于 P1-06。它也不提前实现完整 Benchmark
Orchestrator 或 Recipe 到 llama.cpp 参数的转换。
