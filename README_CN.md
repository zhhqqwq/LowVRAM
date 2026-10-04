# LowVRAM

LowVRAM 是一个面向消费级硬件的开放本地 AI 模型兼容性数据库。长期目标是回答：**我的电脑能不能跑这个模型、应该怎么跑、能跑多快？**

**P0 — Foundation 已完成，当前处于 P1 — Benchmark Runner。**

当前 P1 已完成硬件采集、RAM/VRAM Monitor、Runtime Adapter、llama.cpp Detector，并在
P1-07 新增确定性的 llama.cpp Command Builder：

```python
from lowvram.models import LlamaCppCommandRequest
from lowvram.runtime import build_llama_cpp_command

command = build_llama_cpp_command(
    LlamaCppCommandRequest(
        executable="/path/to/llama-cli",
        model_path="/path/to/model.gguf",
        context_length=4096,
        threads=8,
        gpu_layers=32,
        batch_size=512,
        temperature=0.8,
        seed=42,
    )
)
```

P1-07 固定映射 model/context/threads/gpu_layers/batch/temperature/seed，并保存完整实际
argv。路径包含空格时仍作为单个参数，不经过 shell。

`extra_args` 会保持原顺序追加，但不能重复 `--model`、`--ctx-size`、
`--threads`、`--gpu-layers`、`--batch-size`、`--temp`、`--seed`
及其短别名，避免实际命令与结构化配置不一致。

CPU-only 使用 `gpu_layers=0`。P0 `Recipe` 在 P1-07 保持不变。

P1-08 将负责固定 Standard Prompt。
