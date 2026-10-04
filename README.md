# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 through P1-04 provide hardware inventory plus RAM/VRAM measurement. P1-05 provides the
external process execution boundary. P1-06 discovers and version-probes llama.cpp. P1-07
builds deterministic llama.cpp argv:

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

The builder records the complete actual argv, preserves paths as single arguments, supports
CPU-only `gpu_layers=0`, and rejects `extra_args` that would silently override managed
benchmark parameters.

P0 also provides strict data contracts and validation:

```bash
lowvram validate tests/fixtures/valid/01_dense_single_gpu_success.json
# VALID
```

## Implemented

- Python 3.11+ package and Typer CLI
- strict Pydantic v2 P0 models and JSON Schema Draft 2020-12 contracts
- `lowvram validate <file>`
- `lowvram system`
- hardware/NVIDIA collection plus RAM/VRAM monitoring
- shell-free runtime adapter abstraction
- llama.cpp explicit-path/PATH discovery and version detection
- deterministic llama.cpp command building
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI

## Not implemented yet

Standard benchmark prompt, performance parsing, benchmark orchestration, recommendation
logic, aggregation, Web UI, and model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/llama-cpp-command-builder.md` for P1-07.
