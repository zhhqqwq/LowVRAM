# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 through P1-04 provide hardware inventory plus RAM/VRAM measurement. P1-05 provides the
external process execution boundary. P1-06 discovers and version-probes llama.cpp. P1-07
builds deterministic llama.cpp argv. P1-08 provides the fixed Standard Prompt:

```python
from lowvram.prompts import load_benchmark_prompt

prompt = load_benchmark_prompt()
```

The canonical `benchmark_prompts/v1.txt` asset is versioned as `v1` and pinned by SHA-256.
The loader refuses missing, unknown, or modified prompt content so benchmark inputs cannot
silently drift between runs.

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
- versioned, integrity-pinned Standard Prompt
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI

## Not implemented yet

Performance parsing, benchmark orchestration, recommendation logic, aggregation, Web UI, and
model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/standard-prompt.md` for P1-08.
