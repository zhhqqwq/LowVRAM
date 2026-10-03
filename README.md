# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 provides local hardware inventory:

```bash
python -m pip install -e ".[dev]"
lowvram system
```

The command outputs structured JSON containing OS, CPU, RAM, NVIDIA GPU/VRAM, NVIDIA driver, CUDA compatibility version, and Python version. Machines without a usable NVIDIA installation remain valid and return `gpu: []`.

P1-02 adds a reusable one-shot NVIDIA state API:

```python
from lowvram.collectors import collect_nvidia_snapshot

snapshot = collect_nvidia_snapshot()
```

It captures GPU index/name, total and used VRAM, GPU utilization, NVIDIA driver, and CUDA compatibility version. It is a snapshot only; continuous VRAM monitoring remains later P1 work.

P0 also provides strict data contracts and validation:

```bash
lowvram validate tests/fixtures/valid/01_dense_single_gpu_success.json
# VALID
```

## Implemented

- Python 3.11+ package and Typer CLI
- Strict Pydantic v2 P0 models and JSON Schema Draft 2020-12 contracts
- `lowvram validate <file>`
- `lowvram system`
- one-shot NVIDIA status snapshots
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- P0, system-collector, and NVIDIA-collector documentation

## Not implemented yet

Benchmark execution, llama.cpp runtime integration, continuous process RAM/VRAM monitoring, performance parsing, recommendation logic, aggregation, Web UI, and model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/data-format.md`, `docs/system-collector.md`, and `docs/nvidia-collector.md`.
