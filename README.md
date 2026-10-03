# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 adds local hardware inventory:

```bash
python -m pip install -e ".[dev]"
lowvram system
```

The command outputs structured JSON containing OS, CPU, RAM, NVIDIA GPU/VRAM, NVIDIA driver, CUDA compatibility version, and Python version. Machines without a usable NVIDIA installation remain valid and return `gpu: []`.

P0 also provides strict data contracts and validation:

```bash
lowvram validate tests/fixtures/valid/01_dense_single_gpu_success.json
# VALID
```

## Implemented

- Python 3.11+ package and Typer CLI
- Strict Pydantic v2 models for hardware, model metadata, benchmark runs, and recipes
- JSON Schema Draft 2020-12 files for all four contracts
- `lowvram validate <file>`
- `lowvram system`
- Valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- P0 documentation plus P1-01 system-collector documentation

## Not implemented yet

Benchmark execution, llama.cpp runtime integration, process RAM/VRAM monitoring, performance parsing, recommendation logic, aggregation, Web UI, and model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/data-format.md` and `docs/system-collector.md`.
