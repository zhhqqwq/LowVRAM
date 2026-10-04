# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 provides local hardware inventory:

```bash
python -m pip install -e ".[dev]"
lowvram system
```

P1-02 provides one-shot NVIDIA state snapshots:

```python
from lowvram.collectors import collect_nvidia_snapshot

snapshot = collect_nvidia_snapshot()
```

P1-03 adds target-process RAM monitoring:

```python
from lowvram.collectors import RamMonitor

monitor = RamMonitor(pid)
monitor.start()
result = monitor.stop()
```

The RAM monitor samples every 100 ms, follows recursively discovered child processes, and records baseline/current/peak/delta for both process-tree RSS and whole-system used RAM.

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
- one-shot NVIDIA status snapshots
- target-process RAM monitoring with child-process tracking
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- P0 plus P1 collector/monitor documentation

## Not implemented yet

Benchmark execution, llama.cpp runtime integration, VRAM time-series monitoring, performance parsing, recommendation logic, aggregation, Web UI, and model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/data-format.md`, `docs/system-collector.md`, `docs/nvidia-collector.md`, and `docs/ram-monitor.md`.
