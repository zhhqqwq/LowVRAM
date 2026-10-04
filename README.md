# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 provides local hardware inventory:

```bash
python -m pip install -e ".[dev]"
lowvram system
```

P1-02 provides one-shot NVIDIA state snapshots. P1-03 adds target-process RAM monitoring.
P1-04 adds baseline-adjusted multi-GPU VRAM monitoring:

```python
from lowvram.collectors import RamMonitor, VramMonitor

ram_monitor = RamMonitor(pid)
vram_monitor = VramMonitor(pid)

ram_monitor.start()
vram_monitor.start()

# benchmark process runs here

ram_result = ram_monitor.stop()
vram_result = vram_monitor.stop()
```

Both monitors use a 100 ms default cadence. VRAM results retain total and per-GPU
baseline/current/peak/delta values so pre-existing desktop VRAM is not counted as new load.

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
- baseline-adjusted multi-GPU VRAM monitoring
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- P0 plus P1 collector/monitor documentation

## Not implemented yet

Benchmark execution, llama.cpp runtime integration, performance parsing, recommendation
logic, aggregation, Web UI, and model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/data-format.md`, `docs/system-collector.md`, `docs/nvidia-collector.md`,
`docs/ram-monitor.md`, and `docs/vram-monitor.md`.
