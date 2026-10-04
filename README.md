# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 through P1-04 provide hardware inventory plus RAM/VRAM measurement. P1-05 adds the
external runtime execution boundary:

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

The adapter builds shell-free argv, captures stdout/stderr and exit code, enforces timeout,
terminates timed-out process trees, and normalizes missing runtime / timeout / process crash
outcomes.

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
- runtime adapter abstraction and shell-free subprocess execution
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- P0 plus P1 collector/monitor/runtime documentation

## Not implemented yet

Automatic llama.cpp discovery, recipe-to-runtime command translation, benchmark performance
parsing, recommendation logic, aggregation, Web UI, and model downloading remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/runtime-adapter.md` for P1-05.
