# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 through P1-04 provide hardware inventory plus RAM/VRAM measurement. P1-05 provides the
external process execution boundary. P1-06 discovers and version-probes llama.cpp. P1-07
builds deterministic llama.cpp argv. P1-08 provides the fixed Standard Prompt. P1-09 parses
llama.cpp prompt/generation timing output into structured metrics:

```python
from lowvram.runtime import parse_llama_cpp_output

parsed = parse_llama_cpp_output(
    stdout=execution.stdout,
    stderr=execution.stderr,
)
```

The parser consumes explicit llama.cpp timing values, normalizes printed milliseconds to
seconds, and returns `parse_failed` for partial, malformed, non-finite, or incomplete output.
It does not derive missing tokens/sec values.

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
- structured llama.cpp timing-output parser
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI

## Not implemented yet

Benchmark orchestration, recommendation logic, aggregation, Web UI, and model downloading
remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/benchmark-orchestrator.md` for P1-10.
