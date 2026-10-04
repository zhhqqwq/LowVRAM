# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner is in progress.**

## Current capability

P1-01 through P1-04 provide hardware inventory plus RAM/VRAM measurement. P1-05 provides the
external process execution boundary. P1-06 adds llama.cpp executable discovery and version
probing:

```python
from lowvram.runtime import detect_llama_cpp

detected = detect_llama_cpp()
```

Detection supports an explicit executable path or PATH search, prefers modern `llama-cli`,
supports legacy `main`, covers Windows/Linux executable names, and probes `--version`
then `version`. A detected runtime is verification-eligible only when its version can be
recognized.

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
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- P0 plus P1 collector/monitor/runtime documentation

## Not implemented yet

Recipe-to-llama.cpp command translation, standard benchmark prompt, performance parsing,
benchmark orchestration, recommendation logic, aggregation, Web UI, and model downloading
remain later work.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/runtime-adapter.md` and `docs/llama-cpp-detector.md`.
