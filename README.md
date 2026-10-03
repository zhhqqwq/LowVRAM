# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

This repository is currently **P0 — Foundation only**. P0 defines machine-readable data contracts and validation. It intentionally does **not** run models.

## P0 scope

Implemented in P0:

- Python 3.11+ package and Typer CLI
- Strict Pydantic v2 models for hardware, model metadata, benchmark runs, and recipes
- JSON Schema Draft 2020-12 files for all four contracts
- `lowvram validate <file>`
- Valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI
- Data-format, architecture, and benchmark-methodology documentation

Explicitly out of P0: benchmark execution, llama.cpp integration, GPU/RAM monitoring, recommendations, aggregation, Web UI, and model downloading.

## Install for development

```bash
python -m pip install -e ".[dev]"
lowvram --help
```

## Validate a document

```bash
lowvram validate tests/fixtures/valid/01_dense_single_gpu_success.json
# VALID
```

Invalid input exits non-zero and prints a field path, for example `hardware.gpu.0.vram_total_mb`.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/data-format.md` for the full contract.
