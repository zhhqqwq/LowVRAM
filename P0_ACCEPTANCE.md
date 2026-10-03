# P0 Acceptance Report

Date: 2026-10-03

This report checks only **P0 — Foundation**. No P1/P2/P3 functionality is accepted here.

## Scope delivered

- Python 3.11+ package named `lowvram`
- Typer CLI with `lowvram --help` and `lowvram validate <file>`
- Strict Pydantic v2 models: `HardwareInfo`, `ModelInfo`, `BenchmarkRun`, `Recipe`
- JSON Schema Draft 2020-12 files for all four contracts
- Fixed units: memory = MB, duration = seconds, throughput = tokens/sec
- Successful and failed benchmark records
- CPU-only, single-GPU, multi-GPU, MoE, missing-field, invalid-memory, failed-run tests
- 5 valid benchmark fixtures and 5 invalid benchmark fixtures
- GitHub Actions CI for install, Ruff, mypy, pytest, data validation, fixture validation, and schema drift
- P0 documentation: data format, architecture, benchmark methodology

## Local checks

| Check | Result | Exact outcome |
| --- | --- | --- |
| `python -m pip install -e .` | BLOCKED BY ENVIRONMENT | Build isolation tried to download `setuptools>=68`; sandbox DNS/network access is unavailable. |
| `python -m pip install -e . --no-build-isolation` | PASS | Editable wheel built and `lowvram-0.1.0` installed. |
| `python -c "import lowvram"` | PASS | Import succeeds. |
| `lowvram --help` | PASS | Shows the `validate` subcommand. |
| Valid benchmark CLI validation | PASS | Prints `VALID`, exit code 0. |
| Invalid negative-VRAM CLI validation | PASS | Prints `INVALID`, identifies `hardware.gpu.0.vram_total_mb`, exit code 1. |
| `pytest -q` | PASS | `28 passed`. |
| `python scripts/validate_fixtures.py` | PASS | `Fixture validation PASS`. |
| `python scripts/validate_data.py` | PASS | No P0 data files are present; directories intentionally empty. |
| `python scripts/export_schemas.py` | PASS | All four schemas regenerate successfully. |
| Draft 2020-12 schema validity | PASS | Covered by automated tests using `Draft202012Validator.check_schema`. |
| `python -m compileall -q lowvram scripts` | PASS | No syntax/bytecode compilation errors. |
| Wheel build/install smoke test | PASS | Wheel builds; installed CLI validates a benchmark when runtime dependencies are made available. |
| `ruff check .` | BLOCKED BY ENVIRONMENT | `ruff` is not installed and the sandbox cannot download it. |
| `mypy` | BLOCKED BY ENVIRONMENT | `mypy` is not installed and the sandbox cannot download it. |

## Gate status

**P0 implementation: delivered.**

**P0 Gate certification: pending CI**, because the project specification requires Ruff and mypy to pass and those two tools cannot be executed in the offline sandbox. The checked-in GitHub Actions workflow makes both mandatory before the Gate should be marked PASS.

## Scope audit

The P0 package contains models, validators, and the validation CLI only. It does not implement benchmark execution, llama.cpp subprocess integration, GPU monitoring, recommendations, aggregation, model downloading, or a Web UI.
