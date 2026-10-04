# LowVRAM

LowVRAM is an open local-AI model compatibility database for consumer hardware. The long-term product answers: **Can my computer run this model, how should I run it, and how fast will it be?**

**P0 — Foundation is complete. P1 — Benchmark Runner implementation is complete; final developer-controlled consumer-hardware acceptance is pending.**

## Current capability

P1-01 through P1-04 provide hardware inventory plus RAM/VRAM measurement. P1-05 provides the
external process execution boundary. P1-06 discovers and version-probes llama.cpp. P1-07
builds deterministic llama.cpp argv. P1-08 provides the fixed Standard Prompt. P1-09 parses
both detailed llama.cpp timing blocks and current compact Prompt/Generation throughput output.

P1-10 provides live-PID benchmark orchestration. P1-11 adds deterministic failure
classification so runtime startup, missing/load failures, CPU/GPU memory exhaustion, crashes,
timeouts, parse failures, and unknown errors are recorded under stable error categories.
P1-12 adds `lowvram doctor` for non-model-executing environment readiness diagnostics.
P1-13 adds `lowvram benchmark --dry-run` using the same pre-spawn preparation path as the
real Orchestrator. P1-14 persists each real benchmark attempt under
`runs/<run_id>/` with benchmark JSON plus stdout/stderr logs.

P1-15 adds pinned real-seed validation with real llama.cpp and GGUF files. The checked
technical seed executes 10 CPU-only attempts across two models and two context sizes,
including one intentional timeout failure. Public BenchmarkRun schema 1.1.0 records
`prompt_version`, keeps local paths out of public records, and leaves
`load_time_seconds=null` when the selected llama.cpp timing format does not directly expose
model-load time.

The GitHub-hosted real-seed validation is genuine runtime evidence, but it is not
developer-controlled consumer hardware. The overall P1 Final Gate therefore remains pending
until the same acceptance matrix is run on a developer-controlled consumer machine.

P0 also provides strict data contracts and validation:

```bash
lowvram validate tests/fixtures/valid/01_dense_single_gpu_success.json
# VALID
```

## Implemented

- Python 3.11+ package and Typer CLI
- strict Pydantic v2 models and JSON Schema Draft 2020-12 contracts
- `lowvram validate <file>`
- `lowvram system`
- hardware/NVIDIA collection plus RAM/VRAM monitoring
- shell-free runtime adapter abstraction
- llama.cpp explicit-path/PATH discovery and version detection
- deterministic llama.cpp command building
- versioned, integrity-pinned Standard Prompt
- detailed and compact llama.cpp timing-output parsing
- live-PID benchmark orchestration with validated evidence JSON
- evidence-based P1 failure classification
- `lowvram doctor` human/JSON readiness diagnostics
- `lowvram benchmark --dry-run` exact-command preview
- atomic three-file run logging under `runs/<run_id>/`
- pinned real llama.cpp/GGUF CPU seed validation
- path-free BenchmarkRun 1.1.0 export with prompt-version provenance
- valid/invalid benchmark fixtures and automated tests
- Ruff, mypy, pytest, and GitHub Actions CI

## Remaining P1 Final Gate

Run the P1-15 acceptance matrix on developer-controlled consumer hardware and review the
resulting evidence before declaring the overall P1 stage complete.

P2 open benchmark-database work and P3 compatibility-product work remain later stages.

## Fixed units

- memory: MB
- duration: seconds
- throughput: tokens/sec

See `docs/p1-15-real-seed.md` for the P1 final real-seed gate.
