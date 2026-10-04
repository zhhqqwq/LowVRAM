# Doctor Command — P1-12

P1-12 provides a non-model-executing readiness check:

```bash
lowvram doctor
```

The original P1-12 contract checks Python, llama.cpp, NVIDIA, CUDA, permissions, disk, and
RAM, then reports `READY` or explicit blockers.

## Commands

Human-readable diagnostics:

```bash
lowvram doctor
```

Machine-readable diagnostics:

```bash
lowvram doctor --json
```

Optional local-path checks:

```bash
lowvram doctor \
  --llama-cli /path/to/llama-cli \
  --model /path/to/model.gguf \
  --output-dir /path/to/runs
```

The model is never loaded. `--model` only checks that the path is a readable, non-empty
regular file.

llama.cpp is only version-probed through the accepted P1-06 detector. Doctor does not invoke
a model command.

## Structured result

`DoctorResult` uses schema version `p1.12.0` and contains:

- `ready`;
- `summary` = `READY` or `BLOCKED`;
- ordered checks;
- ordered blocking-check identifiers.

Every check has a stable name, status, message, and small structured details map.

Statuses are:

```text
pass
warn
block
```

Only `block` prevents `READY`.

## Readiness policy

Blocking baseline checks are:

- Python requirement not satisfied;
- llama.cpp missing, not runnable, or version not recognized;
- explicitly requested model missing/unreadable/empty;
- output directory missing/not a directory/not writable;
- output filesystem disk information unavailable or no free space;
- system RAM detection failure.

A recognized llama.cpp version is required because P1-06 already established that an
unrecognized version cannot produce a Verified Benchmark.

## NVIDIA and CUDA

LowVRAM supports CPU-only benchmarks.

Therefore:

- no NVIDIA GPU → warning, not blocker;
- no CUDA when there is no NVIDIA GPU → warning, not blocker;
- NVIDIA GPU found but CUDA compatibility not reported by nvidia-smi → warning, not global
  blocker because CPU-only execution remains possible.

The structured checks still expose GPU count, driver version, and CUDA compatibility when
available.

## Permissions and disk

The output directory is tested by creating and deleting one small temporary file. Doctor
does not leave the probe file behind when cleanup succeeds.

Disk availability is read with the operating system filesystem API. P1-12 reports total and
free MB but intentionally does not invent a universal minimum-free-space threshold; actual
requirements depend on later run artifacts and model size.

If `--model` is supplied, Doctor reports exact file size but does not read model contents.

## Exit codes

```text
READY   -> exit 0
BLOCKED -> exit 1
```

The same exit-code contract applies to human and `--json` output.

## Boundary

P1-12 does not:

- load or execute a model;
- construct benchmark argv;
- auto-fix missing dependencies;
- change runtime settings;
- implement `benchmark --dry-run`.

P1-13 Dry Run is the next Gate.
