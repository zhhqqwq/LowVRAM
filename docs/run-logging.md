# Run Logging — P1-14

P1-14 persists one complete local evidence directory for each real benchmark attempt.

## Layout

The default root is `runs/`:

```text
runs/
└── <run_id>/
    ├── benchmark.json
    ├── stdout.log
    └── stderr.log
```

`BenchmarkOrchestrator` enables this logging by default. Library callers may pass
`runs_dir=None` explicitly when they intentionally need an in-memory-only attempt, such as
isolated unit tests.

## benchmark.json

`benchmark.json` is the validated `BenchmarkOrchestrationRecord` already produced by the
P1 orchestrator. It is validated against its Draft 2020-12 Pydantic-generated schema before
the artifact set is committed.

P1-14 does not invent a second benchmark-result contract.

## stdout.log and stderr.log

When a runtime process exists, the two files contain exactly the `stdout` and `stderr`
strings present in `RuntimeExecutionResult`.

The logger:

- writes UTF-8;
- adds no newline;
- performs no shell quoting;
- performs no command reconstruction;
- does not interpret metacharacters.

P1-05 already decodes subprocess streams in text mode, so P1-14 preserves the structured
runtime strings rather than claiming byte-for-byte capture of the original OS pipe.

If failure occurs before a runtime execution result exists, both log files are still created
as empty files. This keeps the run-directory contract stable for preflight and startup
failures.

## Success and failure attempts

Logging happens in the Orchestrator's common `finish()` path. Therefore the same artifact
layout is used for:

- successful runs;
- parse failures;
- model/runtime preflight failures;
- process crashes;
- timeouts;
- classified OOM/model-load failures.

Crash and timeout stdout/stderr are persisted when P1-05 returned partial captured output.

## Atomic directory publication

P1-14 creates a hidden staging directory under the same `runs` root, writes all three files,
then renames that staging directory to `runs/<run_id>/`.

The final run directory therefore does not appear until all three files have been written.
Because staging and final paths share the same parent filesystem, the directory rename is the
commit boundary.

This is an artifact-set consistency guarantee, not a power-loss durability guarantee; P1-14
does not fsync every file and directory entry.

An existing `runs/<run_id>` is never intentionally overwritten.

## Write failures

Artifact persistence failures raise `RunLoggingError`.

The logger removes its staging directory on write/rename failure and does not return a
successful artifact path. The Orchestrator propagates this exception instead of returning a
benchmark success with missing logs.

A logging failure is not reclassified as model failure, OOM, or process crash because it is a
local evidence-persistence failure that happens after the benchmark outcome has been
determined.

## Legacy output_path

The existing optional single-file `output_path` API remains for compatibility. Default
P1-14 run logging is independent and publishes the canonical three-file run directory first.

## Boundary

P1-14 does not add log rotation, compression, remote upload, public-data sanitization, or
streaming subprocess logs directly to disk.

P1-15 Real Benchmark Seed is the next Gate.
