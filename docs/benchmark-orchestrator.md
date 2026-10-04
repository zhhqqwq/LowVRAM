# Benchmark Orchestrator — P1-10

P1-10 composes the accepted P1 collectors/runtime components into one benchmark attempt.

## Lifecycle

The orchestration path is:

```text
collect system environment
↓
validate model file
↓
detect llama.cpp
↓
load Standard Prompt
↓
build exact command
↓
spawn runtime and expose PID
↓
start RAM + VRAM monitors for that PID
↓
wait for runtime / timeout
↓
stop monitors
↓
parse stdout + stderr timings
↓
build strict orchestration record
↓
JSON Schema validate
↓
optional save
```

The key P1-10 change is the separation of `spawn()` from `wait()`.
`RuntimeProcessSession.pid` is available immediately after `Popen`, allowing P1-03 and
P1-04 to attach while the runtime is live. Existing `execute()` remains compatible and is
implemented as spawn + wait.

## Standard Prompt ownership

P1-10 always injects the registered P1-08 prompt with `--prompt`.

Recipe `extra_args` cannot provide `-p`, `--prompt`, `-f`, `--file`, or
`--prompt-file`, including `--flag=value` forms. This prevents callers from silently
changing the benchmark workload while retaining `prompt_version=v1`.

The exact prompt-bearing argv is preserved in `LlamaCppCommand`.

## P1 orchestration record

P1-10 produces `BenchmarkOrchestrationRecord`, a strict serializable evidence object with:

- run ID and timestamp;
- P1-01 hardware snapshot;
- model metadata and P0 Recipe;
- P1-06 detection result;
- prompt version and SHA-256;
- immutable P1-07 command;
- P1-05 execution result;
- P1-03 RAM result;
- P1-04 VRAM result;
- P1-09 timing metrics;
- success/failure result.

Before a record is saved, the JSON payload is validated against
`BenchmarkOrchestrationRecord.model_json_schema()` using JSON Schema Draft 2020-12.

## Why this is not yet the P0 BenchmarkRun

The existing P0 `BenchmarkRun` success contract requires `load_time_seconds`.

The original P1-09 scope parses prompt eval time, prompt tokens/sec, eval time, and generation
tokens/sec; it does not provide load time. P1-10 therefore does not substitute total runtime
duration for model load time and does not fabricate a value merely to satisfy P0.

Until a trustworthy load-time source is wired into the final Benchmark mapping,
`BenchmarkOrchestrationRecord` is the P1-10 evidence JSON. It preserves every measurement
needed to complete the mapping without losing provenance.

## Failures

P1-10 already preserves failures in the orchestration record, including obvious preconditions
such as `model_not_found`, runtime spawn errors, runtime timeout/crash, and P1-09
`parse_failed`.

P1-11 remains responsible for the complete failure-classification policy.

## Monitor start window

A cross-platform `subprocess.Popen` child begins executing before Python receives its PID.
P1-10 starts both monitors immediately after `spawn()` returns and before `wait()`.

This closes the previous architectural gap where PID was completely hidden, but a small
spawn-to-monitor observation window remains. LowVRAM does not claim that this is equivalent to
OS-level suspended process creation. Real benchmark seed runs must be used to assess whether
that window materially affects early memory peaks.

## Boundary

P1-10 does not implement:

- the complete P1-11 failure classifier;
- `lowvram doctor`;
- dry-run CLI;
- run-directory stdout/stderr logging policy;
- public-database submission.

Those remain later P1/P2 Gates.
