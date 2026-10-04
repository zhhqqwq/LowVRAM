# Failure Classification — P1-11

P1-11 turns benchmark failure evidence into the stable LowVRAM error categories defined in
P0.

## Stable categories

```text
runtime_not_found
model_not_found
model_load_failed
out_of_memory
process_crash
timeout
parse_failed
unknown
```

Failure records remain useful benchmark evidence; P1-11 does not discard RAM/VRAM/runtime
output merely because the attempt failed.

## Precedence

Classification uses this precedence:

```text
1. preflight model absence
2. structured runtime_not_found
3. structured model_not_found
4. structured timeout
5. already-structured out_of_memory / model_load_failed
6. runtime-execution text evidence:
     out_of_memory
     model_not_found
     model_load_failed
7. output-parse failure
8. non-zero process exit
9. unknown fallback
```

The purpose is to prefer lifecycle facts over text guesses. A timeout remains `timeout` even
if partial stderr also contains an out-of-memory phrase.

A generic non-zero exit is only refined when strong runtime evidence exists.

## OOM evidence

P1-11 recognizes explicit allocation failures such as:

- `out of memory`;
- CUDA memory-allocation failures;
- `cudaErrorMemoryAllocation`;
- `std::bad_alloc`;
- `cannot allocate memory`;
- explicit `failed to allocate ... MB/bytes/buffer/memory` diagnostics.

A line that merely mentions "memory" is not sufficient.

The classifier does not attempt to publish a separate CPU-OOM versus VRAM-OOM error type;
both map to the existing stable `out_of_memory` category. The matched rule still records
which evidence family triggered classification.

## Model failures

A missing model is identified from:

- the P1-10 preflight file check; or
- strong runtime evidence such as a GGUF open failure that explicitly says the file does not
  exist.

A model that exists but cannot be parsed/loaded maps to `model_load_failed` when llama.cpp
reports an explicit model-load/open failure.

OOM has precedence over model-load failure because llama.cpp can emit both while a model load
fails specifically due to allocation exhaustion.

## stdout / stderr evidence

P1-11 may inspect stdout, stderr, and an upstream error message. The structured result stores
only:

- the stable error type;
- a rule identifier;
- evidence source categories such as `stderr`;
- the normal benchmark error message.

It does not copy the matched raw log line into classification metadata. This avoids needlessly
duplicating absolute model paths or other local text that may appear in runtime diagnostics.

## Runtime-start OOM

P1-05 now maps OS process-start memory exhaustion to `out_of_memory` using POSIX
`ENOMEM` and common Windows not-enough-memory error codes. Missing/unusable executables
remain `runtime_not_found`; unrelated OS startup errors remain `unknown`.

## Orchestrator integration

`BenchmarkOrchestrationRecord` now uses schema version `p1.11.0`.

Every failed orchestration record must contain `failure_classification`, and its error type
and message must exactly match `result`. Successful records cannot contain a failure
classification.

This makes P1-11 classification part of saved evidence instead of a transient display-only
decision.

## Boundary

P1-11 does not implement recovery, automatic parameter reduction, Doctor, Dry Run, run
directory logging, or public submission.

P1-12 Doctor Command is the next Gate.
