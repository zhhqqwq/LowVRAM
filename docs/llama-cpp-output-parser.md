# llama.cpp Output Parser — P1-09

P1-09 converts captured llama.cpp textual timing output into four trusted performance
measurements.

## Scope

The parser owns exactly:

- prompt eval time;
- prompt tokens/sec;
- generation eval time;
- generation tokens/sec.

It accepts the stdout and stderr captured by P1-05.

It does not execute llama.cpp, infer missing throughput, construct Benchmark records, or
orchestrate RAM/VRAM monitoring.

## Interface

```python
from lowvram.runtime import parse_llama_cpp_output

result = parse_llama_cpp_output(
    stdout=execution.stdout,
    stderr=execution.stderr,
)
```

A successful result contains `LlamaCppTimingMetrics`:

```text
prompt_eval_time_seconds
prompt_tokens_per_second
eval_time_seconds
generation_tokens_per_second
```

Durations are normalized from the milliseconds printed by llama.cpp into LowVRAM's fixed
seconds unit.

Tokens/sec are read directly from llama.cpp output. P1-09 never calculates tokens/sec from
milliseconds-per-token or token counts.

## Supported textual timing forms

P1-09 keys off the timing labels rather than a logger/function prefix. This supports forms
such as:

```text
llama_print_timings: prompt eval time = ... ms / ... tokens (... ms per token, ... tokens per second)
llama_print_timings:        eval time = ... ms / ... runs   (... ms per token, ... tokens per second)
```

and newer prefixes such as:

```text
llama_perf_context_print: prompt eval time = ...
llama_perf_context_print:        eval time = ...
```

`generation eval time` is also accepted as an eval-label variant.

The parser requires the explicit `tokens per second` value in each required line.

## stdout and stderr

P1-05 captures stdout and stderr separately and therefore does not retain their original
interleaving.

P1-09 deterministically scans stdout first and stderr second. This supports the common case
where generated text is stdout and timing logs are stderr, while still accepting timing
output in either stream.

A prompt timing line in stdout may pair with a following eval timing line in stderr.

## Duplicate timing blocks

For repeated runs or interactive output, a later complete timing block replaces an earlier
complete block.

The latest timing sequence must itself be complete and valid. If output ends with a new
prompt timing line but no matching eval timing line, P1-09 returns `parse_failed` instead of
silently publishing stale metrics from an earlier block.

## Numeric validity

All four parsed values must be finite and strictly positive.

Examples rejected as `parse_failed` include:

- zero time;
- zero throughput;
- negative values;
- `inf`;
- `nan`;
- missing tokens/sec;
- malformed numeric text.

This is intentionally stricter than some raw llama.cpp diagnostic output. A Benchmark
performance record must not contain a non-finite or zero throughput measurement.

## Failure contract

Malformed, partial, missing, or numerically invalid timing output returns:

```text
success = false
metrics = null
error_type = parse_failed
```

The parser does not fabricate zeros and does not derive missing throughput.

## Boundary

P1-09 is textual parsing only.

P1-10 Benchmark Orchestrator is the next Gate and will compose:

- P1-03 RAM Monitor;
- P1-04 VRAM Monitor;
- P1-05 Runtime Adapter;
- P1-06 Detector;
- P1-07 Command Builder;
- P1-08 Standard Prompt;
- P1-09 Output Parser.
