# Benchmark Dry Run — P1-13

P1-13 adds a non-model-executing preview:

```bash
lowvram benchmark --dry-run --model model.gguf
```

The original P1-13 contract requires Dry Run to print the runtime, model, configuration, and
command without actually running the model.

## Shared preparation path

Dry Run and the P1-10 real Orchestrator use the same
`prepare_llama_cpp_benchmark()` function.

That shared path performs:

```text
model-path preflight
↓
P1-06 llama.cpp detection + version probe
↓
verified_eligible check
↓
P1-08 Standard Prompt integrity load
↓
prompt-override rejection
↓
P1-07 exact argv construction
```

Only the real Orchestrator continues from the prepared command to
`SubprocessRuntimeAdapter.spawn()`.

Dry Run accepts only the P1-05 `RuntimeAdapter` interface, which has version-probe
`execute()` but no model-process `spawn()` method. The runtime version probe is allowed;
model execution is not.

## CLI

P1 baseline defaults are explicit and visible:

```text
context       4096
threads       8
gpu_layers    0
batch         512
temperature   0.0
seed          42
prompt        v1
```

They may be overridden:

```bash
lowvram benchmark --dry-run \
  --model model.gguf \
  --llama-cli /path/to/llama-cli \
  --context 8192 \
  --threads 12 \
  --gpu-layers 24 \
  --batch 256 \
  --temperature 0.0 \
  --seed 42
```

Machine-readable output:

```bash
lowvram benchmark --dry-run --model model.gguf --json
```

The JSON schema identity is `p1.13.0`.

## Output

A READY result contains:

- runtime name;
- recognized llama.cpp version;
- model path;
- context;
- threads;
- GPU layers;
- batch size;
- temperature;
- seed;
- prompt version;
- exact immutable command argv.

Human output serializes argv as a JSON array instead of a shell command string. This
preserves argument boundaries and avoids implying shell escaping semantics.

## Blocked preflight

Dry Run returns a structured blocker and no command when preparation fails.

Examples include:

- model path missing;
- llama.cpp missing or not runnable;
- llama.cpp version not recognized;
- Standard Prompt integrity failure;
- a configuration attempting to override the Standard Prompt;
- invalid command construction.

P1-13 also closes an earlier orchestration gap: the P1-10 real run now uses the same shared
preparation path and therefore also blocks an unrecognized llama.cpp version, matching the
P1-06 Verified Benchmark rule.

## Exit codes

```text
dry-run READY   -> 0
dry-run BLOCKED -> 1
benchmark without --dry-run in P1-13 -> 2
```

P1-13 does not expose real benchmark execution through the CLI yet.

## Boundary

P1-13 does not implement run-directory logging, log retention, public submission, or the
P1-15 real benchmark seed.

P1-14 Logging is the next Gate.
