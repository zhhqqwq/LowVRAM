# llama.cpp Command Builder — P1-07

P1-07 converts normalized LowVRAM inputs into one deterministic, immutable llama.cpp command
record.

## Rebaseline acceptance contract

Under the restarted P1 sequence, P1-07 must cover:

- model path;
- context;
- threads;
- GPU layers;
- batch size;
- temperature;
- seed;
- deterministic argv ordering;
- complete actual command recording;
- path preservation;
- ordered `extra_args`;
- managed-flag conflict rejection;
- CPU-only `gpu_layers=0`;
- Windows/Linux path strings;
- a canonical bridge to the P1-05 Runtime Adapter.

## Interface

```python
from lowvram.models import LlamaCppCommandRequest
from lowvram.runtime import (
    build_llama_cpp_command,
    to_runtime_execution_request,
)

request = LlamaCppCommandRequest(
    executable="/opt/llama.cpp/llama-cli",
    model_path="/models/model.gguf",
    context_length=4096,
    threads=8,
    gpu_layers=32,
    batch_size=512,
    temperature=0.8,
    seed=42,
    extra_args=("--log-disable",),
)

command = build_llama_cpp_command(request)
execution = to_runtime_execution_request(command, timeout_seconds=300)
```

## Deterministic mapping

The managed parameters are emitted in this fixed order:

```text
--model
--ctx-size
--threads
--gpu-layers
--batch-size
--temp
--seed
```

Then `extra_args` are appended in caller-provided order.

The current llama.cpp CLI continues to expose the corresponding context, batch, GPU-layer,
temperature, and seed options. LowVRAM uses the long forms above as its normalized command
representation.

## One command truth

`LlamaCppCommand` records:

- `executable`;
- immutable `arguments`;
- immutable complete `argv`.

The model requires:

```text
argv == (executable, *arguments)
```

and is frozen after validation. This prevents post-validation mutation from making the saved
full command disagree with the arguments that were originally built.

`to_runtime_execution_request()` is the canonical bridge to P1-05. It derives the execution
request directly from the immutable command record, so later orchestration does not need to
reconstruct llama.cpp arguments independently.

## Path and shell semantics

Executable and model paths remain one argv item even when they contain spaces or Windows
backslashes.

P1-07 does not build a shell command string. P1-05 continues to execute the derived request
with `shell=False`.

NUL bytes are rejected because they are not valid operating-system argv/path content.

## extra_args policy

`extra_args` is stored as a tuple and the request validates assignment. Callers therefore
cannot validate a safe request and then mutate the same object to append a managed override.

The following managed parameters and their aliases cannot appear in `extra_args`:

```text
model
context
threads
gpu layers
batch
temperature
seed
```

Both exact flags and `--flag=value` forms are rejected.

Distinct options that merely share a textual prefix are not rejected. For example,
`--threads-batch` is separate from `--threads`.

## Numeric boundaries

- context, threads, and batch must be positive integers;
- `gpu_layers=0` is valid for CPU-only execution;
- negative GPU layers are invalid;
- temperature must be finite and non-negative;
- seed remains an integer, including `-1` when explicitly requested.

## P0 Recipe boundary

P0 `Recipe` remains unchanged. It does not contain temperature, seed, executable path, or
model filesystem path.

P1-07 continues to use the P1-specific `LlamaCppCommandRequest` rather than silently
changing the P0 JSON contract.

## Boundary

P1-07 builds and records argv only. It does not inject the Standard Prompt, execute the model
workload, parse performance output, or construct Benchmark JSON.

After this rebaseline passes, P1-08 Standard Prompt is the next active Gate.
