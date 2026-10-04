# llama.cpp Command Builder — P1-07

P1-07 converts normalized LowVRAM inputs into deterministic llama.cpp argv.

## Interface

```python
from lowvram.models import LlamaCppCommandRequest
from lowvram.runtime import build_llama_cpp_command

request = LlamaCppCommandRequest(
    executable="/opt/llama.cpp/llama-cli",
    model_path="/models/model.gguf",
    context_length=4096,
    threads=8,
    gpu_layers=32,
    batch_size=512,
    temperature=0.8,
    seed=42,
    extra_args=["--log-disable"],
)

command = build_llama_cpp_command(request)
```

The returned `LlamaCppCommand` contains:

- `executable`
- `arguments`
- `argv`

`argv` is the complete actual command record and is the unambiguous execution form. P1-07
does not convert it into a shell command string.

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

Then `extra_args` are appended in the caller-provided order.

Model and executable paths are single argv items, including paths containing spaces and
Windows-style paths.

## extra_args policy

`extra_args` can add llama.cpp options not managed by the request model.

It cannot repeat or override any managed parameter, including short or long aliases. For
example, `--ctx-size`, `-c`, and `--ctx-size=8192` are rejected inside
`extra_args`.

This prevents a benchmark record from claiming one normalized configuration while the actual
runtime receives a second conflicting value later in argv.

## CPU-only

`gpu_layers=0` is valid and is emitted explicitly:

```text
--gpu-layers 0
```

Negative GPU-layer counts are rejected by the model.

## P0 Recipe boundary

P0 `Recipe` intentionally remains unchanged in P1-07. It does not contain temperature,
seed, executable path, or model filesystem path.

P1-07 therefore introduces the P1-specific `LlamaCppCommandRequest` rather than silently
changing the P0 JSON contract. A later orchestrator can construct this request from Recipe,
model artifact location, detected runtime, and benchmark-standard parameters.

## Boundary

P1-07 only builds argv. It does not execute llama.cpp, add the benchmark prompt, parse
performance output, or create Benchmark JSON.

P1-08 owns the fixed Standard Prompt.
