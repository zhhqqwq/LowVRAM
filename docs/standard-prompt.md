# Standard Prompt — P1-08

P1-08 introduces the first fixed, versioned input workload used by LowVRAM benchmarks.

## Asset

The canonical source file is:

```text
benchmark_prompts/v1.txt
```

Its registered identity is:

```text
prompt_version = v1
sha256 = 83192002c73012391d251db70fb58dac2eb8ada79e9b56a8d92ac0f4e76eb75d
```

The prompt is synthetic and self-contained. It does not require network access, external
knowledge, personal information, or machine-specific values.

## Loader

```python
from lowvram.prompts import load_benchmark_prompt

prompt = load_benchmark_prompt()
```

The returned `BenchmarkPrompt` contains:

- `prompt_version`
- exact `content`
- exact UTF-8 `sha256`

The loader checks the registered SHA-256 before returning the prompt.

## Integrity behavior

A registered version is immutable. Editing `v1.txt` without creating a new version causes
`PromptIntegrityError`.

A missing file causes `PromptNotFoundError`.

An unregistered version such as `v2` causes `UnsupportedPromptVersionError`.

Do not update the v1 hash merely to make an accidental edit pass. A deliberate prompt change
must create a new versioned file and a new registered digest.

## Packaging

The repository keeps the reviewable canonical asset at `benchmark_prompts/v1.txt`.
Setuptools also installs it under `share/lowvram/benchmark_prompts`, allowing the loader to
work from both a source checkout/editable install and an installed distribution.

## Boundary

P1-08 does not attach the prompt to llama.cpp argv and does not execute a benchmark.
P1-10 Benchmark Orchestrator will combine the Standard Prompt with P1-07 command construction
and P1-05 process execution.

P1-09 is next and owns llama.cpp performance-output parsing.
