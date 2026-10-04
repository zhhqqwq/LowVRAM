# llama.cpp Detector — P1-06

P1-06 discovers a local llama.cpp executable and probes its version before benchmark work
uses the P1-05 Runtime Adapter.

## Interface

```python
from lowvram.runtime import detect_llama_cpp

result = detect_llama_cpp()
```

An explicit path takes priority:

```python
result = detect_llama_cpp("/opt/llama.cpp/llama-cli")
```

If an explicit path is supplied but is not a runnable file, detection returns a not-found
result. It does not silently choose a different executable from PATH.

## Candidate names

Linux and other POSIX systems search:

```text
llama-cli
main
```

Windows searches:

```text
llama-cli.exe
llama-cli
main.exe
main
```

`llama-cli` is preferred. `main` remains as a compatibility candidate for older llama.cpp
builds.

## Version probing

For a discovered executable, LowVRAM probes in this order:

```text
--version
version
```

The detector parses common `version:`, `llama.cpp version`, `build:`, and `bNNNN`
forms from stdout/stderr.

The structured result records discovery source, candidate name, executable, runnable state,
parsed version, successful version command, and `verified_eligible`.

## Verified benchmark rule

The original P1 requirement is enforced directly:

> A llama.cpp version that cannot be recognized must not produce a Verified Benchmark.

Accordingly, `verified_eligible` is true only when the executable is runnable and a version
was recognized.

A runnable executable with unrecognized version output is still reported as found so the
caller can show a useful diagnostic, but it is not verification-eligible.

## Boundary

P1-06 does not build model benchmark commands. Recipe-to-llama.cpp argument mapping belongs
to P1-07. It does not execute a model, parse performance metrics, or create benchmark records.
