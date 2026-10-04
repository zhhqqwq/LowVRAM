# llama.cpp Detector — P1-06

P1-06 discovers a local llama.cpp executable and probes its version before later benchmark
work uses the P1-05 Runtime Adapter.

## Rebaseline acceptance contract

Under the restarted P1 sequence, P1-06 must cover:

- explicit executable path;
- PATH lookup;
- Windows/Linux candidate names;
- modern `llama-cli` and legacy `main`;
- executable validation;
- `--version` then `version` probing;
- stdout/stderr version parsing;
- structured not-found and diagnostic results;
- found-but-version-unknown state;
- Verified eligibility only for a recognized runtime version.

## Interface

```python
from lowvram.runtime import detect_llama_cpp

result = detect_llama_cpp()
```

An explicit path takes priority:

```python
result = detect_llama_cpp("/opt/llama.cpp/llama-cli")
```

If an explicit path is not a runnable file, detection returns a not-found result and does not
fall back to PATH.

If the explicit file exists but process startup fails, the detector reports the discovered
file as `found=true` with `runnable=false` and preserves the probe error. Discovery and
successful startup are separate facts.

## Candidate names

Linux/POSIX search order:

```text
llama-cli
main
```

Windows search order:

```text
llama-cli.exe
llama-cli
main.exe
main
```

The modern executable is preferred. Legacy `main` remains a compatibility candidate.

If multiple Windows names resolve to the same executable path, that executable is only probed
once.

## Candidate selection

PATH search does not stop merely because the first preferred executable starts.

Selection order is:

1. first candidate with a recognized version;
2. otherwise the first runnable candidate in preferred name order;
3. otherwise the first discovered-but-not-runnable candidate;
4. not-found only when PATH contains no supported candidate names.

This prevents an unversioned `llama-cli` from hiding a later legacy `main` whose version is
recognizable, while still preserving the modern executable as the diagnostic fallback if no
candidate can be verified.

## Version probing

Each candidate is probed in this order:

```text
--version
version
```

stdout and stderr are combined for parsing.

Recognized forms include numeric/version tokens with at least one digit, explicit
`build: NNNN`, and `bNNNN` build identifiers.

Generic values such as:

```text
version: unknown
version: development
```

are not accepted as recognized versions and therefore cannot create Verified eligibility.

## Runnable semantics

A probe proves that the child process actually started only when:

- execution succeeds;
- it starts and exits non-zero (`process_crash`);
- it starts and times out (`timeout`).

A P1-05 startup failure classified as `unknown` does not make `runnable=true`.

## Verified benchmark rule

The original P1 requirement remains:

> A llama.cpp version that cannot be recognized must not produce a Verified Benchmark.

Accordingly:

```text
runnable + recognized version
→ verified_eligible = true

runnable + unknown version
→ verified_eligible = false

discovered + startup failure
→ verified_eligible = false
```

The structured model also rejects a recognized version without a runnable executable and
rejects a `version_command` when no version was recognized.

## Boundary

P1-06 does not construct benchmark argv, inject the Standard Prompt, execute a model workload,
parse performance metrics, or create Benchmark records.

P1-07 Command Builder is the next active Gate after this rebaseline passes.
