# Standard Prompt — P1-08

P1-08 defines the first fixed, versioned input workload used by LowVRAM benchmarks.

## Rebaseline acceptance contract

Under the restarted P1 sequence, P1-08 must provide:

- `benchmark_prompts/v1.txt` as the reviewable repository copy;
- `prompt_version = v1`;
- fixed, offline, self-contained content;
- no personal or machine-specific information;
- exact SHA-256 integrity verification;
- package-resource loading after installation;
- explicit missing/tampered/unknown-version failures;
- immutable loaded prompt objects;
- packaging that does not depend on a `sys.prefix/share` layout.

## Asset identity

The registered identity is:

```text
prompt_version = v1
sha256 = 83192002c73012391d251db70fb58dac2eb8ada79e9b56a8d92ac0f4e76eb75d
```

The repository keeps:

```text
benchmark_prompts/v1.txt
```

for direct review, and packages the identical bytes as:

```text
lowvram/benchmark_prompts/v1.txt
```

Acceptance tests require the two copies to be byte-for-byte identical and require both to
match the registered SHA-256. A mismatch is a release-blocking test failure.

## Loader

```python
from lowvram.prompts import load_benchmark_prompt

prompt = load_benchmark_prompt()
```

Normal loading uses `importlib.resources` and therefore resolves the prompt from the
installed Python package rather than assuming a platform-specific `sys.prefix/share`
directory.

The returned immutable `BenchmarkPrompt` contains:

- `prompt_version`;
- exact UTF-8 `content`;
- exact `sha256`.

## Byte-level integrity

Integrity is checked against the exact file bytes before UTF-8 decoding.

This means line-ending changes, trailing-byte changes, or any other byte-level edit to
registered `v1` fails with `PromptIntegrityError`; the loader does not silently normalize
the content before hashing.

A registered prompt must also decode as UTF-8.

## Version semantics

A registered version is immutable.

Do not edit `v1.txt` and update the v1 hash merely to make CI pass. A deliberate workload
change must create a new versioned prompt and a new registry entry.

An unsupported version fails before any filesystem/package-resource lookup.

## Error behavior

- missing registered prompt → `PromptNotFoundError`;
- hash mismatch → `PromptIntegrityError`;
- unsupported version → `UnsupportedPromptVersionError`.

A loaded `BenchmarkPrompt` is frozen after validation, so its content/version/hash cannot be
mutated after integrity verification.

## Prompt constraints

The v1 workload is synthetic and self-contained. It:

- requires no network access;
- requires no external knowledge;
- contains no user personal information;
- contains no machine-specific information;
- asks for deterministic numeric/text-processing work from fixed input data.

## Boundary

P1-08 only owns the fixed workload and its identity/integrity.

It does not attach the prompt to llama.cpp argv, execute a model, parse performance output, or
construct Benchmark JSON.

After this rebaseline passes, P1-09 llama.cpp Output Parser is the next active Gate.
