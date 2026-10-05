# P1 Final Consumer Hardware Acceptance Execution Policy

Status: **Normative for P1 Final Gate**  
Policy version: **P1-FINAL-EXECUTION-POLICY-v1**

This policy defines the access and mutation boundary for the developer-controlled
consumer-hardware acceptance that closes P1. It applies to every interactive or remote
execution mechanism used for the gate, including Remote Desktop Commander.

The acceptance session is a constrained verification session, not a development session.
Any action outside this policy is denied. If an out-of-policy action becomes necessary, stop
the gate, keep P1 Final Gate pending, change this policy through a separate reviewed PR, and
restart the full acceptance from a clean `main`. There are no ad-hoc permission expansions
inside an acceptance session.

## 1. Required preconditions

Before any benchmark execution:

1. The target must be a developer-controlled consumer machine.
2. The repository remote must resolve to the official LowVRAM repository.
3. The tracked working tree must be clean.
4. Synchronize with `origin/main` using fast-forward-only Git operations.
5. Record and freeze the resulting `main` commit SHA for the complete acceptance.
6. Use the P1-15 pinned llama.cpp/runtime and GGUF provenance. A platform that cannot execute
   the pinned runtime does not substitute a different runtime during the gate.
7. The environment source passed to the seed runner must be
   `developer-controlled-consumer-hardware`.

After the commit SHA is frozen, tracked repository content is read-only for the remainder of
the acceptance.

## 2. Access roots

At session start, resolve these symbolic roots:

- `<repo>`: the LowVRAM repository root.
- `<acceptance-workspace>`: `<repo>/p1-final-seed/`.
- `<venv>`: `<repo>/.venv/`.
- `<runtime-cache>`: `<repo>/tools/`.
- `<model-cache>`: `<repo>/models/`.

### Allowed read access

The remote executor may read:

- tracked content under `<repo>`;
- files it creates under the four writable roots listed below;
- operating-system/runtime files implicitly required to execute whitelisted programs;
- read-only hardware and process metadata exposed by the whitelisted inspection commands.

It must not enumerate or inspect unrelated user files merely because the operating system
allows access to them.

### Allowed write access

Writes are limited to:

- `<acceptance-workspace>`;
- `<venv>`;
- `<runtime-cache>`;
- `<model-cache>`;
- Git metadata and tracked files only during the pre-freeze fast-forward synchronization step.

The acceptance should create a new evidence workspace rather than deleting an old one.

### Tracked repository files

After the frozen SHA is recorded, these are read-only, including but not limited to:

- `lowvram/`;
- `tests/`;
- `scripts/`;
- `docs/`;
- `schemas/`;
- `data/`;
- `.github/`;
- `pyproject.toml`;
- `README.md` and `README_CN.md`.

No local source or test edit may be used to make the acceptance pass.

## 3. Command whitelist

Only the following command families may be executed. Arguments must stay within the access
roots and purposes defined by this policy.

### Repository verification and synchronization

Allowed:

- `pwd`, `cd <repo>`;
- `git status --short --branch`;
- `git rev-parse HEAD`;
- `git branch --show-current`;
- `git remote get-url origin`;
- `git fetch origin main`;
- `git checkout main` only after confirming the tracked tree is clean;
- `git pull --ff-only origin main`;
- read-only `git diff`, `git log`, and `git show` operations.

Not allowed: stashing, rebasing, committing, resetting, cleaning, force operations, changing
Git configuration, changing remotes, or writing directly to `main`.

### Filesystem inspection and workspace preparation

Allowed only inside the permitted roots:

- `ls`, `dir`, `find`, `stat`, `Get-Item`, `Get-ChildItem`;
- `mkdir` / `New-Item -ItemType Directory`;
- `chmod +x` for the pinned llama.cpp executable;
- archive extraction into `<runtime-cache>`;
- checksum verification with `sha256sum`, `shasum -a 256`, or
  `certutil -hashfile ... SHA256`.

General recursive deletion is not part of the acceptance command set.

### Python environment and project checks

Allowed:

- `python --version` or `python3 --version`;
- `python -m venv <venv>`;
- the virtual environment's Python running
  `-m pip install ".[dev]"`;
- the virtual environment's `ruff check .`;
- the virtual environment's `mypy lowvram scripts`;
- the virtual environment's `pytest`;
- the virtual environment's Python running the repository validation scripts required by CI.

System package managers are not allowed. If the required Python runtime cannot be used without
system-level installation or modification, stop the gate.

### Hardware and process inspection

Read-only inspection is limited to the platform-equivalent forms of:

- `uname`, `lscpu`, `free`, `systeminfo`, read-only `Get-CimInstance`;
- `nvidia-smi`;
- `ps`, `tasklist`, read-only `Get-Process`.

Process termination is allowed only for the benchmark process tree created by the current
acceptance attempt. Unrelated processes must not be stopped.

### Pinned artifact acquisition

Network downloads are limited to:

- the repository's official Git remote for the synchronization step;
- the exact llama.cpp release and GGUF artifact locations pinned by P1-15;
- Python packages needed by the dependency declarations in `pyproject.toml`, installed only
  into `<venv>`.

Allowed download tools are `curl` or the platform-equivalent
`Invoke-WebRequest`, with output restricted to `<runtime-cache>` or
`<model-cache>`.

Downloaded llama.cpp and GGUF files must pass the P1-15 SHA-256 checks before use. A checksum
failure stops the gate.

### LowVRAM acceptance commands

Allowed:

- the virtual environment's `lowvram system`;
- the virtual environment's `lowvram doctor`;
- the virtual environment's `lowvram benchmark --dry-run`;
- the virtual environment's Python running `scripts/run_real_seed.py` with the P1-15 matrix,
  the pinned artifacts, an output location below `<acceptance-workspace>`, and
  `--environment-source developer-controlled-consumer-hardware`.

The ten-attempt P1-15 matrix is unchanged:

- SmolLM2-135M-Instruct Q4_K_M, context 1024 × 3 identical repeat runs;
- SmolLM2-135M-Instruct Q4_K_M, context 2048 × 2;
- Qwen2.5-0.5B-Instruct Q4_0, context 1024 × 2;
- Qwen2.5-0.5B-Instruct Q4_0, context 2048 × 2;
- one intentional real timeout attempt using SmolLM2.

## 4. Explicitly prohibited access and actions

The acceptance session must not:

- use `sudo`, administrator elevation, or equivalent privilege escalation;
- install or remove system packages with `apt`, `dnf`, `yum`, `pacman`, `brew`,
  `winget`, `choco`, or equivalent tools;
- modify system services, startup entries, drivers, firmware, registry, partitions, mounts, or
  security settings;
- inspect browser profiles, email, password stores, keychains, SSH private keys, API tokens,
  cloud credentials, or unrelated project directories;
- read or write unrelated user documents;
- use SSH or another remote-control path to reach additional machines;
- execute arbitrary scripts downloaded from the network;
- use `git reset --hard`, `git clean`, force push, history rewriting, or direct `main`
  commits;
- change tracked LowVRAM source, tests, schemas, workflows, or documentation after the SHA
  freeze;
- edit generated benchmark evidence to convert a failure into a pass;
- terminate unrelated processes;
- upload raw local paths, credentials, or private run logs as public benchmark data.

## 5. Evidence rules

The acceptance must preserve the evidence produced by the existing P1-15 runner and audit:

- exact dry-run argv versus executed argv;
- runtime identity and version;
- prompt version and prompt SHA-256;
- model and runtime checksums;
- the expected three-file run directory for every attempt;
- success/failure semantics, including the intentional timeout;
- RAM and VRAM semantics;
- timing and repeatability;
- bounded logs;
- truthful `load_time_seconds` handling;
- path-free public BenchmarkRun exports;
- the frozen repository SHA and consumer-hardware environment provenance.

Raw local run directories remain private evidence. Public repository updates must not publish
private local paths or unrelated machine data.

## 6. Defect and policy-violation handling

If a product defect is found:

1. stop the acceptance;
2. preserve the evidence;
3. keep P1 Final Gate `PENDING`;
4. fix the defect on a separate development branch through the normal PR/CI path;
5. merge the fix;
6. synchronize a clean `main`;
7. freeze the new SHA;
8. rerun the entire consumer-hardware acceptance matrix.

If an out-of-policy command or access is required, use the same stop-and-restart model. Do not
grant an in-session exception.

If a prohibited modification or access actually occurs, the current acceptance session is
invalid and cannot produce a P1 Final Gate `PASS`.

## 7. Gate decision

P1 Final Gate may become `PASS` only when all of the following are true:

- the target is developer-controlled consumer hardware;
- the frozen `main` SHA remains unchanged throughout execution;
- the full P1-15 matrix completes under this policy;
- the evidence audit passes;
- no prohibited access or modification occurred.

Only after that decision may the separate minimal status PR change the project state from
P1 Final Gate `PENDING` to `PASS` and mark P1 complete.
