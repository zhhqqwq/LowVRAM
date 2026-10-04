# P1-15 Real Benchmark Seed

P1-15 is the final P1 truthfulness gate.

## Required matrix

The automated real-seed workflow executes ten attempts with real llama.cpp and real GGUF
files:

- SmolLM2-135M-Instruct Q4_K_M, context 1024 × 3 identical repeat runs;
- SmolLM2-135M-Instruct Q4_K_M, context 2048 × 2;
- Qwen2.5-0.5B-Instruct Q4_0, context 1024 × 2;
- Qwen2.5-0.5B-Instruct Q4_0, context 2048 × 2;
- one real timeout failure using SmolLM2.

Generation is capped with `--n-predict` so Standard Prompt v1 remains the input while the
seed workflow has bounded runtime.

## Pinned provenance

The CI validation workflow pins:

- llama.cpp release `b10336`, Ubuntu x64 CPU artifact;
- SmolLM2 Q4_K_M SHA-256
  `8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684`;
- Qwen2.5 0.5B Q4_0 SHA-256
  `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`.

## Evidence and acceptance

Each attempt must publish the P1-14 three-file run directory. The seed runner also exports a
path-free BenchmarkRun JSON and validates prompt/runtime identity, dry-run argv equality,
positive finite throughput, RAM/VRAM semantics, repeatability, bounded log sizes, explicit
load-time availability semantics, and private-path exclusion.

The GitHub-hosted workflow is real execution evidence, not a mocked benchmark. The original
P1-15 plan explicitly asks for runs on the developer's own computer, so CI evidence can pass
the technical real-seed checks while the overall P1 Final Gate remains blocked until the same
acceptance runs on a developer-controlled consumer machine.


## Current llama-cli timing compatibility

Pinned llama.cpp b10336 prints a compact timing summary instead of the older detailed
`prompt eval time / eval time` block. LowVRAM parses both formats. For compact timing,
prompt/generation throughput is recorded directly while prompt/eval durations and
`load_time_seconds` remain null because the CLI did not expose those values separately.

Do not derive model load time from total subprocess duration.
