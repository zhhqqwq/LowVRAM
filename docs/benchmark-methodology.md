# Benchmark methodology — P0 contract

P0 defines what a benchmark record means; it does not execute a model.

## What counts as one run

One run is one attempt to execute one identified model artifact using one runtime and one explicit recipe on one captured hardware/software environment. A retry is a new run and must receive a new `run_id`.

## Required metrics for a successful run

A successful run records load time in seconds, prompt throughput in tokens/sec, generation throughput in tokens/sec, peak VRAM in MB, and peak RAM in MB.

## Failed runs

Failures are preserved because they are compatibility evidence. A failed run sets `success=false`, records an `error_type`, and may contain null metrics that were never observed. Do not replace unavailable measurements with invented zeros.

## Immutable semantics

Do not manually alter measured performance or memory values to make a run appear more plausible. Do not change units. Do not silently drop unknown fields. Later runner phases should preserve raw evidence separately from normalized benchmark JSON.

## P0 limitation

P0 fixtures are synthetic contract examples, not real performance claims. Runtime execution, monitoring, parsing, standard prompts, and verified benchmark generation belong to P1.
