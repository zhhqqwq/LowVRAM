# LowVRAM data format — P0

All machine-generated records are JSON. Checked-in schemas use JSON Schema Draft 2020-12. Runtime validation uses strict Pydantic v2 models with unknown fields forbidden.

## Units

Units are part of the field name or contract and must not vary:

- memory: MB (`*_mb`)
- duration: seconds (`*_seconds`)
- throughput: tokens/sec (`*_tokens_per_second`)
- parameter counts: integer parameters

Do not submit GB in a field ending in `_mb`, and do not convert tokens/sec to tokens/minute.

## HardwareInfo

Required top-level fields are `cpu`, `gpu`, `ram_total_mb`, `os`, and `driver`. `gpu` is a list, so CPU-only machines use `[]` and multi-GPU machines contain multiple entries. P0 accelerator records accept NVIDIA GPUs only.

GPU records include `vendor`, `name`, `vram_total_mb`, `driver_version`, and `cuda_version`. Negative VRAM is invalid.

## ModelInfo

Required identity fields include `id`, `name`, `source`, `architecture`, `parameter_count`, `model_type`, `format`, and `quantization`. P0 formats are GGUF and model type is LLM. Architecture is `dense`, `moe`, or `unknown`.

For MoE, `parameter_count` is the total parameter count and `active_parameter_count` is mandatory. Active parameters cannot exceed total parameters.

## BenchmarkRun

A benchmark contains `schema_version`, `run_id`, `timestamp`, `hardware`, `model`, `runtime`, `configuration`, `memory`, `performance`, `result`, and `verification`.

Successful runs record peak VRAM, peak RAM, prompt throughput, and generation throughput. P1-15 clarifies that load time is nullable when the selected runtime format does not expose a separately attributable model-load measurement. Failed runs are first-class records: partial metrics may be null, but `result.error_type` is required. This avoids fabricating zero-valued measurements for failures that occur before a metric can be observed.

`configuration.model_id` must equal `model.id`, and `configuration.runtime` must equal `runtime.name`.

## Recipe

A recipe records `model_id`, `runtime`, `context_length`, `gpu_layers`, `kv_cache_type`, `threads`, `batch_size`, and `extra_args`. P0 defines the contract only; it does not construct or execute a llama.cpp command.


## P1-15 Benchmark contract update

P1-15 adds `prompt_version` to every public Benchmark record so performance cannot be
detached from the fixed workload identity. The Benchmark schema version is now `1.1.0`.

Successful exported Benchmark records use only directly observed values:

- `load_time_seconds` records a directly exposed llama.cpp model-load value when available; current compact CLI timing leaves it `null` instead of substituting total process duration;
- `peak_ram_mb` comes from the target process-tree RAM peak;
- CPU-only runs record `peak_vram_mb=0`;
- NVIDIA runs require attributable process VRAM before a successful public record is emitted.

Local executable/model paths stay in private `runs/<run_id>/` evidence and are intentionally
omitted from the public Benchmark contract.
