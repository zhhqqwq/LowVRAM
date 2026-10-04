# VRAM Monitor — P1-04

P1-04 monitors both whole-GPU VRAM and VRAM attributable to the benchmark process tree.

## Rebaseline acceptance contract

Under the restarted P1 sequence, P1-04 must cover:

- target-process VRAM attribution;
- whole-system/GPU memory usage;
- baseline, current, peak, and delta;
- 100 ms sampling;
- multiple GPUs;
- target process exit;
- CPU-only / no NVIDIA;
- unsupported process-VRAM reporting;
- exclusion of pre-existing desktop/background VRAM from benchmark attribution.

The previous implementation only sampled whole-GPU `memory.used`. That was insufficient for
the original process-VRAM requirement. The rebaseline implementation adds explicit process
attribution.

## Interface

```python
from lowvram.collectors import VramMonitor

monitor = VramMonitor(pid)
monitor.start()

# benchmark process runs here

result = monitor.stop()
```

The monitor tracks the root PID plus recursively discovered child PIDs while they remain
alive.

## Sampling queries

Each measurable sample performs a system-GPU query:

```text
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,memory.used \
  --format=csv,noheader,nounits
```

and a process query:

```text
nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
  --format=csv,noheader,nounits
```

GPU UUID maps process rows back to stable GPU indexes. VRAM rows for unrelated PIDs are not
counted as target-process VRAM.

The default interval is 100 ms.

## System VRAM versus process VRAM

The existing fields remain whole-GPU/system measurements:

- `baseline_vram_mb`
- `current_vram_mb`
- `peak_vram_mb`
- `delta_vram_mb`

The rebaseline adds target-process-tree measurements:

- `baseline_process_vram_mb`
- `current_process_vram_mb`
- `peak_process_vram_mb`
- `delta_process_vram_mb`
- `process_vram_supported`

The same distinction is retained per GPU.

System delta is useful for observing total GPU pressure, but it can include unrelated GPU
processes that change while the benchmark runs.

Process VRAM is the primary attribution signal when supported. It prevents pre-existing
desktop/browser VRAM, and later unrelated background growth, from being counted as model
process memory.

For both system and process aggregates, the total peak is the maximum simultaneous total
observed in one sample. It is not the sum of independent per-GPU peaks from different times.

## Baseline semantics

System baseline excludes VRAM that was already occupied before the benchmark from
`delta_vram_mb`.

Process baseline similarly excludes memory that the target process tree had already allocated
before monitoring began from `delta_process_vram_mb`.

## Unsupported process VRAM reporting

Some NVIDIA/platform combinations can return `N/A` for per-process used GPU memory, notably
configurations where the driver cannot report that value.

LowVRAM does not turn this into zero.

Instead:

```text
process_vram_supported = false
baseline_process_vram_mb = null
current_process_vram_mb = null
peak_process_vram_mb = null
delta_process_vram_mb = null
```

Whole-GPU system monitoring continues normally.

If process-VRAM support changes during a single monitor lifecycle, that sample is rejected
rather than mixing incompatible measurement semantics.

## Target process exit

The target PID must exist when `start()` is called.

After monitoring begins, the root process may exit. Already discovered live child PIDs remain
tracked. If the process tree no longer has GPU allocations, a successful process query
reports process usage as zero while previously observed peaks are retained.

## No NVIDIA

If the system GPU query is unavailable or no measurable NVIDIA GPUs are returned, the monitor
produces the existing valid empty/all-zero system result and marks process VRAM unsupported.

## GPU identity

One lifecycle requires a stable GPU index/name/total-VRAM set. Identity changes are rejected
instead of combining measurements from different devices.

## Boundary

P1-04 only measures VRAM. It does not invoke llama.cpp, parse runtime output, or create
Benchmark records.

After this rebaseline passes, the next active Gate is P1-05 Runtime Adapter.
