# VRAM Monitor — P1-04

P1-04 adds lifecycle monitoring for NVIDIA VRAM usage.

## Interface

```python
from lowvram.collectors import VramMonitor

monitor = VramMonitor(pid)
monitor.start()

# benchmark process runs here

result = monitor.stop()
```

`start()` validates that the associated benchmark PID exists, captures the baseline VRAM
state, and starts background sampling when NVIDIA GPUs are measurable. `sample()` performs
an immediate sample. `stop()` returns the aggregate result.

The target process may exit after monitoring starts. VRAM sampling remains system/GPU based,
so the final sample and `stop()` still work after process exit.

## Sampling

The default sampling interval is 100 ms. Each sample uses one NVIDIA GPU status query and
records each measurable GPU's index, name, total VRAM, and currently used VRAM.

No-NVIDIA or unavailable-`nvidia-smi` environments produce an empty baseline and a valid
all-zero result. This keeps CPU-only benchmark paths valid.

## Baseline and delta

VRAM already occupied before the benchmark must not be attributed to the benchmark. The
monitor therefore records:

- `baseline_vram_mb`
- `current_vram_mb`
- `peak_vram_mb`
- `delta_vram_mb = peak_vram_mb - baseline_vram_mb`

The same fields are retained per GPU.

For multiple GPUs, aggregate `peak_vram_mb` is the maximum of the simultaneous total VRAM
usage observed at any one sampling instant. It is deliberately not the sum of independent
per-GPU peaks, because those peaks may occur at different times.

## GPU identity

A monitor lifecycle expects the same GPU index/name/total-VRAM identity set as its baseline.
A sample with a changed identity set is rejected rather than silently mixing measurements
from different devices.

## Boundary

P1-04 measures GPU/system VRAM usage and baseline-adjusted deltas. It does not invoke
llama.cpp, parse runtime output, or create benchmark records. Those belong to later P1 work.
