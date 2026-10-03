# NVIDIA Collector — P1-02

P1-02 defines a one-shot NVIDIA state snapshot for later benchmark monitoring.

## Interface

```python
from lowvram.collectors import collect_nvidia_snapshot

snapshot = collect_nvidia_snapshot()
```

The return type is `NvidiaSnapshot`.

Each `NvidiaGPUStatus` contains:

- `index`
- `name`
- `vram_total_mb`
- `vram_used_mb`
- `gpu_utilization_percent`

The snapshot also records the machine-level NVIDIA `driver_version` and CUDA compatibility version reported by `nvidia-smi`.

## Collection method

The collector performs one `nvidia-smi` query for GPU state and one banner query for the CUDA compatibility version. It does not start a loop, keep history, compute peaks, or calculate deltas.

Unsupported dynamic fields such as utilization may be represented as `null`. Malformed GPU rows are ignored rather than converted into invented measurements. If `nvidia-smi` is absent, times out, or exits unsuccessfully, the collector returns an empty snapshot.

## Relationship to P1-01

The existing `collect_nvidia()` API now projects the richer snapshot back into the static P0 `GPUInfo` and `DriverInfo` contracts used by `lowvram system`. P0 schemas therefore remain unchanged.

## Boundary

Continuous sampling, baseline VRAM, peak VRAM, and VRAM delta belong to P1-04 VRAM Monitor.
