# NVIDIA Collector — P1-02

P1-02 defines a one-shot NVIDIA state snapshot for later benchmark monitoring.

## Rebaseline acceptance contract

Under the restarted P1 acceptance sequence, P1-02 must collect and expose exactly these
NVIDIA snapshot categories:

- GPU index
- GPU name
- total VRAM
- used VRAM
- GPU utilization
- NVIDIA driver version
- CUDA compatibility version

The collector remains a one-shot snapshot. Continuous sampling, baseline, peak, and delta are
not part of P1-02.

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

The snapshot also records machine-level `driver_version` and the CUDA compatibility version
reported by `nvidia-smi`.

## Collection contract

The GPU-state query is fixed to:

```text
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
  --format=csv,noheader,nounits
```

After a successful GPU query, one normal `nvidia-smi` banner query is used to read the CUDA
compatibility version.

This means P1-02 performs a bounded one-shot collection. It does not start a background loop,
retain history, compute peaks, or calculate deltas.

## Failure and malformed-data behavior

If the primary `nvidia-smi` query:

- is not installed / cannot be started;
- raises an OS error;
- times out;
- exits with a non-zero status;

the collector returns an empty `NvidiaSnapshot` and does not attempt a CUDA banner query.

Unsupported dynamic fields such as used VRAM or utilization are represented as `null` when
`nvidia-smi` explicitly reports them as unavailable.

Malformed rows and rows that violate the model contract are ignored rather than converted
into invented measurements.

Multi-GPU rows are sorted by GPU index before being returned.

## CUDA semantics

The CUDA value is the compatibility version displayed by the NVIDIA driver through
`nvidia-smi`. It is not proof that the same CUDA Toolkit version is installed on the
machine.

## Relationship to P1-01

The existing `collect_nvidia()` API projects the richer snapshot back into the static P0
`GPUInfo` and `DriverInfo` contracts used by `lowvram system`. P0 schemas therefore
remain unchanged.

## Boundary

Continuous sampling, baseline VRAM, peak VRAM, and VRAM delta belong to P1-04 VRAM Monitor.

After this rebaseline passes, the next active Gate is P1-03 RAM Monitor.
