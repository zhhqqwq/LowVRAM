"""P1-04 NVIDIA VRAM monitoring."""

import threading

import psutil

from lowvram.collectors.nvidia import _QUERY_ARGS, _parse_gpu_rows, _run_nvidia_smi
from lowvram.models.vram import (
    VramGPUResult,
    VramGPUUsage,
    VramMonitorResult,
    VramSample,
)

DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS = 0.1


class VramMonitorStateError(RuntimeError):
    """Raised when the VRAM monitor lifecycle is used in an invalid state."""


class VramMonitorSampleError(RuntimeError):
    """Raised when a sample cannot preserve the monitor's GPU identity set."""


def _collect_vram_sample() -> VramSample:
    """Capture one measurable multi-GPU VRAM sample with one nvidia-smi query."""
    query = _run_nvidia_smi(_QUERY_ARGS)
    if query is None:
        return VramSample(gpus=[], total_vram_used_mb=0)

    statuses, _ = _parse_gpu_rows(query.stdout)
    gpus = [
        VramGPUUsage(
            index=gpu.index,
            name=gpu.name,
            vram_total_mb=gpu.vram_total_mb,
            vram_used_mb=gpu.vram_used_mb,
        )
        for gpu in statuses
        if gpu.vram_used_mb is not None
    ]
    return VramSample(
        gpus=gpus,
        total_vram_used_mb=sum(gpu.vram_used_mb for gpu in gpus),
    )


class VramMonitor:
    """Monitor total NVIDIA VRAM usage at the P1-04 100 ms cadence."""

    def __init__(self, pid: int) -> None:
        if pid <= 0:
            raise ValueError("pid must be a positive integer")

        self._pid = pid
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._started = False
        self._stopped = False
        self._baseline: VramSample | None = None
        self._current: VramSample | None = None
        self._peak_vram_mb = 0
        self._peak_by_index: dict[int, int] = {}
        self._sample_count = 0

    @property
    def pid(self) -> int:
        """Return the benchmark process ID associated with this monitor."""
        return self._pid

    def start(self) -> VramSample:
        """Capture baseline VRAM and start background sampling when GPUs are measurable."""
        with self._lock:
            if self._started:
                raise VramMonitorStateError("VRAM monitor can only be started once")
            if not psutil.pid_exists(self._pid):
                raise VramMonitorStateError(f"target process does not exist: {self._pid}")

            initial = _collect_vram_sample()
            self._record_sample(initial)
            self._started = True

        if initial.gpus:
            self._thread = threading.Thread(
                target=self._run,
                name=f"lowvram-vram-monitor-{self._pid}",
                daemon=True,
            )
            self._thread.start()
        return initial

    def sample(self) -> VramSample:
        """Capture an immediate sample and update current/peak state."""
        with self._lock:
            if not self._started:
                raise VramMonitorStateError("VRAM monitor has not been started")
            if self._stopped:
                raise VramMonitorStateError("VRAM monitor has already been stopped")

            sample = _collect_vram_sample()
            self._record_sample(sample)
            return sample

    def stop(self) -> VramMonitorResult:
        """Stop background sampling and return the aggregated VRAM result."""
        with self._lock:
            if not self._started:
                raise VramMonitorStateError("VRAM monitor has not been started")
            if self._stopped:
                raise VramMonitorStateError("VRAM monitor has already been stopped")
            self._stopped = True

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        with self._lock:
            baseline = self._require_sample(self._baseline)
            current = self._require_sample(self._current)
            gpu_results = self._build_gpu_results(baseline, current)
            return VramMonitorResult(
                gpus=gpu_results,
                baseline_vram_mb=baseline.total_vram_used_mb,
                current_vram_mb=current.total_vram_used_mb,
                peak_vram_mb=self._peak_vram_mb,
                delta_vram_mb=self._peak_vram_mb - baseline.total_vram_used_mb,
                sample_count=self._sample_count,
                sample_interval_seconds=DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS,
            )

    def _run(self) -> None:
        """Sample repeatedly until stop() signals the monitor thread."""
        while not self._stop_event.wait(DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS):
            try:
                self.sample()
            except VramMonitorSampleError:
                continue
            except VramMonitorStateError:
                return

    def _record_sample(self, sample: VramSample) -> None:
        """Update lifecycle aggregate state from one VRAM sample."""
        if self._baseline is None:
            self._baseline = sample
            self._peak_by_index = {
                gpu.index: gpu.vram_used_mb
                for gpu in sample.gpus
            }
        else:
            self._validate_gpu_identity(self._baseline, sample)

        self._current = sample
        self._peak_vram_mb = max(self._peak_vram_mb, sample.total_vram_used_mb)
        for gpu in sample.gpus:
            self._peak_by_index[gpu.index] = max(
                self._peak_by_index[gpu.index],
                gpu.vram_used_mb,
            )
        self._sample_count += 1

    @staticmethod
    def _validate_gpu_identity(baseline: VramSample, sample: VramSample) -> None:
        """Require a stable set of GPU indexes, names, and total VRAM."""
        baseline_by_index = {gpu.index: gpu for gpu in baseline.gpus}
        sample_by_index = {gpu.index: gpu for gpu in sample.gpus}
        if baseline_by_index.keys() != sample_by_index.keys():
            raise VramMonitorSampleError("GPU index set changed during VRAM monitoring")

        for index, baseline_gpu in baseline_by_index.items():
            sample_gpu = sample_by_index[index]
            if (
                sample_gpu.name != baseline_gpu.name
                or sample_gpu.vram_total_mb != baseline_gpu.vram_total_mb
            ):
                raise VramMonitorSampleError(
                    f"GPU identity changed during VRAM monitoring: index {index}"
                )

    def _build_gpu_results(
        self,
        baseline: VramSample,
        current: VramSample,
    ) -> list[VramGPUResult]:
        """Build per-GPU aggregate results without summing peaks across time."""
        current_by_index = {gpu.index: gpu for gpu in current.gpus}
        results: list[VramGPUResult] = []
        for baseline_gpu in baseline.gpus:
            current_gpu = current_by_index[baseline_gpu.index]
            peak = self._peak_by_index[baseline_gpu.index]
            results.append(
                VramGPUResult(
                    index=baseline_gpu.index,
                    name=baseline_gpu.name,
                    vram_total_mb=baseline_gpu.vram_total_mb,
                    baseline_vram_mb=baseline_gpu.vram_used_mb,
                    current_vram_mb=current_gpu.vram_used_mb,
                    peak_vram_mb=peak,
                    delta_vram_mb=peak - baseline_gpu.vram_used_mb,
                )
            )
        return sorted(results, key=lambda gpu: gpu.index)

    @staticmethod
    def _require_sample(sample: VramSample | None) -> VramSample:
        if sample is None:
            raise VramMonitorStateError("VRAM monitor has no samples")
        return sample
