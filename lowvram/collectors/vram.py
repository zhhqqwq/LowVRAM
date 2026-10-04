"""P1-04 NVIDIA system/process VRAM monitoring."""

import threading
from dataclasses import dataclass

import psutil

from lowvram.collectors.nvidia import _run_nvidia_smi
from lowvram.models.vram import (
    VramGPUResult,
    VramGPUUsage,
    VramMonitorResult,
    VramSample,
)

DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS = 0.1

_SYSTEM_VRAM_QUERY_ARGS = [
    "--query-gpu=index,uuid,name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]
_PROCESS_VRAM_QUERY_ARGS = [
    "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
    "--format=csv,noheader,nounits",
]
_UNAVAILABLE_VALUES = {"", "n/a", "[n/a]", "[not supported]", "not supported"}
_PROCESS_GONE_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


@dataclass(frozen=True)
class _SystemGPUState:
    index: int
    uuid: str
    name: str
    vram_total_mb: int
    vram_used_mb: int


class VramMonitorStateError(RuntimeError):
    """Raised when the VRAM monitor lifecycle is used in an invalid state."""


class VramMonitorSampleError(RuntimeError):
    """Raised when a sample cannot preserve monitor measurement semantics."""


def _parse_nonnegative_int(value: str) -> int:
    """Parse one non-negative integer-valued nvidia-smi field."""
    parsed = float(value.strip())
    if parsed < 0 or not parsed.is_integer():
        raise ValueError("expected a non-negative integer")
    return int(parsed)


def _parse_system_gpu_rows(output: str) -> list[_SystemGPUState]:
    """Parse the per-GPU system-memory query used by P1-04."""
    gpus: list[_SystemGPUState] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue

        index_text, uuid, name, total_text, used_text = fields
        if not uuid or not name or used_text.lower() in _UNAVAILABLE_VALUES:
            continue

        try:
            gpu = _SystemGPUState(
                index=_parse_nonnegative_int(index_text),
                uuid=uuid,
                name=name,
                vram_total_mb=_parse_nonnegative_int(total_text),
                vram_used_mb=_parse_nonnegative_int(used_text),
            )
        except ValueError:
            continue

        if gpu.vram_used_mb > gpu.vram_total_mb:
            continue
        gpus.append(gpu)

    return sorted(gpus, key=lambda gpu: gpu.index)


def _refresh_target_pids(root_pid: int, tracked_pids: set[int]) -> set[int]:
    """Refresh the target process tree while retaining already discovered live children."""
    candidate_pids = set(tracked_pids)
    candidate_pids.add(root_pid)

    try:
        root = psutil.Process(root_pid)
        candidate_pids.update(child.pid for child in root.children(recursive=True))
    except _PROCESS_GONE_ERRORS:
        pass

    alive_pids: set[int] = set()
    for pid in candidate_pids:
        try:
            psutil.Process(pid)
            alive_pids.add(pid)
        except _PROCESS_GONE_ERRORS:
            continue

    tracked_pids.clear()
    tracked_pids.update(alive_pids)
    return alive_pids


def _parse_process_vram_rows(
    output: str,
    *,
    target_pids: set[int],
    gpu_uuid_to_index: dict[str, int],
) -> tuple[bool, dict[int, int]]:
    """Attribute compute-app VRAM rows to the monitored process tree."""
    usage_by_index = {index: 0 for index in gpu_uuid_to_index.values()}

    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue

        pid_text, gpu_uuid, used_text = fields
        try:
            pid = int(pid_text)
        except ValueError:
            continue

        if pid not in target_pids:
            continue

        gpu_index = gpu_uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            return False, {}

        if used_text.lower() in _UNAVAILABLE_VALUES:
            return False, {}

        try:
            used_mb = _parse_nonnegative_int(used_text)
        except ValueError:
            return False, {}

        usage_by_index[gpu_index] += used_mb

    return True, usage_by_index


def _collect_vram_sample(root_pid: int, tracked_pids: set[int]) -> VramSample:
    """Capture one system VRAM sample plus target-process attribution when supported."""
    target_pids = _refresh_target_pids(root_pid, tracked_pids)

    system_query = _run_nvidia_smi(_SYSTEM_VRAM_QUERY_ARGS)
    if system_query is None:
        return VramSample(gpus=[], total_vram_used_mb=0)

    system_gpus = _parse_system_gpu_rows(system_query.stdout)
    if not system_gpus:
        return VramSample(gpus=[], total_vram_used_mb=0)

    process_query = _run_nvidia_smi(_PROCESS_VRAM_QUERY_ARGS)
    process_supported = process_query is not None
    process_usage: dict[int, int] = {}

    if process_query is not None:
        process_supported, process_usage = _parse_process_vram_rows(
            process_query.stdout,
            target_pids=target_pids,
            gpu_uuid_to_index={gpu.uuid: gpu.index for gpu in system_gpus},
        )

    gpus = [
        VramGPUUsage(
            index=gpu.index,
            name=gpu.name,
            vram_total_mb=gpu.vram_total_mb,
            vram_used_mb=gpu.vram_used_mb,
            process_vram_used_mb=(
                process_usage[gpu.index] if process_supported else None
            ),
        )
        for gpu in system_gpus
    ]

    return VramSample(
        gpus=gpus,
        total_vram_used_mb=sum(gpu.vram_used_mb for gpu in gpus),
        process_vram_supported=process_supported,
        total_process_vram_used_mb=(
            sum(process_usage.values()) if process_supported else None
        ),
    )


class VramMonitor:
    """Monitor NVIDIA system and target-process VRAM at the P1-04 100 ms cadence."""

    def __init__(self, pid: int) -> None:
        if pid <= 0:
            raise ValueError("pid must be a positive integer")

        self._pid = pid
        self._tracked_pids: set[int] = {pid}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._started = False
        self._stopped = False
        self._baseline: VramSample | None = None
        self._current: VramSample | None = None

        self._peak_vram_mb = 0
        self._peak_by_index: dict[int, int] = {}

        self._peak_process_vram_mb: int | None = None
        self._peak_process_by_index: dict[int, int] = {}

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

            initial = _collect_vram_sample(self._pid, self._tracked_pids)
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

            sample = _collect_vram_sample(self._pid, self._tracked_pids)
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

            process_supported = baseline.process_vram_supported
            baseline_process = baseline.total_process_vram_used_mb
            current_process = current.total_process_vram_used_mb

            return VramMonitorResult(
                gpus=gpu_results,
                baseline_vram_mb=baseline.total_vram_used_mb,
                current_vram_mb=current.total_vram_used_mb,
                peak_vram_mb=self._peak_vram_mb,
                delta_vram_mb=self._peak_vram_mb - baseline.total_vram_used_mb,
                process_vram_supported=process_supported,
                baseline_process_vram_mb=(
                    baseline_process if process_supported else None
                ),
                current_process_vram_mb=(
                    current_process if process_supported else None
                ),
                peak_process_vram_mb=(
                    self._peak_process_vram_mb if process_supported else None
                ),
                delta_process_vram_mb=(
                    self._process_delta(baseline) if process_supported else None
                ),
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
                gpu.index: gpu.vram_used_mb for gpu in sample.gpus
            }
            if sample.process_vram_supported:
                process_total = self._require_process_total(sample)
                self._peak_process_vram_mb = process_total
                self._peak_process_by_index = {
                    gpu.index: self._require_process_gpu_usage(gpu)
                    for gpu in sample.gpus
                }
        else:
            self._validate_gpu_identity(self._baseline, sample)
            self._validate_process_support(self._baseline, sample)

        self._current = sample
        self._peak_vram_mb = max(self._peak_vram_mb, sample.total_vram_used_mb)

        for gpu in sample.gpus:
            self._peak_by_index[gpu.index] = max(
                self._peak_by_index[gpu.index],
                gpu.vram_used_mb,
            )

        if sample.process_vram_supported:
            process_total = self._require_process_total(sample)
            current_peak = self._peak_process_vram_mb
            if current_peak is None:
                raise VramMonitorSampleError("process VRAM peak state is unavailable")
            self._peak_process_vram_mb = max(current_peak, process_total)

            for gpu in sample.gpus:
                process_used = self._require_process_gpu_usage(gpu)
                self._peak_process_by_index[gpu.index] = max(
                    self._peak_process_by_index[gpu.index],
                    process_used,
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

    @staticmethod
    def _validate_process_support(baseline: VramSample, sample: VramSample) -> None:
        """Keep process-attribution capability stable for one monitor lifecycle."""
        if sample.process_vram_supported != baseline.process_vram_supported:
            raise VramMonitorSampleError(
                "process VRAM support changed during VRAM monitoring"
            )

    def _build_gpu_results(
        self,
        baseline: VramSample,
        current: VramSample,
    ) -> list[VramGPUResult]:
        """Build per-GPU aggregates without summing independent peaks across time."""
        current_by_index = {gpu.index: gpu for gpu in current.gpus}
        results: list[VramGPUResult] = []

        for baseline_gpu in baseline.gpus:
            current_gpu = current_by_index[baseline_gpu.index]
            peak = self._peak_by_index[baseline_gpu.index]
            process_supported = baseline.process_vram_supported

            baseline_process = baseline_gpu.process_vram_used_mb
            current_process = current_gpu.process_vram_used_mb
            peak_process = (
                self._peak_process_by_index[baseline_gpu.index]
                if process_supported
                else None
            )

            results.append(
                VramGPUResult(
                    index=baseline_gpu.index,
                    name=baseline_gpu.name,
                    vram_total_mb=baseline_gpu.vram_total_mb,
                    baseline_vram_mb=baseline_gpu.vram_used_mb,
                    current_vram_mb=current_gpu.vram_used_mb,
                    peak_vram_mb=peak,
                    delta_vram_mb=peak - baseline_gpu.vram_used_mb,
                    process_vram_supported=process_supported,
                    baseline_process_vram_mb=(
                        baseline_process if process_supported else None
                    ),
                    current_process_vram_mb=(
                        current_process if process_supported else None
                    ),
                    peak_process_vram_mb=peak_process,
                    delta_process_vram_mb=(
                        self._gpu_process_delta(baseline_gpu, peak_process)
                        if process_supported
                        else None
                    ),
                )
            )

        return sorted(results, key=lambda gpu: gpu.index)

    def _process_delta(self, baseline: VramSample) -> int:
        """Return aggregate process peak minus aggregate process baseline."""
        baseline_process = self._require_process_total(baseline)
        peak_process = self._peak_process_vram_mb
        if peak_process is None:
            raise VramMonitorSampleError("process VRAM peak state is unavailable")
        return peak_process - baseline_process

    @staticmethod
    def _gpu_process_delta(
        baseline_gpu: VramGPUUsage,
        peak_process: int | None,
    ) -> int:
        """Return one GPU's process peak minus its process baseline."""
        baseline_process = baseline_gpu.process_vram_used_mb
        if baseline_process is None or peak_process is None:
            raise VramMonitorSampleError("per-GPU process VRAM state is unavailable")
        return peak_process - baseline_process

    @staticmethod
    def _require_process_total(sample: VramSample) -> int:
        value = sample.total_process_vram_used_mb
        if value is None:
            raise VramMonitorSampleError("process VRAM total is unavailable")
        return value

    @staticmethod
    def _require_process_gpu_usage(gpu: VramGPUUsage) -> int:
        value = gpu.process_vram_used_mb
        if value is None:
            raise VramMonitorSampleError("per-GPU process VRAM usage is unavailable")
        return value

    @staticmethod
    def _require_sample(sample: VramSample | None) -> VramSample:
        if sample is None:
            raise VramMonitorStateError("VRAM monitor has no samples")
        return sample
