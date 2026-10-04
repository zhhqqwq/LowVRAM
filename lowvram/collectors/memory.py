"""System-memory collection and P1 RAM monitoring helpers."""

import threading

import psutil

from lowvram.models.memory import RamMonitorResult, RamSample

DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1
_BYTES_PER_MB = 1024 * 1024
_PROCESS_GONE_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


class RamMonitorStateError(RuntimeError):
    """Raised when the RAM monitor lifecycle is used in an invalid state."""


def _bytes_to_mb(value: int) -> int:
    """Convert bytes to the integer MB unit used by LowVRAM."""
    return max(0, value // _BYTES_PER_MB)


def collect_ram_total_mb() -> int:
    """Return total physical RAM in MB."""
    total_bytes = int(psutil.virtual_memory().total)
    return max(1, _bytes_to_mb(total_bytes))


def _collect_process_tree_ram_mb(root_pid: int, tracked_pids: set[int]) -> int:
    """Sum RSS for the target process and recursively discovered child processes."""
    candidate_pids = set(tracked_pids)
    candidate_pids.add(root_pid)

    try:
        root = psutil.Process(root_pid)
        candidate_pids.update(child.pid for child in root.children(recursive=True))
    except _PROCESS_GONE_ERRORS:
        pass

    alive_pids: set[int] = set()
    rss_bytes = 0
    for pid in candidate_pids:
        try:
            process = psutil.Process(pid)
            rss_bytes += int(process.memory_info().rss)
            alive_pids.add(pid)
        except _PROCESS_GONE_ERRORS:
            continue

    tracked_pids.clear()
    tracked_pids.update(alive_pids)
    return _bytes_to_mb(rss_bytes)


def _collect_system_used_ram_mb() -> int:
    """Return current whole-system used physical RAM in MB."""
    return _bytes_to_mb(int(psutil.virtual_memory().used))


def _collect_ram_sample(root_pid: int, tracked_pids: set[int]) -> RamSample:
    """Collect one process-tree and whole-system RAM sample."""
    return RamSample(
        process_ram_mb=_collect_process_tree_ram_mb(root_pid, tracked_pids),
        system_ram_mb=_collect_system_used_ram_mb(),
    )


class RamMonitor:
    """Monitor one target process tree at the P1-03 100 ms sampling cadence."""

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
        self._baseline: RamSample | None = None
        self._current: RamSample | None = None
        self._peak_process_ram_mb = 0
        self._peak_system_ram_mb = 0
        self._sample_count = 0

    @property
    def pid(self) -> int:
        """Return the monitored root process ID."""
        return self._pid

    def start(self) -> RamSample:
        """Capture the baseline sample and start background sampling."""
        with self._lock:
            if self._started:
                raise RamMonitorStateError("RAM monitor can only be started once")
            if not psutil.pid_exists(self._pid):
                raise RamMonitorStateError(f"target process does not exist: {self._pid}")

            initial = _collect_ram_sample(self._pid, self._tracked_pids)
            self._record_sample(initial)
            self._started = True

        self._thread = threading.Thread(
            target=self._run,
            name=f"lowvram-ram-monitor-{self._pid}",
            daemon=True,
        )
        self._thread.start()
        return initial

    def sample(self) -> RamSample:
        """Capture an immediate sample and update current/peak state."""
        with self._lock:
            if not self._started:
                raise RamMonitorStateError("RAM monitor has not been started")
            if self._stopped:
                raise RamMonitorStateError("RAM monitor has already been stopped")

            sample = _collect_ram_sample(self._pid, self._tracked_pids)
            self._record_sample(sample)
            return sample

    def stop(self) -> RamMonitorResult:
        """Stop background sampling and return the aggregated result."""
        with self._lock:
            if not self._started:
                raise RamMonitorStateError("RAM monitor has not been started")
            if self._stopped:
                raise RamMonitorStateError("RAM monitor has already been stopped")
            self._stopped = True

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        with self._lock:
            baseline = self._require_sample(self._baseline)
            current = self._require_sample(self._current)
            return RamMonitorResult(
                baseline_process_ram_mb=baseline.process_ram_mb,
                current_process_ram_mb=current.process_ram_mb,
                peak_process_ram_mb=self._peak_process_ram_mb,
                delta_process_ram_mb=self._peak_process_ram_mb - baseline.process_ram_mb,
                baseline_system_ram_mb=baseline.system_ram_mb,
                current_system_ram_mb=current.system_ram_mb,
                peak_system_ram_mb=self._peak_system_ram_mb,
                delta_system_ram_mb=self._peak_system_ram_mb - baseline.system_ram_mb,
                sample_count=self._sample_count,
                sample_interval_seconds=DEFAULT_SAMPLE_INTERVAL_SECONDS,
            )

    def _run(self) -> None:
        """Sample repeatedly until stop() signals the monitor thread."""
        while not self._stop_event.wait(DEFAULT_SAMPLE_INTERVAL_SECONDS):
            try:
                self.sample()
            except RamMonitorStateError:
                return

    def _record_sample(self, sample: RamSample) -> None:
        """Update lifecycle aggregate state from one sample."""
        if self._baseline is None:
            self._baseline = sample
        self._current = sample
        self._peak_process_ram_mb = max(self._peak_process_ram_mb, sample.process_ram_mb)
        self._peak_system_ram_mb = max(self._peak_system_ram_mb, sample.system_ram_mb)
        self._sample_count += 1

    @staticmethod
    def _require_sample(sample: RamSample | None) -> RamSample:
        if sample is None:
            raise RamMonitorStateError("RAM monitor has no samples")
        return sample
