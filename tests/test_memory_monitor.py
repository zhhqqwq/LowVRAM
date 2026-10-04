"""P1-03 RAM monitor tests."""

import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lowvram.collectors import memory
from lowvram.collectors.memory import RamMonitor, RamMonitorStateError
from lowvram.models.memory import RamMonitorResult, RamSample


class FakeProcess:
    def __init__(
        self,
        pid: int,
        rss_mb: int,
        children: list["FakeProcess"] | None = None,
    ) -> None:
        self.pid = pid
        self._rss_bytes = rss_mb * 1024 * 1024
        self._children = children or []

    def children(self, recursive: bool = False) -> list["FakeProcess"]:
        if not recursive:
            return list(self._children)

        descendants: list[FakeProcess] = []
        pending = list(self._children)
        while pending:
            child = pending.pop(0)
            descendants.append(child)
            pending.extend(child._children)
        return descendants

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self._rss_bytes)


def test_bytes_to_mb_uses_lowvram_integer_mb_unit() -> None:
    assert memory._bytes_to_mb(3 * 1024 * 1024 + 1024) == 3
    assert memory._bytes_to_mb(-1) == 0


def test_system_used_ram_uses_psutil_used_value(monkeypatch) -> None:
    monkeypatch.setattr(
        memory.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(used=1536 * 1024 * 1024),
    )

    assert memory._collect_system_used_ram_mb() == 1536


def test_process_tree_ram_includes_recursive_children(monkeypatch) -> None:
    grandchild = FakeProcess(102, 25)
    child = FakeProcess(101, 50, [grandchild])
    root = FakeProcess(100, 100, [child])
    processes = {process.pid: process for process in [root, child, grandchild]}

    def fake_process(pid: int) -> FakeProcess:
        if pid not in processes:
            raise memory.psutil.NoSuchProcess(pid)
        return processes[pid]

    monkeypatch.setattr(memory.psutil, "Process", fake_process)

    tracked = {100}
    used_mb = memory._collect_process_tree_ram_mb(100, tracked)

    assert used_mb == 175
    assert tracked == {100, 101, 102}


def test_known_child_remains_tracked_after_root_exits(monkeypatch) -> None:
    child = FakeProcess(101, 50)
    root = FakeProcess(100, 100, [child])
    processes = {100: root, 101: child}

    def fake_process(pid: int) -> FakeProcess:
        if pid not in processes:
            raise memory.psutil.NoSuchProcess(pid)
        return processes[pid]

    monkeypatch.setattr(memory.psutil, "Process", fake_process)

    tracked = {100}
    assert memory._collect_process_tree_ram_mb(100, tracked) == 150

    del processes[100]

    assert memory._collect_process_tree_ram_mb(100, tracked) == 50
    assert tracked == {101}


def test_access_denied_root_does_not_hide_tracked_child(monkeypatch) -> None:
    child = FakeProcess(101, 50)

    def fake_process(pid: int) -> FakeProcess:
        if pid == 100:
            raise memory.psutil.AccessDenied(pid)
        if pid == 101:
            return child
        raise memory.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(memory.psutil, "Process", fake_process)

    tracked = {100, 101}

    assert memory._collect_process_tree_ram_mb(100, tracked) == 50
    assert tracked == {101}


def test_zombie_tracked_process_is_skipped(monkeypatch) -> None:
    root = FakeProcess(100, 100)

    def fake_process(pid: int) -> FakeProcess:
        if pid == 100:
            return root
        if pid == 101:
            raise memory.psutil.ZombieProcess(pid)
        raise memory.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(memory.psutil, "Process", fake_process)

    tracked = {100, 101}

    assert memory._collect_process_tree_ram_mb(100, tracked) == 100
    assert tracked == {100}


def test_ram_monitor_tracks_baseline_current_peak_and_delta(monkeypatch) -> None:
    samples = iter(
        [
            RamSample(process_ram_mb=100, system_ram_mb=1000),
            RamSample(process_ram_mb=150, system_ram_mb=1200),
            RamSample(process_ram_mb=120, system_ram_mb=1100),
        ]
    )

    monkeypatch.setattr(memory.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(memory, "DEFAULT_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        memory,
        "_collect_ram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = RamMonitor(1234)
    baseline = monitor.start()
    second = monitor.sample()
    third = monitor.sample()
    result = monitor.stop()

    assert baseline.process_ram_mb == 100
    assert second.process_ram_mb == 150
    assert third.process_ram_mb == 120

    assert result.baseline_process_ram_mb == 100
    assert result.current_process_ram_mb == 120
    assert result.peak_process_ram_mb == 150
    assert result.delta_process_ram_mb == 50

    assert result.baseline_system_ram_mb == 1000
    assert result.current_system_ram_mb == 1100
    assert result.peak_system_ram_mb == 1200
    assert result.delta_system_ram_mb == 200
    assert result.sample_count == 3
    assert result.sample_interval_seconds == 60.0


def test_ram_monitor_background_loop_samples(monkeypatch) -> None:
    sampled_twice = threading.Event()
    call_count = 0

    def fake_sample(pid: int, tracked_pids: set[int]) -> RamSample:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            sampled_twice.set()
        return RamSample(process_ram_mb=100 + call_count, system_ram_mb=1000 + call_count)

    monkeypatch.setattr(memory.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(memory, "DEFAULT_SAMPLE_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(memory, "_collect_ram_sample", fake_sample)

    monitor = RamMonitor(1234)
    monitor.start()

    assert sampled_twice.wait(timeout=0.2)
    result = monitor.stop()

    assert result.sample_count >= 2
    assert result.peak_process_ram_mb >= result.baseline_process_ram_mb
    assert result.peak_system_ram_mb >= result.baseline_system_ram_mb


def test_ram_monitor_uses_100ms_default_interval() -> None:
    assert memory.DEFAULT_SAMPLE_INTERVAL_SECONDS == 0.1


@pytest.mark.parametrize("pid", [0, -1])
def test_ram_monitor_rejects_nonpositive_pid(pid: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RamMonitor(pid)


def test_ram_monitor_requires_existing_target(monkeypatch) -> None:
    monkeypatch.setattr(memory.psutil, "pid_exists", lambda pid: False)

    monitor = RamMonitor(1234)

    with pytest.raises(RamMonitorStateError, match="does not exist"):
        monitor.start()


def test_ram_monitor_rejects_invalid_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(memory.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(memory, "DEFAULT_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        memory,
        "_collect_ram_sample",
        lambda pid, tracked_pids: RamSample(process_ram_mb=10, system_ram_mb=100),
    )

    monitor = RamMonitor(1234)

    with pytest.raises(RamMonitorStateError, match="not been started"):
        monitor.sample()

    monitor.start()

    with pytest.raises(RamMonitorStateError, match="started once"):
        monitor.start()

    monitor.stop()

    with pytest.raises(RamMonitorStateError, match="already been stopped"):
        monitor.sample()

    with pytest.raises(RamMonitorStateError, match="already been stopped"):
        monitor.stop()


def test_ram_monitor_result_rejects_inconsistent_delta() -> None:
    with pytest.raises(ValidationError):
        RamMonitorResult(
            baseline_process_ram_mb=100,
            current_process_ram_mb=110,
            peak_process_ram_mb=120,
            delta_process_ram_mb=19,
            baseline_system_ram_mb=1000,
            current_system_ram_mb=1010,
            peak_system_ram_mb=1020,
            delta_system_ram_mb=20,
            sample_count=3,
            sample_interval_seconds=0.1,
        )


def test_ram_monitor_result_rejects_peak_below_current() -> None:
    with pytest.raises(ValidationError):
        RamMonitorResult(
            baseline_process_ram_mb=100,
            current_process_ram_mb=130,
            peak_process_ram_mb=120,
            delta_process_ram_mb=20,
            baseline_system_ram_mb=1000,
            current_system_ram_mb=1010,
            peak_system_ram_mb=1020,
            delta_system_ram_mb=20,
            sample_count=3,
            sample_interval_seconds=0.1,
        )
