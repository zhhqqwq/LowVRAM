"""P1-04 VRAM monitor tests."""

import subprocess
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lowvram.collectors import vram
from lowvram.collectors.vram import (
    VramMonitor,
    VramMonitorSampleError,
    VramMonitorStateError,
)
from lowvram.models.vram import (
    VramGPUResult,
    VramGPUUsage,
    VramSample,
)


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _sample(
    *usage: tuple[int, str, int, int, int | None],
    process_supported: bool = True,
) -> VramSample:
    gpus = [
        VramGPUUsage(
            index=index,
            name=name,
            vram_total_mb=total,
            vram_used_mb=system_used,
            process_vram_used_mb=process_used if process_supported else None,
        )
        for index, name, total, system_used, process_used in usage
    ]
    return VramSample(
        gpus=gpus,
        total_vram_used_mb=sum(gpu.vram_used_mb for gpu in gpus),
        process_vram_supported=process_supported,
        total_process_vram_used_mb=(
            sum(gpu.process_vram_used_mb or 0 for gpu in gpus)
            if process_supported
            else None
        ),
    )


def test_collect_vram_sample_attributes_target_process_tree_by_gpu_uuid(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        vram,
        "_refresh_target_pids",
        lambda root_pid, tracked_pids: {1234, 1235},
    )

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        calls.append(list(args))
        if args == vram._SYSTEM_VRAM_QUERY_ARGS:
            return _completed(
                "1, GPU-b, NVIDIA RTX 4090, 24564, 12000\n"
                "0, GPU-a, NVIDIA RTX 4060, 8188, 2048\n"
            )
        if args == vram._PROCESS_VRAM_QUERY_ARGS:
            return _completed(
                "1234, GPU-a, 500\n"
                "1235, GPU-a, 250\n"
                "9999, GPU-b, 1000\n"
            )
        pytest.fail(f"unexpected nvidia-smi args: {args}")

    monkeypatch.setattr(vram, "_run_nvidia_smi", fake_run)

    sample = vram._collect_vram_sample(1234, {1234})

    assert [gpu.index for gpu in sample.gpus] == [0, 1]
    assert [gpu.vram_used_mb for gpu in sample.gpus] == [2048, 12000]
    assert [gpu.process_vram_used_mb for gpu in sample.gpus] == [750, 0]
    assert sample.total_vram_used_mb == 14048
    assert sample.process_vram_supported is True
    assert sample.total_process_vram_used_mb == 750
    assert calls == [
        vram._SYSTEM_VRAM_QUERY_ARGS,
        vram._PROCESS_VRAM_QUERY_ARGS,
    ]


def test_process_query_failure_preserves_system_vram_and_marks_unsupported(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vram,
        "_refresh_target_pids",
        lambda root_pid, tracked_pids: {1234},
    )

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args == vram._SYSTEM_VRAM_QUERY_ARGS:
            return _completed("0, GPU-a, NVIDIA GPU, 8192, 2048\n")
        if args == vram._PROCESS_VRAM_QUERY_ARGS:
            return None
        pytest.fail(f"unexpected nvidia-smi args: {args}")

    monkeypatch.setattr(vram, "_run_nvidia_smi", fake_run)

    sample = vram._collect_vram_sample(1234, {1234})

    assert sample.total_vram_used_mb == 2048
    assert sample.process_vram_supported is False
    assert sample.total_process_vram_used_mb is None
    assert sample.gpus[0].process_vram_used_mb is None


def test_process_n_a_marks_process_attribution_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        vram,
        "_refresh_target_pids",
        lambda root_pid, tracked_pids: {1234},
    )

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args == vram._SYSTEM_VRAM_QUERY_ARGS:
            return _completed("0, GPU-a, NVIDIA GPU, 8192, 2048\n")
        return _completed("1234, GPU-a, N/A\n")

    monkeypatch.setattr(vram, "_run_nvidia_smi", fake_run)

    sample = vram._collect_vram_sample(1234, {1234})

    assert sample.process_vram_supported is False
    assert sample.total_process_vram_used_mb is None
    assert sample.gpus[0].process_vram_used_mb is None


def test_target_pid_tracking_keeps_recursive_child_after_root_exit(monkeypatch) -> None:
    child = SimpleNamespace(pid=1235)
    root = SimpleNamespace(pid=1234, children=lambda recursive: [child])
    live = {1234: root, 1235: child}

    def fake_process(pid: int) -> object:
        process = live.get(pid)
        if process is None:
            raise vram.psutil.NoSuchProcess(pid)
        return process

    monkeypatch.setattr(vram.psutil, "Process", fake_process)

    tracked = {1234}
    assert vram._refresh_target_pids(1234, tracked) == {1234, 1235}

    del live[1234]

    assert vram._refresh_target_pids(1234, tracked) == {1235}
    assert tracked == {1235}


def test_vram_monitor_tracks_system_and_process_peaks_without_background_attribution(
    monkeypatch,
) -> None:
    samples = iter(
        [
            _sample(
                (0, "GPU 0", 10000, 3000, 100),
                (1, "GPU 1", 10000, 2000, 0),
            ),
            _sample(
                (0, "GPU 0", 10000, 5000, 900),
                (1, "GPU 1", 10000, 2500, 100),
            ),
            _sample(
                (0, "GPU 0", 10000, 7000, 1100),
                (1, "GPU 1", 10000, 3500, 200),
            ),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = VramMonitor(1234)
    monitor.start()
    monitor.sample()
    monitor.sample()
    result = monitor.stop()

    assert result.baseline_vram_mb == 5000
    assert result.current_vram_mb == 10500
    assert result.peak_vram_mb == 10500
    assert result.delta_vram_mb == 5500

    assert result.process_vram_supported is True
    assert result.baseline_process_vram_mb == 100
    assert result.current_process_vram_mb == 1300
    assert result.peak_process_vram_mb == 1300
    assert result.delta_process_vram_mb == 1200

    assert result.delta_vram_mb != result.delta_process_vram_mb
    assert [gpu.peak_process_vram_mb for gpu in result.gpus] == [1100, 200]
    assert [gpu.delta_process_vram_mb for gpu in result.gpus] == [1000, 200]


def test_aggregate_peaks_are_simultaneous_not_sum_of_per_gpu_peaks(monkeypatch) -> None:
    samples = iter(
        [
            _sample(
                (0, "GPU 0", 1000, 100, 10),
                (1, "GPU 1", 1000, 100, 10),
            ),
            _sample(
                (0, "GPU 0", 1000, 200, 100),
                (1, "GPU 1", 1000, 50, 20),
            ),
            _sample(
                (0, "GPU 0", 1000, 120, 30),
                (1, "GPU 1", 1000, 180, 90),
            ),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = VramMonitor(1234)
    monitor.start()
    monitor.sample()
    monitor.sample()
    result = monitor.stop()

    assert result.peak_vram_mb == 300
    assert sum(gpu.peak_vram_mb for gpu in result.gpus) == 380

    assert result.peak_process_vram_mb == 120
    assert sum((gpu.peak_process_vram_mb or 0) for gpu in result.gpus) == 190


def test_vram_monitor_background_loop_samples(monkeypatch) -> None:
    sampled_twice = threading.Event()
    call_count = 0

    def fake_sample(pid: int, tracked_pids: set[int]) -> VramSample:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            sampled_twice.set()
        return _sample((0, "GPU 0", 1000, 100 + call_count, 10 + call_count))

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(vram, "_collect_vram_sample", fake_sample)

    monitor = VramMonitor(1234)
    monitor.start()

    assert sampled_twice.wait(timeout=0.2)
    result = monitor.stop()

    assert result.sample_count >= 2
    assert result.peak_vram_mb >= result.baseline_vram_mb
    assert result.peak_process_vram_mb is not None


def test_vram_monitor_handles_no_nvidia_as_empty_result(monkeypatch) -> None:
    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: VramSample(gpus=[], total_vram_used_mb=0),
    )

    monitor = VramMonitor(1234)
    baseline = monitor.start()
    result = monitor.stop()

    assert baseline.gpus == []
    assert result.gpus == []
    assert result.baseline_vram_mb == 0
    assert result.current_vram_mb == 0
    assert result.peak_vram_mb == 0
    assert result.delta_vram_mb == 0
    assert result.process_vram_supported is False
    assert result.peak_process_vram_mb is None
    assert result.sample_count == 1


def test_process_exit_after_start_keeps_peak_and_allows_zero_current(monkeypatch) -> None:
    process_alive = True
    samples = iter(
        [
            _sample((0, "GPU 0", 1000, 100, 25)),
            _sample((0, "GPU 0", 1000, 250, 0)),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: process_alive)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = VramMonitor(1234)
    monitor.start()

    process_alive = False
    sample = monitor.sample()
    result = monitor.stop()

    assert sample.total_process_vram_used_mb == 0
    assert result.current_process_vram_mb == 0
    assert result.peak_process_vram_mb == 25
    assert result.delta_process_vram_mb == 0


def test_unsupported_process_vram_still_tracks_system_baseline_peak_delta(
    monkeypatch,
) -> None:
    samples = iter(
        [
            _sample((0, "GPU 0", 1000, 100, None), process_supported=False),
            _sample((0, "GPU 0", 1000, 250, None), process_supported=False),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = VramMonitor(1234)
    monitor.start()
    monitor.sample()
    result = monitor.stop()

    assert result.baseline_vram_mb == 100
    assert result.peak_vram_mb == 250
    assert result.delta_vram_mb == 150
    assert result.process_vram_supported is False
    assert result.baseline_process_vram_mb is None
    assert result.peak_process_vram_mb is None


def test_vram_monitor_rejects_process_support_changes(monkeypatch) -> None:
    samples = iter(
        [
            _sample((0, "GPU 0", 1000, 100, 10)),
            _sample((0, "GPU 0", 1000, 120, None), process_supported=False),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = VramMonitor(1234)
    monitor.start()

    with pytest.raises(VramMonitorSampleError, match="process VRAM support changed"):
        monitor.sample()

    monitor.stop()


def test_vram_monitor_rejects_gpu_identity_changes(monkeypatch) -> None:
    samples = iter(
        [
            _sample((0, "GPU 0", 1000, 100, 10)),
            _sample((1, "GPU 1", 1000, 100, 10)),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: next(samples),
    )

    monitor = VramMonitor(1234)
    monitor.start()

    with pytest.raises(VramMonitorSampleError, match="GPU index set changed"):
        monitor.sample()

    monitor.stop()


def test_vram_monitor_uses_100ms_default_interval() -> None:
    assert vram.DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS == 0.1


@pytest.mark.parametrize("pid", [0, -1])
def test_vram_monitor_rejects_nonpositive_pid(pid: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        VramMonitor(pid)


def test_vram_monitor_requires_existing_target(monkeypatch) -> None:
    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: False)

    monitor = VramMonitor(1234)

    with pytest.raises(VramMonitorStateError, match="does not exist"):
        monitor.start()


def test_vram_monitor_rejects_invalid_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda pid, tracked_pids: _sample((0, "GPU 0", 1000, 100, 10)),
    )

    monitor = VramMonitor(1234)

    with pytest.raises(VramMonitorStateError, match="not been started"):
        monitor.sample()

    monitor.start()

    with pytest.raises(VramMonitorStateError, match="started once"):
        monitor.start()

    monitor.stop()

    with pytest.raises(VramMonitorStateError, match="already been stopped"):
        monitor.sample()

    with pytest.raises(VramMonitorStateError, match="already been stopped"):
        monitor.stop()


def test_vram_models_reject_inconsistent_system_and_process_values() -> None:
    with pytest.raises(ValidationError):
        VramGPUResult(
            index=0,
            name="GPU 0",
            vram_total_mb=1000,
            baseline_vram_mb=100,
            current_vram_mb=150,
            peak_vram_mb=200,
            delta_vram_mb=99,
        )

    with pytest.raises(ValidationError):
        VramGPUResult(
            index=0,
            name="GPU 0",
            vram_total_mb=1000,
            baseline_vram_mb=100,
            current_vram_mb=150,
            peak_vram_mb=200,
            delta_vram_mb=100,
            process_vram_supported=True,
            baseline_process_vram_mb=10,
            current_process_vram_mb=20,
            peak_process_vram_mb=30,
            delta_process_vram_mb=19,
        )

    with pytest.raises(ValidationError):
        VramSample(
            gpus=[
                VramGPUUsage(
                    index=0,
                    name="GPU 0",
                    vram_total_mb=1000,
                    vram_used_mb=100,
                    process_vram_used_mb=10,
                )
            ],
            total_vram_used_mb=100,
            process_vram_supported=False,
        )
