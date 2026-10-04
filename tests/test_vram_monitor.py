"""P1-04 VRAM monitor tests."""

import subprocess
import threading

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
    VramMonitorResult,
    VramSample,
)


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _sample(*usage: tuple[int, str, int, int]) -> VramSample:
    gpus = [
        VramGPUUsage(
            index=index,
            name=name,
            vram_total_mb=total,
            vram_used_mb=used,
        )
        for index, name, total, used in usage
    ]
    return VramSample(
        gpus=gpus,
        total_vram_used_mb=sum(gpu.vram_used_mb for gpu in gpus),
    )


def test_collect_vram_sample_supports_multiple_gpus(monkeypatch) -> None:
    monkeypatch.setattr(
        vram,
        "_run_nvidia_smi",
        lambda args: _completed(
            "1, NVIDIA RTX 4090, 24564, 12000, 91, 555.42\n"
            "0, NVIDIA RTX 4060, 8188, 2048, 37, 555.42\n"
        ),
    )

    sample = vram._collect_vram_sample()

    assert [gpu.index for gpu in sample.gpus] == [0, 1]
    assert [gpu.vram_used_mb for gpu in sample.gpus] == [2048, 12000]
    assert sample.total_vram_used_mb == 14048


def test_vram_monitor_tracks_baseline_current_peak_and_delta(monkeypatch) -> None:
    samples = iter(
        [
            _sample(
                (0, "GPU 0", 1000, 100),
                (1, "GPU 1", 1000, 100),
            ),
            _sample(
                (0, "GPU 0", 1000, 200),
                (1, "GPU 1", 1000, 50),
            ),
            _sample(
                (0, "GPU 0", 1000, 120),
                (1, "GPU 1", 1000, 180),
            ),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(vram, "_collect_vram_sample", lambda: next(samples))

    monitor = VramMonitor(1234)
    monitor.start()
    monitor.sample()
    monitor.sample()
    result = monitor.stop()

    assert result.baseline_vram_mb == 200
    assert result.current_vram_mb == 300
    assert result.peak_vram_mb == 300
    assert result.delta_vram_mb == 100
    assert result.sample_count == 3

    assert [gpu.peak_vram_mb for gpu in result.gpus] == [200, 180]
    assert [gpu.delta_vram_mb for gpu in result.gpus] == [100, 80]
    assert sum(gpu.peak_vram_mb for gpu in result.gpus) == 380
    assert result.peak_vram_mb == 300


def test_vram_monitor_background_loop_samples(monkeypatch) -> None:
    sampled_twice = threading.Event()
    call_count = 0

    def fake_sample() -> VramSample:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            sampled_twice.set()
        return _sample((0, "GPU 0", 1000, 100 + call_count))

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(vram, "_collect_vram_sample", fake_sample)

    monitor = VramMonitor(1234)
    monitor.start()

    assert sampled_twice.wait(timeout=0.2)
    result = monitor.stop()

    assert result.sample_count >= 2
    assert result.peak_vram_mb >= result.baseline_vram_mb


def test_vram_monitor_handles_no_nvidia_as_empty_result(monkeypatch) -> None:
    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(
        vram,
        "_collect_vram_sample",
        lambda: VramSample(gpus=[], total_vram_used_mb=0),
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
    assert result.sample_count == 1


def test_process_exit_after_start_does_not_break_vram_sampling(monkeypatch) -> None:
    process_alive = True
    samples = iter(
        [
            _sample((0, "GPU 0", 1000, 100)),
            _sample((0, "GPU 0", 1000, 250)),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: process_alive)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(vram, "_collect_vram_sample", lambda: next(samples))

    monitor = VramMonitor(1234)
    monitor.start()

    process_alive = False
    sample = monitor.sample()
    result = monitor.stop()

    assert sample.total_vram_used_mb == 250
    assert result.peak_vram_mb == 250
    assert result.delta_vram_mb == 150


def test_vram_monitor_rejects_gpu_identity_changes(monkeypatch) -> None:
    samples = iter(
        [
            _sample((0, "GPU 0", 1000, 100)),
            _sample((1, "GPU 1", 1000, 100)),
        ]
    )

    monkeypatch.setattr(vram.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(vram, "DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(vram, "_collect_vram_sample", lambda: next(samples))

    monitor = VramMonitor(1234)
    monitor.start()

    with pytest.raises(VramMonitorSampleError, match="GPU index set changed"):
        monitor.sample()

    monitor.stop()


def test_vram_monitor_uses_100ms_default_interval() -> None:
    assert vram.DEFAULT_VRAM_SAMPLE_INTERVAL_SECONDS == 0.1


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
        lambda: _sample((0, "GPU 0", 1000, 100)),
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


def test_vram_models_reject_inconsistent_values() -> None:
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

    gpu = VramGPUResult(
        index=0,
        name="GPU 0",
        vram_total_mb=1000,
        baseline_vram_mb=100,
        current_vram_mb=150,
        peak_vram_mb=200,
        delta_vram_mb=100,
    )
    with pytest.raises(ValidationError):
        VramMonitorResult(
            gpus=[gpu],
            baseline_vram_mb=100,
            current_vram_mb=150,
            peak_vram_mb=200,
            delta_vram_mb=99,
            sample_count=2,
            sample_interval_seconds=0.1,
        )
