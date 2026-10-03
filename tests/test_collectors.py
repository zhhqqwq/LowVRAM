"""P1 collector tests."""

import subprocess

import pytest
from pydantic import ValidationError

from lowvram.collectors import nvidia
from lowvram.collectors.system import collect_system
from lowvram.models.nvidia import NvidiaGPUStatus


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def test_nvidia_snapshot_collects_requested_fields(monkeypatch) -> None:
    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args:
            return _completed("0, NVIDIA GeForce RTX 4060, 8188, 2048, 37, 555.42\n")
        return _completed("| NVIDIA-SMI 555.42   CUDA Version: 12.5 |\n")

    monkeypatch.setattr(nvidia, "_run_nvidia_smi", fake_run)

    snapshot = nvidia.collect_nvidia_snapshot()

    assert snapshot.driver_version == "555.42"
    assert snapshot.cuda_version == "12.5"
    assert len(snapshot.gpus) == 1
    gpu = snapshot.gpus[0]
    assert gpu.index == 0
    assert gpu.name == "NVIDIA GeForce RTX 4060"
    assert gpu.vram_total_mb == 8188
    assert gpu.vram_used_mb == 2048
    assert gpu.gpu_utilization_percent == 37.0


def test_nvidia_snapshot_supports_multiple_gpus_and_sorts_by_index(monkeypatch) -> None:
    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args:
            return _completed(
                "1, NVIDIA GeForce RTX 4090, 24564, 12000, 91, 555.42\n"
                "0, NVIDIA GeForce RTX 4060, 8188, 1024, 12, 555.42\n"
            )
        return _completed("CUDA Version: 12.5\n")

    monkeypatch.setattr(nvidia, "_run_nvidia_smi", fake_run)

    snapshot = nvidia.collect_nvidia_snapshot()

    assert [gpu.index for gpu in snapshot.gpus] == [0, 1]
    assert [gpu.vram_used_mb for gpu in snapshot.gpus] == [1024, 12000]
    assert [gpu.gpu_utilization_percent for gpu in snapshot.gpus] == [12.0, 91.0]


def test_nvidia_snapshot_treats_missing_nvidia_smi_as_empty(monkeypatch) -> None:
    monkeypatch.setattr(nvidia, "_run_nvidia_smi", lambda args: None)

    snapshot = nvidia.collect_nvidia_snapshot()

    assert snapshot.gpus == []
    assert snapshot.driver_version is None
    assert snapshot.cuda_version is None


def test_run_nvidia_smi_nonzero_exit_is_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed("", returncode=1),
    )

    assert nvidia._run_nvidia_smi([]) is None


def test_nvidia_snapshot_preserves_unsupported_dynamic_metrics(monkeypatch) -> None:
    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args:
            return _completed(
                "bad row\n"
                "0, NVIDIA GPU, 8192, N/A, [Not Supported], 555.42\n"
                "1, Broken GPU, 8192, 9000, 10, 555.42\n"
            )
        return _completed("CUDA Version: 12.5\n")

    monkeypatch.setattr(nvidia, "_run_nvidia_smi", fake_run)

    snapshot = nvidia.collect_nvidia_snapshot()

    assert len(snapshot.gpus) == 1
    gpu = snapshot.gpus[0]
    assert gpu.index == 0
    assert gpu.vram_used_mb is None
    assert gpu.gpu_utilization_percent is None


def test_nvidia_gpu_status_rejects_used_vram_above_total() -> None:
    with pytest.raises(ValidationError):
        NvidiaGPUStatus(
            index=0,
            name="NVIDIA GPU",
            vram_total_mb=8192,
            vram_used_mb=8193,
            gpu_utilization_percent=50,
        )


def test_static_nvidia_collector_projects_snapshot(monkeypatch) -> None:
    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args:
            return _completed("0, NVIDIA GeForce RTX 4060, 8188, 2048, 37, 555.42\n")
        return _completed("CUDA Version: 12.5\n")

    monkeypatch.setattr(nvidia, "_run_nvidia_smi", fake_run)

    gpus, driver = nvidia.collect_nvidia()

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 4060"
    assert gpus[0].vram_total_mb == 8188
    assert gpus[0].driver_version == "555.42"
    assert gpus[0].cuda_version == "12.5"
    assert driver.nvidia_driver_version == "555.42"
    assert driver.cuda_version == "12.5"


def test_system_collector_returns_valid_hardware_snapshot() -> None:
    hardware = collect_system()

    assert hardware.cpu.name
    assert hardware.ram_total_mb > 0
    assert hardware.os.name
    assert hardware.python_version
    assert isinstance(hardware.gpu, list)
