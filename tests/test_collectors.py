"""P1-01 system collector tests."""

import subprocess

from lowvram.collectors import nvidia
from lowvram.collectors.system import collect_system


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def test_nvidia_collector_supports_multiple_gpus(monkeypatch) -> None:
    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args:
            return _completed(
                "NVIDIA GeForce RTX 4060, 8188, 555.42\n"
                "NVIDIA GeForce RTX 4090, 24564, 555.42\n"
            )
        return _completed("| NVIDIA-SMI 555.42   CUDA Version: 12.5 |\n")

    monkeypatch.setattr(nvidia, "_run_nvidia_smi", fake_run)

    gpus, driver = nvidia.collect_nvidia()

    assert [gpu.name for gpu in gpus] == [
        "NVIDIA GeForce RTX 4060",
        "NVIDIA GeForce RTX 4090",
    ]
    assert [gpu.vram_total_mb for gpu in gpus] == [8188, 24564]
    assert driver.nvidia_driver_version == "555.42"
    assert driver.cuda_version == "12.5"
    assert all(gpu.cuda_version == "12.5" for gpu in gpus)


def test_nvidia_collector_treats_missing_nvidia_smi_as_cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(nvidia, "_run_nvidia_smi", lambda args: None)

    gpus, driver = nvidia.collect_nvidia()

    assert gpus == []
    assert driver.nvidia_driver_version is None
    assert driver.cuda_version is None


def test_nvidia_collector_skips_malformed_rows(monkeypatch) -> None:
    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if args:
            return _completed("bad row\nNVIDIA GPU, not-a-number, 555.42\n")
        return _completed("CUDA Version: 12.5\n")

    monkeypatch.setattr(nvidia, "_run_nvidia_smi", fake_run)

    gpus, driver = nvidia.collect_nvidia()

    assert gpus == []
    assert driver.nvidia_driver_version is None
    assert driver.cuda_version == "12.5"


def test_system_collector_returns_valid_hardware_snapshot() -> None:
    hardware = collect_system()

    assert hardware.cpu.name
    assert hardware.ram_total_mb > 0
    assert hardware.os.name
    assert hardware.python_version
    assert isinstance(hardware.gpu, list)
