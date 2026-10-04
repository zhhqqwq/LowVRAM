"""P0 data model tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lowvram.models import BenchmarkRun, HardwareInfo, ModelInfo

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(group: str, name: str) -> dict[str, object]:
    """Load one JSON fixture."""
    return json.loads((FIXTURES / group / name).read_text(encoding="utf-8"))


def test_cpu_only_hardware_is_valid() -> None:
    data = load_fixture("valid", "02_cpu_only_failure.json")["hardware"]
    hardware = HardwareInfo.model_validate(data)
    assert hardware.gpu == []


def test_single_gpu_hardware_is_valid() -> None:
    data = load_fixture("valid", "01_dense_single_gpu_success.json")["hardware"]
    hardware = HardwareInfo.model_validate(data)
    assert len(hardware.gpu) == 1


def test_multi_gpu_hardware_is_valid() -> None:
    data = load_fixture("valid", "03_multi_gpu_success.json")["hardware"]
    hardware = HardwareInfo.model_validate(data)
    assert len(hardware.gpu) == 2


def test_moe_tracks_total_and_active_parameters() -> None:
    data = load_fixture("valid", "04_moe_success.json")["model"]
    model = ModelInfo.model_validate(data)
    assert model.parameter_count == 46_000_000_000
    assert model.active_parameter_count == 12_000_000_000


def test_moe_requires_active_parameter_count() -> None:
    data = load_fixture("invalid", "03_moe_missing_active_parameters.json")["model"]
    with pytest.raises(ValidationError):
        ModelInfo.model_validate(data)


def test_failed_benchmark_is_valid() -> None:
    data = load_fixture("valid", "05_oom_failure.json")
    benchmark = BenchmarkRun.model_validate(data)
    assert benchmark.result.success is False
    assert benchmark.result.error_type is not None


def test_negative_memory_is_invalid() -> None:
    data = load_fixture("invalid", "01_negative_vram.json")
    with pytest.raises(ValidationError):
        BenchmarkRun.model_validate(data)


def test_missing_required_field_is_invalid() -> None:
    data = load_fixture("invalid", "02_missing_model_name.json")
    with pytest.raises(ValidationError):
        BenchmarkRun.model_validate(data)


def test_unknown_critical_field_is_not_ignored() -> None:
    data = load_fixture("valid", "01_dense_single_gpu_success.json")
    data["mystery_critical_field"] = "must fail"
    with pytest.raises(ValidationError):
        BenchmarkRun.model_validate(data)


def test_success_benchmark_allows_unreported_load_time() -> None:
    data = load_fixture("valid", "01_dense_single_gpu_success.json")
    data["performance"]["load_time_seconds"] = None

    benchmark = BenchmarkRun.model_validate(data)

    assert benchmark.result.success is True
    assert benchmark.performance.load_time_seconds is None


def test_success_benchmark_still_requires_throughput_metrics() -> None:
    data = load_fixture("valid", "01_dense_single_gpu_success.json")
    data["performance"]["prompt_tokens_per_second"] = None

    with pytest.raises(ValidationError, match="prompt_tokens_per_second"):
        BenchmarkRun.model_validate(data)
