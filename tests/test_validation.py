"""Validator and checked-in fixture tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from lowvram.models import BenchmarkRun, HardwareInfo, ModelInfo, Recipe
from lowvram.validators import DataValidationError, validate_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.mark.parametrize("path", sorted((FIXTURES / "valid").glob("*.json")))
def test_valid_benchmark_fixtures(path: Path) -> None:
    assert validate_file(path, "benchmark") == "benchmark"


@pytest.mark.parametrize("path", sorted((FIXTURES / "invalid").glob("*.json")))
def test_invalid_benchmark_fixtures(path: Path) -> None:
    with pytest.raises(DataValidationError):
        validate_file(path, "benchmark")


def test_fixture_minimums() -> None:
    assert len(list((FIXTURES / "valid").glob("*.json"))) >= 5
    assert len(list((FIXTURES / "invalid").glob("*.json"))) >= 5


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("benchmark", BenchmarkRun),
        ("hardware", HardwareInfo),
        ("model", ModelInfo),
        ("recipe", Recipe),
    ],
)
def test_checked_in_schemas_are_draft_2020_12(
    name: str, model: type[BenchmarkRun] | type[HardwareInfo] | type[ModelInfo] | type[Recipe]
) -> None:
    schema_path = ROOT / "schema" / f"{name}.schema.json"
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)
    expected = {"$schema": schema["$schema"], **model.model_json_schema(mode="validation")}
    assert schema == expected


def test_all_document_kinds_validate() -> None:
    benchmark_path = FIXTURES / "valid" / "01_dense_single_gpu_success.json"
    benchmark: dict[str, Any] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    documents = {
        "hardware": benchmark["hardware"],
        "model": benchmark["model"],
        "recipe": benchmark["configuration"],
    }
    for kind, data in documents.items():
        path = ROOT / f".tmp-{kind}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            assert validate_file(path) == kind
        finally:
            path.unlink(missing_ok=True)
