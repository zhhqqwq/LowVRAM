"""Pydantic + JSON Schema validation helpers."""

import json
from pathlib import Path
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, ValidationError

from lowvram.models import BenchmarkRun, HardwareInfo, ModelInfo, Recipe

DataKind = Literal["benchmark", "hardware", "model", "recipe"]
MODEL_BY_KIND: dict[DataKind, type[BaseModel]] = {
    "benchmark": BenchmarkRun,
    "hardware": HardwareInfo,
    "model": ModelInfo,
    "recipe": Recipe,
}


class DataValidationError(ValueError):
    """Raised when a data file violates either validation layer."""


def infer_kind(data: dict[str, Any]) -> DataKind:
    """Infer the LowVRAM document kind from required structural keys."""
    if {"run_id", "hardware", "model", "result"}.issubset(data):
        return "benchmark"
    if {"cpu", "gpu", "ram_total_mb", "os"}.issubset(data):
        return "hardware"
    if {"id", "architecture", "parameter_count", "format", "quantization"}.issubset(data):
        return "model"
    if {"model_id", "runtime", "context_length", "gpu_layers"}.issubset(data):
        return "recipe"
    raise DataValidationError("unable to infer document type from top-level keys")


def _schema_path(kind: DataKind) -> Path:
    return Path(__file__).resolve().parents[2] / "schema" / f"{kind}.schema.json"


def _format_pydantic_error(error: ValidationError) -> str:
    lines: list[str] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"{location}: {item['msg']}")
    return "\n".join(lines)


def _format_jsonschema_error(error: JSONSchemaValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{location}: {error.message}"


def validate_document(data: dict[str, Any], kind: DataKind | None = None) -> DataKind:
    """Validate a parsed LowVRAM document against Pydantic and Draft 2020-12 schema."""
    resolved_kind = kind or infer_kind(data)
    model = MODEL_BY_KIND[resolved_kind]
    try:
        model.model_validate(data)
    except ValidationError as exc:
        raise DataValidationError(_format_pydantic_error(exc)) from exc

    schema_path = _schema_path(resolved_kind)
    if schema_path.exists():
        schema = cast(
            dict[str, Any],
            json.loads(schema_path.read_text(encoding="utf-8")),
        )
    else:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **model.model_json_schema(mode="validation"),
        }
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if errors:
        raise DataValidationError("\n".join(_format_jsonschema_error(error) for error in errors))
    return resolved_kind


def validate_file(path: Path, kind: DataKind | None = None) -> DataKind:
    """Load and validate a JSON file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise DataValidationError("top-level JSON value must be an object")
    return validate_document(raw, kind)
