"""Export Pydantic models as checked-in JSON Schema Draft 2020-12 documents."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lowvram.models import BenchmarkRun, HardwareInfo, ModelInfo, Recipe

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
MODELS: dict[str, type[BaseModel]] = {
    "benchmark": BenchmarkRun,
    "hardware": HardwareInfo,
    "model": ModelInfo,
    "recipe": Recipe,
}


def export_schema(name: str, model: type[BaseModel]) -> None:
    """Export a single schema with an explicit Draft 2020-12 declaration."""
    schema: dict[str, Any] = model.model_json_schema(mode="validation")
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
    path = SCHEMA_DIR / f"{name}.schema.json"
    path.write_text(json.dumps(schema, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    """Export every public P0 schema."""
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        export_schema(name, model)


if __name__ == "__main__":
    main()
