# Contributing to LowVRAM

P0 is deliberately limited to repository structure, data contracts, validation, tests, CI, and documentation. Do not add benchmark execution, runtime integration, GPU monitoring, recommendation logic, database aggregation, model downloading, or Web UI during P0.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy
python scripts/validate_fixtures.py
```

Any data-contract change must update its Pydantic model, exported JSON Schema, fixtures, tests, and relevant documentation in the same pull request. Unknown fields are rejected by design; do not weaken validation merely to accept an undocumented payload.

## Definition of done

A change is complete only when implementation, tests, error handling, documentation, and CI all pass.
