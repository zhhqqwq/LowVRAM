# Contributing to LowVRAM

LowVRAM is developed phase by phase. A pull request should implement only the issue and phase it targets; do not bundle later-phase functionality into earlier work.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy
python scripts/validate_fixtures.py
```

Any data-contract change must update its Pydantic model, exported JSON Schema, fixtures, tests, and relevant documentation in the same pull request. Unknown fields are rejected by design; do not weaken validation merely to accept an undocumented payload.

P1 collector and runner changes must remain read-only with respect to the user's system unless an issue explicitly requires otherwise. Do not collect usernames, hostnames, IP addresses, MAC addresses, device serial numbers, home paths, environment secrets, or API keys.

## Definition of done

A change is complete only when implementation, tests, error handling, documentation, and CI all pass.
