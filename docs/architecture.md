# Architecture — P0

P0 is a contract-and-validation layer, not a benchmark runner.

```text
JSON file
   |
   v
Typer CLI: lowvram validate
   |
   +--> Pydantic v2 strict model validation
   |
   +--> JSON Schema Draft 2020-12 validation
   |
   v
VALID / INVALID + field path
```

The package exposes four public data models: `HardwareInfo`, `ModelInfo`, `BenchmarkRun`, and `Recipe`. Nested models keep the public contracts structured while `extra="forbid"` prevents silent acceptance of unknown critical fields.

Checked-in schemas are exported from the Pydantic models by `scripts/export_schemas.py`, then CI regenerates them and requires a clean git diff. This makes schema drift visible.

Data directories exist for future phases but remain empty in P0. No collector, monitor, runtime adapter, database builder, aggregator, compatibility engine, estimator, or recommender is implemented.
