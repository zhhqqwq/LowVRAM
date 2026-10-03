## Summary

Describe the P0 change and why it is needed.

## P0 scope check

- [ ] This PR does not implement benchmark execution, llama.cpp integration, GPU monitoring, recommendations, aggregation, model downloading, or a Web UI.
- [ ] New behavior has tests.
- [ ] Data-contract changes update schemas, fixtures, and documentation together.

## Quality gate

- [ ] `pytest`
- [ ] `ruff check .`
- [ ] `mypy`
- [ ] `python scripts/validate_fixtures.py`
