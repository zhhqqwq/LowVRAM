## Summary

Describe the issue/phase implemented and why the change is needed.

## Scope check

- [ ] This PR implements only its declared issue/phase.
- [ ] Later-phase functionality is not bundled into this change.
- [ ] New behavior has tests.
- [ ] User-visible behavior is documented.
- [ ] Data-contract changes update models, schemas, fixtures, tests, and docs together.

## Quality gate

- [ ] `pytest`
- [ ] `ruff check .`
- [ ] `mypy`
- [ ] `python scripts/validate_data.py`
- [ ] `python scripts/validate_fixtures.py`
