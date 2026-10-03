"""Assert that valid fixtures pass and invalid fixtures fail."""

from pathlib import Path

from lowvram.validators import DataValidationError, validate_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def main() -> int:
    """Validate fixture expectations."""
    failures: list[str] = []
    for path in sorted((FIXTURES / "valid").glob("*.json")):
        try:
            validate_file(path, "benchmark")
        except DataValidationError as exc:
            failures.append(f"expected VALID: {path.name}: {exc}")
    for path in sorted((FIXTURES / "invalid").glob("*.json")):
        try:
            validate_file(path, "benchmark")
        except DataValidationError:
            continue
        failures.append(f"expected INVALID: {path.name}")
    if failures:
        print("\n".join(failures))
        return 1
    print("Fixture validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
