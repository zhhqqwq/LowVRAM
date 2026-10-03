"""Validate checked-in LowVRAM JSON data files."""

from pathlib import Path

from lowvram.validators import DataValidationError, validate_file

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> int:
    """Validate every JSON file below data/."""
    failed = False
    files = sorted(DATA_DIR.rglob("*.json"))
    for path in files:
        try:
            validate_file(path)
        except DataValidationError as exc:
            print(f"INVALID {path.relative_to(ROOT)}\n{exc}")
            failed = True
        else:
            print(f"VALID   {path.relative_to(ROOT)}")
    if not files:
        print("No data files found; P0 data directories are intentionally empty.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
