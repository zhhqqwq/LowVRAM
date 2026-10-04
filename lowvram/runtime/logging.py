"""P1-14 atomic run-artifact persistence."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from lowvram.models.orchestrator import BenchmarkOrchestrationRecord


@dataclass(frozen=True)
class RunArtifactPaths:
    """Committed paths for one persisted benchmark attempt."""

    run_dir: Path
    benchmark_json: Path
    stdout_log: Path
    stderr_log: Path


class RunLoggingError(RuntimeError):
    """Raised when a complete run-artifact set cannot be committed."""

    def __init__(self, *, run_id: str, runs_root: Path, message: str) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.runs_root = runs_root
        self.message = message


def render_benchmark_orchestration_record(
    record: BenchmarkOrchestrationRecord,
) -> str:
    """Validate and render the benchmark record as stable UTF-8 JSON text."""
    payload = record.model_dump(mode="json")
    schema = BenchmarkOrchestrationRecord.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def persist_run_artifacts(
    record: BenchmarkOrchestrationRecord,
    runs_root: Path = Path("runs"),
) -> RunArtifactPaths:
    """Atomically publish benchmark.json, stdout.log, and stderr.log as one run directory."""
    final_dir = runs_root / record.run_id
    staging_dir: Path | None = None

    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise RunLoggingError(
                run_id=record.run_id,
                runs_root=runs_root,
                message=f"run directory already exists: {final_dir}",
            )

        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{record.run_id}.tmp-",
                dir=runs_root,
            )
        )
        benchmark_json = staging_dir / "benchmark.json"
        stdout_log = staging_dir / "stdout.log"
        stderr_log = staging_dir / "stderr.log"

        _write_utf8_text(
            benchmark_json,
            render_benchmark_orchestration_record(record),
        )
        execution = record.execution
        _write_utf8_text(
            stdout_log,
            execution.stdout if execution is not None else "",
        )
        _write_utf8_text(
            stderr_log,
            execution.stderr if execution is not None else "",
        )

        os.rename(staging_dir, final_dir)
        staging_dir = None
    except RunLoggingError:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except Exception as exc:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RunLoggingError(
            run_id=record.run_id,
            runs_root=runs_root,
            message=f"failed to persist run artifacts for {record.run_id}: {exc}",
        ) from exc

    return RunArtifactPaths(
        run_dir=final_dir,
        benchmark_json=final_dir / "benchmark.json",
        stdout_log=final_dir / "stdout.log",
        stderr_log=final_dir / "stderr.log",
    )


def _write_utf8_text(path: Path, content: str) -> None:
    """Write already-decoded text without shell processing or added newline conversion."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
