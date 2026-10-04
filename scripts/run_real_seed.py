"""Execute the P1-15 real CPU seed matrix and emit an acceptance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lowvram.benchmark_export import BenchmarkExportError, build_benchmark_run
from lowvram.doctor import run_doctor
from lowvram.dry_run import build_benchmark_dry_run
from lowvram.models.benchmark import BenchmarkRun
from lowvram.models.model import ModelInfo
from lowvram.models.orchestrator import BenchmarkOrchestratorRequest
from lowvram.models.preparation import BenchmarkPreparationRequest
from lowvram.models.recipe import Recipe
from lowvram.runtime import BenchmarkOrchestrator


@dataclass(frozen=True)
class SeedModel:
    key: str
    path: Path
    model: ModelInfo
    sha256: str


@dataclass(frozen=True)
class SeedCase:
    name: str
    model_key: str
    context_length: int
    batch_size: int
    timeout_seconds: float
    n_predict: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_benchmark(path: Path, benchmark: BenchmarkRun) -> None:
    _write_json(path, benchmark.model_dump(mode="json"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--smollm", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--environment-source", required=True)
    default_threads = max(1, min(4, os.cpu_count() or 1))
    parser.add_argument("--threads", type=int, default=default_threads)
    return parser.parse_args()


def _models(args: argparse.Namespace) -> dict[str, SeedModel]:
    smollm = SeedModel(
        key="smollm2",
        path=args.smollm,
        model=ModelInfo(
            id="smollm2-135m-instruct-q4_k_m",
            name="SmolLM2-135M-Instruct Q4_K_M",
            source=(
                "https://huggingface.co/QuantFactory/"
                "SmolLM2-135M-Instruct-GGUF"
            ),
            architecture="dense",
            parameter_count=135_000_000,
            quantization="Q4_K_M",
        ),
        sha256=_sha256(args.smollm),
    )
    qwen = SeedModel(
        key="qwen",
        path=args.qwen,
        model=ModelInfo(
            id="qwen2.5-0.5b-instruct-q4_0",
            name="Qwen2.5-0.5B-Instruct Q4_0",
            source="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            architecture="dense",
            parameter_count=500_000_000,
            quantization="Q4_0",
        ),
        sha256=_sha256(args.qwen),
    )
    return {smollm.key: smollm, qwen.key: qwen}


def _matrix() -> list[SeedCase]:
    return [
        SeedCase("smollm-repeat-1", "smollm2", 1024, 128, 120.0, 64),
        SeedCase("smollm-repeat-2", "smollm2", 1024, 128, 120.0, 64),
        SeedCase("smollm-repeat-3", "smollm2", 1024, 128, 120.0, 64),
        SeedCase("smollm-ctx2048-1", "smollm2", 2048, 256, 120.0, 64),
        SeedCase("smollm-ctx2048-2", "smollm2", 2048, 256, 120.0, 64),
        SeedCase("qwen-ctx1024-1", "qwen", 1024, 128, 120.0, 64),
        SeedCase("qwen-ctx1024-2", "qwen", 1024, 128, 120.0, 64),
        SeedCase("qwen-ctx2048-1", "qwen", 2048, 256, 120.0, 64),
        SeedCase("qwen-ctx2048-2", "qwen", 2048, 256, 120.0, 64),
        SeedCase("smollm-timeout", "smollm2", 1024, 128, 0.01, 1024),
    ]


def _recipe(case: SeedCase, model: SeedModel, threads: int) -> Recipe:
    return Recipe(
        model_id=model.model.id,
        runtime="llama.cpp",
        context_length=case.context_length,
        gpu_layers=0,
        threads=threads,
        batch_size=case.batch_size,
        extra_args=["--n-predict", str(case.n_predict)],
    )


def _dry_run_request(
    case: SeedCase,
    model: SeedModel,
    llama_cli: Path,
    threads: int,
) -> BenchmarkPreparationRequest:
    return BenchmarkPreparationRequest(
        model_path=str(model.path),
        llama_cli=str(llama_cli),
        context_length=case.context_length,
        threads=threads,
        gpu_layers=0,
        batch_size=case.batch_size,
        temperature=0.0,
        seed=42,
        prompt_version="v1",
        extra_args=("--n-predict", str(case.n_predict)),
    )


def _run_request(
    case: SeedCase,
    model: SeedModel,
    llama_cli: Path,
    threads: int,
) -> BenchmarkOrchestratorRequest:
    return BenchmarkOrchestratorRequest(
        model_path=str(model.path),
        model=model.model,
        configuration=_recipe(case, model, threads),
        llama_cli=str(llama_cli),
        prompt_version="v1",
        temperature=0.0,
        seed=42,
        timeout_seconds=case.timeout_seconds,
    )


def _validate_artifact_set(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    return {path.name for path in run_dir.iterdir()} == {
        "benchmark.json",
        "stdout.log",
        "stderr.log",
    }


def _contains_private_path(benchmark: BenchmarkRun, private_paths: list[str]) -> bool:
    payload = benchmark.model_dump_json()
    return any(path and path in payload for path in private_paths)


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    models = _models(args)

    doctor = run_doctor(
        llama_cli=str(args.llama_cli),
        model_path=args.smollm,
        output_dir=args.output_dir,
    )
    if not doctor.ready:
        raise SystemExit("P1-15 doctor preflight is BLOCKED")

    orchestrator = BenchmarkOrchestrator(runs_dir=args.output_dir)
    runs: list[dict[str, Any]] = []
    exported: list[BenchmarkRun] = []
    repeat_generation_tps: list[float] = []

    for index, case in enumerate(_matrix(), start=1):
        seed_model = models[case.model_key]
        dry_run = build_benchmark_dry_run(
            _dry_run_request(case, seed_model, args.llama_cli, args.threads)
        )
        if not dry_run.ready or dry_run.command is None:
            raise SystemExit(
                f"dry-run blocked for {case.name}: {dry_run.error_message}"
            )

        record = orchestrator.run(
            _run_request(case, seed_model, args.llama_cli, args.threads)
        )
        argv_consistent = (
            record.command is not None
            and record.command.argv == dry_run.command.argv
        )
        run_dir = args.output_dir / record.run_id
        artifact_set_valid = _validate_artifact_set(run_dir)

        try:
            benchmark = build_benchmark_run(record)
        except BenchmarkExportError as exc:
            raise SystemExit(
                f"benchmark export failed for {case.name}: {exc}"
            ) from exc

        benchmark_path = args.benchmark_dir / f"{index:02d}-{case.name}.json"
        _write_benchmark(benchmark_path, benchmark)
        exported.append(benchmark)

        performance = record.performance
        if case.name.startswith("smollm-repeat-") and performance is not None:
            repeat_generation_tps.append(
                performance.generation_tokens_per_second
            )

        runs.append(
            {
                "index": index,
                "case": case.name,
                "run_id": record.run_id,
                "model_id": seed_model.model.id,
                "model_sha256": seed_model.sha256,
                "context_length": case.context_length,
                "batch_size": case.batch_size,
                "n_predict": case.n_predict,
                "timeout_seconds": case.timeout_seconds,
                "success": record.result.success,
                "error_type": (
                    record.result.error_type.value
                    if record.result.error_type is not None
                    else None
                ),
                "runtime_version": (
                    record.detection.version
                    if record.detection is not None
                    else None
                ),
                "prompt_version": record.prompt_version,
                "prompt_sha256": record.prompt_sha256,
                "argv_consistent": argv_consistent,
                "artifact_set_valid": artifact_set_valid,
                "load_time_seconds": (
                    performance.load_time_seconds
                    if performance is not None
                    else None
                ),
                "prompt_tokens_per_second": (
                    performance.prompt_tokens_per_second
                    if performance is not None
                    else None
                ),
                "generation_tokens_per_second": (
                    performance.generation_tokens_per_second
                    if performance is not None
                    else None
                ),
                "peak_process_ram_mb": (
                    record.ram.peak_process_ram_mb
                    if record.ram is not None
                    else None
                ),
                "peak_vram_mb": benchmark.memory.peak_vram_mb,
                "verification_status": benchmark.verification.status,
            }
        )

    success_runs = [run for run in runs if run["success"]]
    failed_runs = [run for run in runs if not run["success"]]
    success_timings_valid = all(
        _positive_finite(run["load_time_seconds"])
        and _positive_finite(run["prompt_tokens_per_second"])
        and _positive_finite(run["generation_tokens_per_second"])
        for run in success_runs
    )
    ram_sane = all(
        isinstance(run["peak_process_ram_mb"], int)
        and run["peak_process_ram_mb"] > 0
        for run in success_runs
    )
    vram_sane = all(run["peak_vram_mb"] == 0 for run in success_runs)
    repeatability_ratio = (
        max(repeat_generation_tps) / min(repeat_generation_tps)
        if len(repeat_generation_tps) == 3 and min(repeat_generation_tps) > 0
        else math.inf
    )
    repeatability_cv = (
        statistics.pstdev(repeat_generation_tps)
        / statistics.mean(repeat_generation_tps)
        if len(repeat_generation_tps) == 3
        else math.inf
    )

    private_paths = [
        str(args.llama_cli),
        str(args.smollm),
        str(args.qwen),
        str(Path.cwd()),
    ]
    privacy_clean = all(
        not _contains_private_path(benchmark, private_paths)
        for benchmark in exported
    )

    acceptance = {
        "attempt_count_at_least_10": len(runs) >= 10,
        "success_count_at_least_9": len(success_runs) >= 9,
        "failed_run_present": len(failed_runs) >= 1,
        "two_models": len({run["model_id"] for run in runs}) >= 2,
        "two_contexts": (
            len({run["context_length"] for run in success_runs}) >= 2
        ),
        "two_configurations": (
            len(
                {
                    (
                        run["model_id"],
                        run["context_length"],
                        run["batch_size"],
                    )
                    for run in success_runs
                }
            )
            >= 2
        ),
        "prompt_consistent": all(
            run["prompt_version"] == "v1" for run in runs
        ),
        "runtime_version_consistent": (
            len(
                {
                    run["runtime_version"]
                    for run in runs
                    if run["runtime_version"] is not None
                }
            )
            == 1
        ),
        "argv_consistent": all(run["argv_consistent"] for run in runs),
        "artifact_sets_valid": all(
            run["artifact_set_valid"] for run in runs
        ),
        "success_timings_valid": success_timings_valid,
        "ram_sane": ram_sane,
        "cpu_only_vram_sane": vram_sane,
        "repeatability_no_order_of_magnitude_outlier": (
            repeatability_ratio < 10.0
        ),
        "load_time_contract_resolved": all(
            run["load_time_seconds"] is not None for run in success_runs
        ),
        "privacy_clean_benchmark_exports": privacy_clean,
    }
    technical_pass = all(acceptance.values())

    report = {
        "schema_version": "p1.15.0",
        "environment_source": args.environment_source,
        "github_actions": os.getenv("GITHUB_ACTIONS") == "true",
        "runner_os": os.getenv("RUNNER_OS"),
        "runner_arch": os.getenv("RUNNER_ARCH"),
        "threads": args.threads,
        "runtime_binary_sha256": _sha256(args.llama_cli),
        "models": {
            key: {
                "id": value.model.id,
                "sha256": value.sha256,
                "quantization": value.model.quantization,
            }
            for key, value in models.items()
        },
        "runs": runs,
        "repeatability": {
            "generation_tokens_per_second": repeat_generation_tps,
            "max_min_ratio": repeatability_ratio,
            "coefficient_of_variation": repeatability_cv,
        },
        "acceptance": acceptance,
        "technical_real_seed_status": (
            "PASS" if technical_pass else "FAIL"
        ),
        "p1_final_gate": (
            "BLOCKED_PERSONAL_CONSUMER_HARDWARE_REQUIRED"
            if args.environment_source == "github-actions-hosted"
            else ("PASS" if technical_pass else "FAIL")
        ),
    }
    _write_json(args.report_json, report)

    md_lines = [
        "# P1-15 Real Benchmark Seed Report",
        "",
        f"- Environment source: `{args.environment_source}`",
        f"- Attempts: **{len(runs)}**",
        f"- Successful runs: **{len(success_runs)}**",
        f"- Failed runs: **{len(failed_runs)}**",
        (
            "- Technical real-seed status: "
            f"**{report['technical_real_seed_status']}**"
        ),
        f"- P1 Final Gate: **{report['p1_final_gate']}**",
        "",
        "## Repeatability",
        "",
        f"- Generation tok/s: `{repeat_generation_tps}`",
        f"- max/min ratio: `{repeatability_ratio:.4f}`",
        f"- coefficient of variation: `{repeatability_cv:.4f}`",
        "",
        "## Acceptance",
        "",
    ]
    md_lines.extend(
        f"- {'PASS' if value else 'FAIL'} — {name}"
        for name, value in acceptance.items()
    )
    md_lines.extend(
        [
            "",
            "## Gate note",
            "",
            (
                "The workflow executed real llama.cpp binaries and real GGUF "
                "files on a GitHub-hosted CPU runner. The original P1-15 plan "
                "asks for runs on the developer's own computer, so CI evidence "
                "alone does not satisfy that provenance requirement."
            ),
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(md_lines), encoding="utf-8")

    return 0 if technical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
