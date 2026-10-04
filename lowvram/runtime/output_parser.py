"""P1-09 llama.cpp textual timing output parser."""

import math
import re
from dataclasses import dataclass

from lowvram.models.benchmark import ErrorType
from lowvram.models.output_parser import (
    LlamaCppOutputParseResult,
    LlamaCppTimingMetrics,
)

_LOAD_LABEL = re.compile(r"\bload\s+time\b", re.IGNORECASE)
_PROMPT_LABEL = re.compile(r"\bprompt\s+eval\s+time\b", re.IGNORECASE)
_EVAL_LABEL = re.compile(
    r"\b(?:(?:generation|token\s+generation)\s+)?eval\s+time\b",
    re.IGNORECASE,
)
_LOAD_TIME_VALUE = re.compile(r"=\s*(?P<time_ms>\S+)\s*ms\b", re.IGNORECASE)
_TIMING_VALUES = re.compile(
    r"=\s*(?P<time_ms>\S+)\s*ms\b"
    r".*?"
    r"\(\s*\S+\s*ms\s+per\s+token\s*,\s*"
    r"(?P<tokens_per_second>\S+)\s+tokens?\s+per\s+second\s*\)",
    re.IGNORECASE,
)
_COMPACT_HINT = re.compile(r"\[\s*Prompt:.*Generation:", re.IGNORECASE)
_COMPACT_TIMING = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt_tps>\S+)\s*t/s\s*\|\s*"
    r"Generation:\s*(?P<generation_tps>\S+)\s*t/s\s*\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _TimingLine:
    time_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class _TimingBlock:
    load_time_seconds: float | None
    prompt: _TimingLine
    generation: _TimingLine


@dataclass(frozen=True)
class _CompactTiming:
    prompt_tokens_per_second: float
    generation_tokens_per_second: float


def _parse_positive_finite(value: str) -> float | None:
    """Parse a strictly positive finite floating-point number."""
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _parse_load_time_line(line: str) -> float | None:
    """Extract a directly printed positive finite llama.cpp load time."""
    match = _LOAD_TIME_VALUE.search(line)
    if match is None:
        return None
    time_ms = _parse_positive_finite(match.group("time_ms"))
    if time_ms is None:
        return None
    return time_ms / 1000.0


def _is_prompt_timing_line(line: str) -> bool:
    return _PROMPT_LABEL.search(line) is not None


def _is_generation_timing_line(line: str) -> bool:
    if _is_prompt_timing_line(line):
        return False
    return _EVAL_LABEL.search(line) is not None


def _parse_timing_line(line: str) -> _TimingLine | None:
    """Extract milliseconds and direct tokens/sec from one detailed timing line."""
    match = _TIMING_VALUES.search(line)
    if match is None:
        return None

    time_ms = _parse_positive_finite(match.group("time_ms"))
    tokens_per_second = _parse_positive_finite(match.group("tokens_per_second"))
    if time_ms is None or tokens_per_second is None:
        return None

    return _TimingLine(
        time_seconds=time_ms / 1000.0,
        tokens_per_second=tokens_per_second,
    )


def _parse_latest_complete_block(text: str) -> _TimingBlock | None:
    """Return the final complete detailed prompt/eval timing block."""
    latest_load_time_seconds: float | None = None
    pending_prompt: _TimingLine | None = None
    latest_complete: _TimingBlock | None = None
    latest_timing_sequence_complete = False
    saw_timing_label = False

    for line in text.splitlines():
        if _LOAD_LABEL.search(line) is not None:
            latest_load_time_seconds = _parse_load_time_line(line)
            continue

        if _is_prompt_timing_line(line):
            saw_timing_label = True
            pending_prompt = _parse_timing_line(line)
            latest_timing_sequence_complete = False
            continue

        if not _is_generation_timing_line(line):
            continue

        saw_timing_label = True
        generation = _parse_timing_line(line)
        if pending_prompt is not None and generation is not None:
            latest_complete = _TimingBlock(
                load_time_seconds=latest_load_time_seconds,
                prompt=pending_prompt,
                generation=generation,
            )
            latest_timing_sequence_complete = True
        else:
            latest_timing_sequence_complete = False
        pending_prompt = None

    if not saw_timing_label or not latest_timing_sequence_complete:
        return None
    return latest_complete


def _parse_latest_compact_timing(text: str) -> _CompactTiming | None:
    """Return the final valid b10336-style compact timing line."""
    saw_compact_hint = False
    latest: _CompactTiming | None = None

    for line in text.splitlines():
        if _COMPACT_HINT.search(line) is None:
            continue
        saw_compact_hint = True
        latest = None
        match = _COMPACT_TIMING.search(line)
        if match is None:
            continue
        prompt_tps = _parse_positive_finite(match.group("prompt_tps"))
        generation_tps = _parse_positive_finite(match.group("generation_tps"))
        if prompt_tps is None or generation_tps is None:
            continue
        latest = _CompactTiming(
            prompt_tokens_per_second=prompt_tps,
            generation_tokens_per_second=generation_tps,
        )

    if not saw_compact_hint:
        return None
    return latest


def _contains_detailed_timing_labels(text: str) -> bool:
    return _PROMPT_LABEL.search(text) is not None or _EVAL_LABEL.search(text) is not None


def parse_llama_cpp_output(
    *,
    stdout: str,
    stderr: str,
) -> LlamaCppOutputParseResult:
    """Parse the final supported llama.cpp timing block from captured output.

    Detailed prompt/eval timing remains preferred because it carries duration evidence.
    Current llama-cli builds such as b10336 emit only a compact Prompt/Generation throughput
    line; missing duration or load-time values remain null rather than being derived.
    """
    combined = "\n".join(part for part in (stdout, stderr) if part)

    block = _parse_latest_complete_block(combined)
    if block is not None:
        return LlamaCppOutputParseResult(
            success=True,
            metrics=LlamaCppTimingMetrics(
                timing_format="detailed",
                load_time_seconds=block.load_time_seconds,
                prompt_eval_time_seconds=block.prompt.time_seconds,
                prompt_tokens_per_second=block.prompt.tokens_per_second,
                eval_time_seconds=block.generation.time_seconds,
                generation_tokens_per_second=block.generation.tokens_per_second,
            ),
        )

    if _contains_detailed_timing_labels(combined):
        return _parse_failure()

    compact = _parse_latest_compact_timing(combined)
    if compact is not None:
        return LlamaCppOutputParseResult(
            success=True,
            metrics=LlamaCppTimingMetrics(
                timing_format="compact",
                load_time_seconds=None,
                prompt_eval_time_seconds=None,
                prompt_tokens_per_second=compact.prompt_tokens_per_second,
                eval_time_seconds=None,
                generation_tokens_per_second=compact.generation_tokens_per_second,
            ),
        )

    return _parse_failure()


def _parse_failure() -> LlamaCppOutputParseResult:
    return LlamaCppOutputParseResult(
        success=False,
        error_type=ErrorType.PARSE_FAILED,
        error_message=(
            "llama.cpp output does not end with a complete, finite, positive "
            "supported timing block"
        ),
    )
