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


@dataclass(frozen=True)
class _TimingLine:
    time_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class _TimingBlock:
    load_time_seconds: float | None
    prompt: _TimingLine
    generation: _TimingLine


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


def _parse_positive_finite(value: str) -> float | None:
    """Parse a strictly positive finite floating-point number."""
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _parse_timing_line(line: str) -> _TimingLine | None:
    """Extract milliseconds and direct tokens/sec from one timing line."""
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
    """Return the final complete prompt/eval block when the latest block is valid."""
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


def parse_llama_cpp_output(
    *,
    stdout: str,
    stderr: str,
) -> LlamaCppOutputParseResult:
    """Parse the final complete llama.cpp timing block from captured process output.

    stdout is scanned before stderr because P1-05 captures the streams separately and cannot
    reconstruct their original interleaving. A complete later block replaces an earlier one.
    """
    combined = "\n".join(part for part in (stdout, stderr) if part)
    block = _parse_latest_complete_block(combined)
    if block is None:
        return LlamaCppOutputParseResult(
            success=False,
            error_type=ErrorType.PARSE_FAILED,
            error_message=(
                "llama.cpp output does not end with a complete, finite, positive "
                "prompt/eval timing block"
            ),
        )

    return LlamaCppOutputParseResult(
        success=True,
        metrics=LlamaCppTimingMetrics(
            load_time_seconds=block.load_time_seconds,
            prompt_eval_time_seconds=block.prompt.time_seconds,
            prompt_tokens_per_second=block.prompt.tokens_per_second,
            eval_time_seconds=block.generation.time_seconds,
            generation_tokens_per_second=block.generation.tokens_per_second,
        ),
    )
