"""P1-11 evidence-based benchmark failure classification."""

import re
from collections.abc import Sequence

from lowvram.models.benchmark import ErrorType
from lowvram.models.failure import (
    FailureClassificationRequest,
    FailureClassificationResult,
    FailureEvidenceSource,
    FailureStage,
)

_TextRule = tuple[str, re.Pattern[str]]

_OOM_RULES: tuple[_TextRule, ...] = (
    (
        "oom.cuda_error",
        re.compile(r"\bcuda[^\n]{0,120}\bout of memory\b", re.IGNORECASE),
    ),
    (
        "oom.cuda_memory_allocation",
        re.compile(r"\bcudaErrorMemoryAllocation\b", re.IGNORECASE),
    ),
    (
        "oom.device_memory",
        re.compile(
            r"\b(?:hipErrorOutOfMemory|out of device memory)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oom.cpu_bad_alloc",
        re.compile(r"\b(?:std::)?bad_alloc(?:\s+error)?\b", re.IGNORECASE),
    ),
    (
        "oom.explicit_out_of_memory",
        re.compile(r"\bout of memory\b", re.IGNORECASE),
    ),
    (
        "oom.explicit_memory_failure",
        re.compile(
            r"\b(?:cannot allocate memory|not enough memory|ENOMEM)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oom.allocation_failure",
        re.compile(
            r"\bfailed to allocate\b[^\n]{0,120}"
            r"\b(?:bytes|kib|mib|gib|kb|mb|gb|buffer|memory)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oom.cuda_malloc_failure",
        re.compile(r"\bcudaMalloc\b[^\n]{0,120}\bfailed\b", re.IGNORECASE),
    ),
)

_MODEL_NOT_FOUND_RULES: tuple[_TextRule, ...] = (
    (
        "model_not_found.gguf_open",
        re.compile(
            r"failed to open GGUF file[^\n]*"
            r"(?:no such file or directory|file not found|cannot find the file)",
            re.IGNORECASE,
        ),
    ),
    (
        "model_not_found.model_file",
        re.compile(
            r"\bmodel file\b[^\n]*(?:not found|does not exist)",
            re.IGNORECASE,
        ),
    ),
)

_MODEL_LOAD_RULES: tuple[_TextRule, ...] = (
    (
        "model_load_failed.explicit",
        re.compile(
            r"\b(?:failed to load model|error loading model|model load failed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "model_load_failed.gguf_open",
        re.compile(r"\bfailed to open GGUF file\b", re.IGNORECASE),
    ),
)


def classify_failure(
    request: FailureClassificationRequest,
) -> FailureClassificationResult:
    """Classify one failed benchmark attempt using deterministic precedence."""
    if request.model_path_exists is False:
        return _structured(
            ErrorType.MODEL_NOT_FOUND,
            "model_not_found.preflight",
            _message(request, "model file does not exist"),
        )

    upstream = _upstream_error_type(request)

    if upstream == ErrorType.RUNTIME_NOT_FOUND:
        return _structured(
            ErrorType.RUNTIME_NOT_FOUND,
            "runtime_not_found.structured",
            _message(request, "runtime executable is unavailable"),
        )

    if upstream == ErrorType.MODEL_NOT_FOUND:
        return _structured(
            ErrorType.MODEL_NOT_FOUND,
            "model_not_found.structured",
            _message(request, "model file is unavailable"),
        )

    if upstream == ErrorType.TIMEOUT:
        return _structured(
            ErrorType.TIMEOUT,
            "timeout.structured",
            _message(request, "runtime exceeded the configured timeout"),
        )

    if upstream == ErrorType.OUT_OF_MEMORY:
        return _structured(
            ErrorType.OUT_OF_MEMORY,
            "oom.structured",
            _message(request, "runtime could not allocate required memory"),
        )

    if upstream == ErrorType.MODEL_LOAD_FAILED:
        return _structured(
            ErrorType.MODEL_LOAD_FAILED,
            "model_load_failed.structured",
            _message(request, "runtime failed to load the model"),
        )

    if request.stage == FailureStage.RUNTIME_EXECUTION:
        matched = _match_rules(request, _OOM_RULES)
        if matched is not None:
            rule, sources = matched
            return FailureClassificationResult(
                error_type=ErrorType.OUT_OF_MEMORY,
                rule=rule,
                message=_message(
                    request,
                    "runtime output contains explicit memory-allocation failure evidence",
                ),
                evidence_sources=_with_structured_source(request, sources),
            )

        matched = _match_rules(request, _MODEL_NOT_FOUND_RULES)
        if matched is not None:
            rule, sources = matched
            return FailureClassificationResult(
                error_type=ErrorType.MODEL_NOT_FOUND,
                rule=rule,
                message=_message(
                    request,
                    "runtime output indicates the model file became unavailable",
                ),
                evidence_sources=_with_structured_source(request, sources),
            )

        matched = _match_rules(request, _MODEL_LOAD_RULES)
        if matched is not None:
            rule, sources = matched
            return FailureClassificationResult(
                error_type=ErrorType.MODEL_LOAD_FAILED,
                rule=rule,
                message=_message(request, "runtime failed while loading the model"),
                evidence_sources=_with_structured_source(request, sources),
            )

    if request.stage == FailureStage.OUTPUT_PARSE or upstream == ErrorType.PARSE_FAILED:
        return _structured(
            ErrorType.PARSE_FAILED,
            "parse_failed.structured",
            _message(request, "runtime output could not be parsed"),
        )

    execution = request.execution
    if upstream == ErrorType.PROCESS_CRASH or (
        execution is not None
        and not execution.success
        and execution.exit_code is not None
    ):
        return _structured(
            ErrorType.PROCESS_CRASH,
            "process_crash.exit",
            _message(request, "runtime process exited unsuccessfully"),
        )

    return _structured(
        ErrorType.UNKNOWN,
        "unknown.fallback",
        _message(request, "benchmark failed without recognized evidence"),
    )


def _upstream_error_type(request: FailureClassificationRequest) -> ErrorType | None:
    if request.upstream_error_type is not None:
        return request.upstream_error_type
    if request.execution is not None:
        return request.execution.error_type
    return None


def _message(request: FailureClassificationRequest, default: str) -> str:
    if request.message:
        return request.message
    if request.execution is not None and request.execution.error_message:
        return request.execution.error_message
    return default


def _structured(
    error_type: ErrorType,
    rule: str,
    message: str,
) -> FailureClassificationResult:
    return FailureClassificationResult(
        error_type=error_type,
        rule=rule,
        message=message,
        evidence_sources=(FailureEvidenceSource.STRUCTURED,),
    )


def _match_rules(
    request: FailureClassificationRequest,
    rules: Sequence[_TextRule],
) -> tuple[str, tuple[FailureEvidenceSource, ...]] | None:
    texts = _evidence_texts(request)
    for rule, pattern in rules:
        sources = tuple(
            source for source, text in texts if text and pattern.search(text) is not None
        )
        if sources:
            return rule, sources
    return None


def _evidence_texts(
    request: FailureClassificationRequest,
) -> tuple[tuple[FailureEvidenceSource, str], ...]:
    execution = request.execution
    return (
        (
            FailureEvidenceSource.STDERR,
            execution.stderr if execution is not None else "",
        ),
        (
            FailureEvidenceSource.STDOUT,
            execution.stdout if execution is not None else "",
        ),
        (FailureEvidenceSource.MESSAGE, request.message or ""),
    )


def _with_structured_source(
    request: FailureClassificationRequest,
    sources: tuple[FailureEvidenceSource, ...],
) -> tuple[FailureEvidenceSource, ...]:
    execution = request.execution
    if (
        request.upstream_error_type is not None
        or (
            execution is not None
            and (execution.error_type is not None or execution.exit_code is not None)
        )
    ):
        return (*sources, FailureEvidenceSource.STRUCTURED)
    return sources
