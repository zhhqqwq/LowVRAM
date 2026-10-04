# Runtime Adapter — P1-05

P1-05 defines the process-execution boundary between LowVRAM and external model runtimes.

## Public interface

```python
from lowvram.models import RuntimeExecutionRequest
from lowvram.runtime import LlamaCppRuntimeAdapter

adapter = LlamaCppRuntimeAdapter()
request = RuntimeExecutionRequest(
    executable="/path/to/llama-cli",
    arguments=["--version"],
    timeout_seconds=10,
)

result = adapter.execute(request)
```

`RuntimeAdapter` defines two operations:

- `build_command(request) -> list[str]`
- `execute(request) -> RuntimeExecutionResult`

`SubprocessRuntimeAdapter` implements shell-free local process execution. The initial
`LlamaCppRuntimeAdapter` supplies the stable runtime name `llama.cpp`.

## Structured execution result

Every execution records:

- runtime name
- success/failure
- exit code when the process actually exited
- stdout
- stderr
- elapsed seconds
- normalized error type
- error message

P1-05 reuses the existing benchmark error taxonomy:

- missing/unstartable executable → `runtime_not_found`
- timeout → `timeout`
- non-zero process exit → `process_crash`

A timeout intentionally reports no exit code because the adapter terminates the process
rather than treating the forced termination code as the runtime's own result.

## Process lifecycle

The adapter starts the process with `shell=False`, waits with the request timeout, captures
stdout/stderr as UTF-8 text, and returns exit code 0 as success.

On timeout, LowVRAM terminates the runtime process and recursively discovered descendants
using `psutil`, escalating surviving processes to kill. This prevents a timed-out benchmark
from intentionally leaving its launched runtime tree running.

## Scope boundary

P1-05 does not search for llama.cpp and does not translate a LowVRAM `Recipe` into concrete
llama.cpp benchmark flags. The caller supplies an explicit executable and argv.

Executable discovery belongs to P1-06. Higher-level recipe-to-command construction and
benchmark orchestration remain later P1 work.

The execution result does not persist environment variables or a working-directory field, so
secrets and private paths are not added to benchmark evidence by this layer.
