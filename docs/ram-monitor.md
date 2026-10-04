# RAM Monitor — P1-03

P1-03 adds a lifecycle RAM monitor for one target process and the child processes it creates.

## Rebaseline acceptance contract

Under the restarted P1 acceptance sequence, P1-03 must:

- use `psutil`;
- monitor the target process plus recursively discovered child processes;
- report `peak_process_ram_mb`;
- report `peak_system_ram_mb`;
- sample every 100 ms by default;
- tolerate target/child disappearance and supported psutil process-access failures;
- keep monitoring responsibilities separate from VRAM/runtime work.

The implementation also records baseline/current/delta values because later orchestration needs
them, but the original required outputs remain the process and system RAM peaks.

## Interface

```python
from lowvram.collectors import RamMonitor

monitor = RamMonitor(pid)
monitor.start()

# benchmark process runs here

result = monitor.stop()
```

`start()` captures the baseline and starts background sampling. `sample()` is public for
an immediate explicit sample. `stop()` stops the background sampler and returns
`RamMonitorResult`.

The lifecycle is one-shot: a monitor cannot be sampled before start, started twice, or used
again after stop.

## Sampling

The default sampling interval is fixed at:

```text
0.1 seconds = 100 ms
```

Process RAM is the sum of RSS for:

1. the target PID;
2. recursively discovered descendants;
3. already discovered child PIDs that remain alive after the root process exits.

A process that disappears, becomes inaccessible, or is reported as a zombie is skipped rather
than converted into an invented measurement. Supported psutil process-state exceptions are:

- `NoSuchProcess`;
- `AccessDenied`;
- `ZombieProcess`.

System RAM is the whole-system used physical memory reported by
`psutil.virtual_memory().used`.

## Units

LowVRAM stores these values as integer MB. The current implementation converts bytes using
`1024 * 1024` bytes per stored MB and truncates fractional units.

## Result

`RamMonitorResult` records:

- `baseline_process_ram_mb`
- `current_process_ram_mb`
- `peak_process_ram_mb`
- `delta_process_ram_mb`
- `baseline_system_ram_mb`
- `current_system_ram_mb`
- `peak_system_ram_mb`
- `delta_system_ram_mb`
- `sample_count`
- `sample_interval_seconds`

For both process and system RAM:

```text
delta = peak - baseline
peak >= baseline
peak >= current
```

The model rejects inconsistent aggregates.

## Boundary

P1-03 does not inspect GPU memory and does not invoke a model runtime.

P1-04 owns VRAM monitoring. Runtime execution and Benchmark orchestration remain later P1
Gates.

After this rebaseline passes, the next active Gate is P1-04 VRAM Monitor.
