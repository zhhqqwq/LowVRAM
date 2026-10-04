# RAM Monitor — P1-03

P1-03 adds a lifecycle RAM monitor for one target process and the child processes it creates.

## Interface

```python
from lowvram.collectors import RamMonitor

monitor = RamMonitor(pid)
monitor.start()

# benchmark process runs here

result = monitor.stop()
```

`start()` captures the baseline and starts background sampling. `sample()` is also public for an immediate explicit sample. `stop()` stops the background sampler and returns `RamMonitorResult`.

## Sampling

The P1-03 default sampling interval is fixed at 100 ms.

Process RAM is the sum of RSS for the target PID plus recursively discovered child processes. Once a child PID has been discovered, it remains tracked while alive even if the original parent exits.

System RAM is the whole-system used physical memory reported by `psutil.virtual_memory().used`.

All memory values use the LowVRAM MB unit.

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

The original P1-03 required outputs are represented by `peak_process_ram_mb` and `peak_system_ram_mb`. Delta is defined as peak minus baseline.

## Boundary

P1-03 does not inspect GPU memory and does not invoke a runtime. P1-04 owns VRAM monitoring; later runtime/orchestrator work will connect these monitors to llama.cpp execution.
