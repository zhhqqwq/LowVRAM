# System Collector — P1-01

P1-01 adds a read-only hardware inventory command:

```bash
lowvram system
```

The command prints one structured JSON document that validates as the existing
`HardwareInfo` contract.

## P1-01 acceptance fields

The CLI output must include all of the following categories:

- OS
- CPU
- RAM
- NVIDIA GPU
- VRAM
- NVIDIA Driver
- CUDA
- Python Version

The current JSON representation is:

```json
{
  "cpu": {
    "name": "...",
    "architecture": "...",
    "physical_cores": 8,
    "logical_cores": 16
  },
  "gpu": [
    {
      "vendor": "nvidia",
      "name": "...",
      "vram_total_mb": 8192,
      "driver_version": "...",
      "cuda_version": "..."
    }
  ],
  "ram_total_mb": 32768,
  "os": {
    "name": "Linux",
    "version": "...",
    "architecture": "x86_64"
  },
  "driver": {
    "nvidia_driver_version": "...",
    "cuda_version": "..."
  },
  "python_version": "3.11.9"
}
```

## Collection behavior

OS data comes from Python's `platform` module.

CPU identity and core counts use platform information plus `psutil`.

Total physical RAM uses `psutil.virtual_memory().total`.

NVIDIA information is collected with `nvidia-smi`. The static P1-01 projection records:

- GPU name;
- total VRAM in MB;
- NVIDIA driver version;
- CUDA compatibility version reported by `nvidia-smi`.

The CUDA value is the driver compatibility version shown by `nvidia-smi`; it is not proof
that a matching CUDA Toolkit is installed locally.

Python Version comes from `platform.python_version()`.

## CPU-only behavior

If `nvidia-smi` is absent, times out, returns an error, or no usable GPU rows can be parsed,
system collection still succeeds with:

```json
{
  "gpu": [],
  "driver": {
    "nvidia_driver_version": null,
    "cuda_version": null
  }
}
```

This is the expected CPU-only representation.

## Privacy boundary

The collector does not record username, hostname, IP address, MAC address, serial number,
home directory, environment secrets, or command history.

## Scope boundary

P1-01 is inventory only. Dynamic GPU utilization, RAM monitoring, VRAM monitoring, runtime
execution, and benchmark orchestration are later P1 concerns.
