# System Collector — P1-01

P1-01 adds a read-only hardware inventory command:

```bash
lowvram system
```

The command prints one JSON document that validates as the existing `HardwareInfo` contract.

## Collected fields

- operating system name, release, and architecture
- CPU model, architecture, physical core count, and logical core count
- total system RAM in MB
- NVIDIA GPU name and total VRAM in MB
- NVIDIA driver version
- CUDA compatibility version reported by `nvidia-smi`
- Python version

## NVIDIA behavior

NVIDIA information is collected with `nvidia-smi`. If the executable is absent, times out, returns an error, or no usable GPU rows can be parsed, system collection still succeeds with `gpu: []`. This is the expected CPU-only representation in the LowVRAM hardware schema.

P1-01 records inventory only. GPU utilization and process/system VRAM sampling belong to later P1 monitor issues.

## Privacy boundary

The collector does not record username, hostname, IP address, MAC address, serial number, home directory, environment secrets, or command history.
