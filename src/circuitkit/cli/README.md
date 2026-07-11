# cli

The `circuitkit` command-line entry point. It runs discovery, evaluation, pruning, quantization, benchmarking, and debugging from the shell.

## Key modules

- `main.py` — Click command group defining the CLI: `discover`, `discover-yaml`, `discover-smart`, `evaluate`, `prune`, `quantize`, `benchmark`, and related subcommands; wires in logging, config, and debug groups.
- `config.py` — `ConfigManager`: loads, validates, generates, and saves YAML configuration (model, discovery, pruning, output paths).
- `debug.py` — `debug` command group: PyTorch debugging toggles, memory cleanup, and profiling tools (Rich-formatted output).
- `utils.py` — CLI helpers: `setup_logging`, `validate_model_name`, and supported-model listing.

## Public API / entry points

The console entry point is the `cli` Click group in `main.py` (installed as the `circuitkit` command). Subcommands and the `debug` group are its primary interface; `ConfigManager` backs `--config` handling.

## How it fits

A command-line front-end over `circuitkit.api` and `circuitkit.applications`. It parses arguments and config into the dict-config engine, and uses `utils.memory` and `utils.exceptions` for validation and memory-aware defaults.
