# utils

Low-level helpers used across CircuitKit: device, config, and logging setup, memory and performance tooling, caching, and error handling.

## Key modules

- `device.py` — `get_device`: auto-detects the best compute device (CUDA > MPS > CPU) or forces a requested one.
- `config.py` — `DEFAULT_CONFIG` and config-merging helpers for the dict-config discovery/eval pipeline.
- `logging.py` — logging setup, warning-filter configuration, and `get_logger`.
- `memory.py` — memory-requirement checks and memory-efficient config suggestions per model/algorithm.
- `exceptions.py` — `CircuitKitError` hierarchy (with context/suggestions) and algorithm validation registries.
- `artifacts.py` — read/write sidecar JSON metadata next to `.pt` artifacts.
- `token_utils.py` — `TokenIDGenerator`: model-agnostic token ID generation with no hardcoded defaults.
- `dataset_cache.py` — `DatasetCache` / `CorruptionCache`: cache datasets and corruptions with tokenizer-compatibility metadata.
- `bootstrap.py` — `bootstrap`: resampling loop for metric confidence intervals.
- `debug.py` / `debugging.py` — `Debugger` and `PerformanceProfiler`, breakpoint/watch and timing/memory profiling utilities.
- `profiling.py` — `PerformanceMetrics` and detailed profiling/monitoring with optimization recommendations.
- `optimization.py` — gradient checkpointing and mixed-precision optimization helpers.
- `async_processing.py` — `AsyncIO` and concurrent/batch I/O utilities.
- `distributed.py` — `DistributedTraining`: multi-GPU/multi-node (DDP) helpers.
- `corruption_validation.py` — deprecated compatibility shim re-exporting `corruption.validators`.

## Public API / entry points

No package-level `__all__`; import symbols directly (e.g. `from circuitkit.utils.device import get_device`, `from circuitkit.utils.logging import get_logger`).

## How it fits

Imported by nearly every other subpackage (api, cli, backends, applications, corruption).
