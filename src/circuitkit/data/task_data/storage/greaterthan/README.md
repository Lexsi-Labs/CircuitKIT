# greaterthan

Cached generated data for the `greaterthan` ACDC task.

## Data files

- `greaterthan_32_ffd33106.json` — metadata sidecar for a 32-example run
  (`config.task_name = "greaterthan"`). Keys: `num_samples`, `data_types`,
  `tensor_shapes`, `memory_usage`, `generation_time`, `config_hash`, `config`.

At runtime a matching `greaterthan_32_ffd33106.pkl` holds the actual generated
tensors/prompts; only the `.json` sidecar is committed to the repo.

## How it fits

Written and read by `task_data/generation` (`ACDCCache` / `FileManager`) so the
greaterthan task data is generated once and reused across discovery runs.
