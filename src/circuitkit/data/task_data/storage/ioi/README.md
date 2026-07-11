# ioi

Cached generated data for the IOI (Indirect Object Identification) ACDC task.

## Data files

Metadata sidecars, one per generation config (`config.task_name = "ioi"`,
`prompt_type = "ABBA"`), named `ioi_<num_examples>_<config_hash>.json`:

- `ioi_8_e87df42e.json` — 8 examples
- `ioi_16_8c879ddb.json` — 16 examples
- `ioi_32_a432ca4a.json` — 32 examples
- `ioi_64_3bba747e.json`, `ioi_64_f4a164db.json` — 64 examples (two configs)
- `ioi_500_1f7e7324.json` — 500 examples

Each sidecar carries `num_samples`, `data_types` (tokens, prompts, sentences,
word_idx, io_tokenIDs, s_tokenIDs), `tensor_shapes`, `memory_usage`,
`generation_time`, `config_hash`, and the full `config`. A matching `.pkl` holds
the generated tensors at runtime; only `.json` sidecars are committed.

## How it fits

Written and read by `task_data/generation` (`ACDCCache` / `FileManager`) so IOI
task data is generated once per config and reused across discovery runs.
