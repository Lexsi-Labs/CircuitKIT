# Binary align

This task probes model alignment/refusal behavior. It uses length-matched pairs of structurally identical benign and harmful requests, optionally framed by safe vs. jailbreak system prompts.

## Key modules

- `generate_binary_align.py` — builds the request-pair index, samples benign/harmful pairs across several confound-controlled types (lexical homonyms, target swaps, context swaps), and writes CSV datasets. Runs as a CLI (`argparse`).
- `safe_binary.csv` — generated safe/benign prompt dataset.
- `jailbreak_binary.csv` — generated harmful/jailbreak prompt dataset.

## Public API / entry points

- `sample_pairs(...)` — sample matched benign/harmful request pairs.
- `generate_csv(...)` — emit the CSV datasets.
- Command-line entry via `_build_parser()` / `argparse`.

## How it fits

Registered in the task registry. Supplies paired safe/jailbreak data for discovery and evaluation of alignment and refusal circuits.
