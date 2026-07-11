# WMDP

Multiple-choice knowledge probing on the WMDP (Weapons of Mass Destruction Proxy) benchmark, for circuit discovery over hazardous-knowledge question answering.

## Key modules

- `wmdp_utils.py` — loads WMDP configs from the Hugging Face Hub (with a cache-format fix), formats multiple-choice prompts, builds clean/corrupt EAP data, checks model correctness, and defines the logit-diff metric.

## Public API / entry points

- `load_wmdp_dataset(...)` — load a WMDP config/split.
- `format_wmdp_prompt(...)` / `corrupt_wmdp_query(...)`: prompt formatting and corruption.
- `generate_wmdp_eap_data(...)` — clean/corrupt dataset builder for edge-attribution patching.
- `wmdp_logit_diff_metric()` — evaluation metric.

## How it fits

Registered in the task registry. Supplies WMDP multiple-choice data and metrics for circuit discovery and evaluation.
