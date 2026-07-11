# Interpretability Tasks

Task-specific data generation and utilities (adapted from ACDC and related work) for circuit discovery and evaluation in circuitkit. Each subfolder holds the prompts, clean/corrupt pairs, model loaders, and metrics for one mechanistic-interpretability task.

## Task subfolders

- **binary_align** — safe vs. jailbreak binary request pairs (alignment/refusal probing).
- **capital_country** — factual recall of a country's capital city.
- **docstring** — predicting the next argument name in a Python docstring.
- **double_io** — IOI variant where both subject and indirect object appear twice.
- **gender_bias** — occupation-to-pronoun coreference / gender-stereotype probing.
- **greaterthan** — predicting a year strictly greater than a given start year.
- **hypernymy** — mapping a hyponym to its category (is-a) hypernym.
- **induction** — copying a repeated token (induction heads).
- **ioi** — Indirect Object Identification.
- **sva** — Subject-Verb Agreement (number contrast).
- **wmdp** — WMDP multiple-choice knowledge probing.

`__init__.py` re-exports the ACDC-derived tasks (`docstring`, `greaterthan`, `induction`, `ioi`).

## How it fits

These tasks back circuitkit's task registry. They supply the clean/corrupt datasets, models, and metrics that circuit-discovery and evaluation routines run against.
