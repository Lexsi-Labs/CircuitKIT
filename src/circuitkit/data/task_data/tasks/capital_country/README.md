# Capital-country

A factual-recall task: given a prompt ending "The capital of {country} is", the model must predict the correct capital city as the next token.

## Key modules

- `utils.py` — country/capital pairs, prompt construction, and clean/corrupt dataset generation. Each clean prompt ("The capital of Austria is") is paired with a corrupt prompt that flips the country to a length-matched different one, so the logit diff at the prediction position stays non-degenerate and differentiable.
- `__init__.py` — re-exports the utils API.

## Public API / entry points

- `CapitalCountryData` (dataclass) and `CapitalCountryDataset`: data containers.
- `generate_capital_country_data(...)` — dataset builder.
- `CAPITAL_COUNTRY_PAIRS` — the (capital, country) source list.

## How it fits

Registered in the task registry. Supplies paired factual-recall data for circuit discovery and evaluation.
