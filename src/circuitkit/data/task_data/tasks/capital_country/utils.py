"""
Capital-Country Task Data Generation and Utilities

Redesigned (2026-05) to produce a *non-degenerate*, differentiable circuit
discovery signal. See module docstring of ``to_dataframe`` for the contract.

The earlier implementation had two fatal bugs:

1. ``incorrect_idx`` was set equal to ``correct_idx`` on every row, so the
   discovery metric ``logit(correct) - logit(incorrect)`` was identically 0
   with zero gradient -> EAP / EAP-IG produced all-zero, all-equal scores.
2. The prompt placed the answer entity *mid-sentence*
   ("Vienna is the capital of Austria"), but the metric scores the position
   ``input_length - 1`` (the position whose prediction is the *next* token).
   The scored position therefore had nothing to do with the answer.

Both are fixed here by mirroring how ``boolq.py`` / ``ioi.py`` build pairs:

* The prompt ends *immediately before* the answer token:
  ``"The capital of Austria is"`` -> the model predicts ``" Vienna"``.
  The answer token is now exactly at the prediction position
  (``input_length - 1``).
* Each clean prompt is paired with a corrupt prompt that *flips the country*
  to a length-matched different country, so the corrupt prompt has a
  *different* correct capital.  ``correct_idx`` is the clean capital and
  ``incorrect_idx`` is the corrupt prompt's capital -- a genuinely different,
  wrong token for the clean prompt.  The logit-diff is a real, non-degenerate,
  differentiable signal.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .....utils.logging import get_logger

logger = get_logger(__name__)

# Capital-Country data
CAPITAL_COUNTRY_PAIRS = [
    ("Paris", "France"),
    ("London", "United Kingdom"),
    ("Berlin", "Germany"),
    ("Madrid", "Spain"),
    ("Rome", "Italy"),
    ("Amsterdam", "Netherlands"),
    ("Brussels", "Belgium"),
    ("Vienna", "Austria"),
    ("Prague", "Czech Republic"),
    ("Warsaw", "Poland"),
    ("Budapest", "Hungary"),
    ("Bucharest", "Romania"),
    ("Sofia", "Bulgaria"),
    ("Zagreb", "Croatia"),
    ("Ljubljana", "Slovenia"),
    ("Bratislava", "Slovakia"),
    ("Vilnius", "Lithuania"),
    ("Riga", "Latvia"),
    ("Tallinn", "Estonia"),
    ("Helsinki", "Finland"),
    ("Stockholm", "Sweden"),
    ("Oslo", "Norway"),
    ("Copenhagen", "Denmark"),
    ("Reykjavik", "Iceland"),
    ("Dublin", "Ireland"),
    ("Lisbon", "Portugal"),
    ("Athens", "Greece"),
    ("Nicosia", "Cyprus"),
    ("Valletta", "Malta"),
    ("Luxembourg", "Luxembourg"),
    ("Monaco", "Monaco"),
    ("Vatican City", "Vatican City"),
    ("San Marino", "San Marino"),
    ("Andorra", "Andorra"),
    ("Liechtenstein", "Liechtenstein"),
    ("Washington", "United States"),
    ("Ottawa", "Canada"),
    ("Mexico City", "Mexico"),
    ("Brasilia", "Brazil"),
    ("Buenos Aires", "Argentina"),
    ("Santiago", "Chile"),
    ("Lima", "Peru"),
    ("Bogota", "Colombia"),
    ("Caracas", "Venezuela"),
    ("Quito", "Ecuador"),
    ("La Paz", "Bolivia"),
    ("Asuncion", "Paraguay"),
    ("Montevideo", "Uruguay"),
    ("Georgetown", "Guyana"),
    ("Paramaribo", "Suriname"),
    ("Cayenne", "French Guiana"),
    ("Tokyo", "Japan"),
    ("Beijing", "China"),
    ("Seoul", "South Korea"),
    ("Pyongyang", "North Korea"),
    ("Ulaanbaatar", "Mongolia"),
    ("Taipei", "Taiwan"),
    ("Hong Kong", "Hong Kong"),
    ("Macau", "Macau"),
    ("Manila", "Philippines"),
    ("Jakarta", "Indonesia"),
    ("Kuala Lumpur", "Malaysia"),
    ("Singapore", "Singapore"),
    ("Bangkok", "Thailand"),
    ("Hanoi", "Vietnam"),
    ("Phnom Penh", "Cambodia"),
    ("Vientiane", "Laos"),
    ("Yangon", "Myanmar"),
    ("Dhaka", "Bangladesh"),
    ("Kathmandu", "Nepal"),
    ("Thimphu", "Bhutan"),
    ("Colombo", "Sri Lanka"),
    ("Malé", "Maldives"),
    ("New Delhi", "India"),
    ("Islamabad", "Pakistan"),
    ("Kabul", "Afghanistan"),
    ("Tehran", "Iran"),
    ("Baghdad", "Iraq"),
    ("Damascus", "Syria"),
    ("Beirut", "Lebanon"),
    ("Amman", "Jordan"),
    ("Jerusalem", "Israel"),
    ("Ramallah", "Palestine"),
    ("Ankara", "Turkey"),
    ("Tbilisi", "Georgia"),
    ("Yerevan", "Armenia"),
    ("Baku", "Azerbaijan"),
    ("Moscow", "Russia"),
    ("Minsk", "Belarus"),
    ("Kiev", "Ukraine"),
    ("Chisinau", "Moldova"),
    ("Tashkent", "Uzbekistan"),
    ("Astana", "Kazakhstan"),
    ("Bishkek", "Kyrgyzstan"),
    ("Dushanbe", "Tajikistan"),
    ("Ashgabat", "Turkmenistan"),
    ("Cairo", "Egypt"),
    ("Tripoli", "Libya"),
    ("Tunis", "Tunisia"),
    ("Algiers", "Algeria"),
    ("Rabat", "Morocco"),
    ("Dakar", "Senegal"),
    ("Banjul", "Gambia"),
    ("Conakry", "Guinea"),
    ("Freetown", "Sierra Leone"),
    ("Monrovia", "Liberia"),
    ("Accra", "Ghana"),
    ("Lome", "Togo"),
    ("Niamey", "Niger"),
    ("Bamako", "Mali"),
    ("Asmara", "Eritrea"),
    ("Djibouti", "Djibouti"),
    ("Mogadishu", "Somalia"),
    ("Nairobi", "Kenya"),
    ("Kampala", "Uganda"),
    ("Kigali", "Rwanda"),
    ("Lusaka", "Zambia"),
    ("Harare", "Zimbabwe"),
    ("Windhoek", "Namibia"),
    ("Pretoria", "South Africa"),
    ("Maputo", "Mozambique"),
    ("Libreville", "Gabon"),
    ("Luanda", "Angola"),
    ("Canberra", "Australia"),
    ("Suva", "Fiji"),
]

# The model predicts the answer token immediately after this prompt. Keeping a
# single fixed template means clean and corrupt prompts differ *only* in the
# country, which is required for EAP token-length alignment.
CAPITAL_COUNTRY_PROMPT = "The capital of {country} is"

# Retained for backwards compatibility with any external callers / finetuning
# code that imported the old template list.
CAPITAL_COUNTRY_TEMPLATES = [
    "The capital of {country} is {capital}.",
    "{capital} is the capital city of {country}.",
]

@dataclass
class CapitalCountryData:
    """Container for a single capital-country example.

    ``text`` is the *prompt only* -- it ends right before the answer token,
    so the model's prediction at the final position is the answer.
    """

    text: str
    capital: str
    country: str
    is_correct: bool
    question_type: str  # always "capital_to_country" in the redesigned task

class CapitalCountryDataset:
    """Generate capital-country circuit-discovery data.

    The task is ``country -> capital`` recall framed so the answer is the
    *next token*: prompt ``"The capital of {country} is"`` -> ``" {capital}"``.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int = 42,
        balance_question_types: bool = True,
        balance_correctness: bool = True,
        model=None,
    ):
        self.n_samples = n_samples
        self.seed = seed
        # These flags are kept for signature compatibility; the redesigned
        # task is a single homogeneous direction so balancing is a no-op.
        self.balance_question_types = balance_question_types
        self.balance_correctness = balance_correctness
        self.model = model

        if model is None:
            raise ValueError("Model required for CapitalCountryDataset. No default model.")

        random.seed(seed)
        self._rng = random.Random(seed)

        # Restrict to pairs that are *fully single-token* in this model:
        #   * answer  ' {capital}' must be a single token (logit-diff metric)
        #   * country ' {country}' must be a single token (so clean/corrupt
        #     prompts are trivially length-aligned)
        self.usable_pairs = self._compute_usable_pairs()
        if len(self.usable_pairs) < 2:
            raise ValueError(
                "Capital-country task needs at least 2 single-token "
                f"capital/country pairs for model "
                f"'{getattr(model.cfg, 'model_name', 'unknown')}'; "
                f"found {len(self.usable_pairs)}."
            )

        self.data = self._generate_data()

    # ------------------------------------------------------------------
    # tokenization helpers
    # ------------------------------------------------------------------
    def _get_token_len(self, text: str) -> int:
        """Number of tokens for *text* (no BOS)."""
        return len(self.model.to_tokens(text, prepend_bos=False)[0])

    def _is_single_token(self, word: str) -> bool:
        """True if ' word' (leading-space form) is exactly one token."""
        return self._get_token_len(f" {word}") == 1

    def _compute_usable_pairs(self) -> List[tuple]:
        """Keep only pairs whose capital *and* country are single tokens."""
        usable = []
        for capital, country in CAPITAL_COUNTRY_PAIRS:
            if self._is_single_token(capital) and self._is_single_token(country):
                usable.append((capital, country))
        return usable

    # ------------------------------------------------------------------
    def _generate_data(self) -> List[CapitalCountryData]:
        """Generate ``n_samples`` clean examples (sampled with replacement
        if there are fewer usable pairs than requested)."""
        data: List[CapitalCountryData] = []
        for _ in range(self.n_samples):
            capital, country = self._rng.choice(self.usable_pairs)
            data.append(
                CapitalCountryData(
                    text=CAPITAL_COUNTRY_PROMPT.format(country=country),
                    capital=capital,
                    country=country,
                    is_correct=True,
                    question_type="capital_to_country",
                )
            )
        return data

    # ------------------------------------------------------------------
    def _pick_corrupt_pair(self, item: CapitalCountryData) -> Optional[tuple]:
        """Pick a *different* (capital, country) pair to act as the corrupt
        counterpart.

        Both the clean and corrupt countries are single tokens (guaranteed by
        ``usable_pairs``), so the clean and corrupt prompts always tokenize to
        the same length -- the invariant required by attribution patching.
        The corrupt pair must have a different capital so the corrupt prompt's
        correct answer (and therefore ``incorrect_idx``) is a genuinely
        different, wrong token for the clean prompt.
        """
        candidates = [
            (cap, cy) for cap, cy in self.usable_pairs if cap != item.capital and cy != item.country
        ]
        return self._rng.choice(candidates) if candidates else None

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a clean / corrupt DataFrame for EAP / EAP-IG.

        KEY INVARIANTS (required by attribution patching):

        1. ``tokenize(clean) .shape == tokenize(corrupted).shape``
           Guaranteed: both countries are single tokens and the template is
           fixed, so the two prompts differ only in one equal-length slot.

        2. ``correct_idx != incorrect_idx``  (non-degenerate logit-diff).
           ``correct_idx``   = clean prompt's capital token.
           ``incorrect_idx`` = corrupt prompt's capital token (the answer the
                               *corrupt* country would imply -- wrong for the
                               clean prompt).

        3. The answer token sits exactly at the prediction position. The
           prompt ends with " is", so the model's next-token prediction
           (scored at ``input_length - 1``) *is* the answer.

        Resulting design (capital_to_country recall, country -> capital):

            clean      ->  "The capital of {country}    is"   target ' {capital}'
            corrupted  ->  "The capital of {cf_country} is"   target ' {cf_capital}'

        EAP measures how much each node contributes to keeping
        logit(' {capital}') above logit(' {cf_capital}') as the country slot
        is patched from clean to corrupt.
        """
        rows = []
        for item in self.data:
            cf = self._pick_corrupt_pair(item)
            if cf is None:
                continue
            cf_capital, cf_country = cf

            clean_text = CAPITAL_COUNTRY_PROMPT.format(country=item.country)
            corrupted_text = CAPITAL_COUNTRY_PROMPT.format(country=cf_country)

            # Single-token answers (enforced -- usable_pairs already guarantees
            # this, but the metric contract makes it worth a hard check).
            try:
                correct_idx = self.model.to_single_token(f" {item.capital}")
                incorrect_idx = self.model.to_single_token(f" {cf_capital}")
            except Exception:
                continue
            if correct_idx == incorrect_idx:
                # Degenerate logit-diff -- skip (should not happen given the
                # distinct-capital filter, but guard the invariant anyway).
                continue

            # Hard length guard -- clean/corrupt MUST be token-length aligned.
            if self._get_token_len(clean_text) != self._get_token_len(corrupted_text):
                continue

            rows.append(
                {
                    "clean": clean_text,
                    "corrupted": corrupted_text,
                    "correct_idx": correct_idx,
                    "incorrect_idx": incorrect_idx,
                    "capital": item.capital,
                    "country": item.country,
                    "cf_capital": cf_capital,
                    "cf_country": cf_country,
                    "is_correct": True,
                    "question_type": item.question_type,
                    "label": correct_idx,
                }
            )

        return pd.DataFrame(rows)

    def save_to_csv(self, filepath: str):
        """Save dataset to CSV file."""

        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} capital-country examples to {filepath}")

def generate_capital_country_data(
    n_samples: int = 1000, output_path: Optional[str] = None, seed: int = 42, model=None
) -> pd.DataFrame:
    """Generate capital-country dataset and optionally save to CSV."""
    if model is None:
        raise ValueError("Model required for capital country data generation. No default model.")

    dataset = CapitalCountryDataset(n_samples=n_samples, seed=seed, model=model)
    df = dataset.to_dataframe()

    if output_path:
        dataset.save_to_csv(output_path)

    return df

# Example usage and testing
if __name__ == "__main__":

    df = generate_capital_country_data(n_samples=100, output_path="capital_country_sample.csv")
    logger.info("Generated Capital-Country Dataset:")
    logger.info(df.head())
    logger.debug(f"\nDataset shape: {df.shape}")
