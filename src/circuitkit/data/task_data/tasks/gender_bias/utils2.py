"""
Gender Bias Task Data Generation and Utilities
Occupation→pronoun coreference structure (Winogender schema, Rudinger et al. 2018)
"""

import random
from typing import Optional

import pandas as pd

from .....utils.logging import get_logger

logger = get_logger(__name__)

# Occupations with strong, documented gender stereotypes in LM training corpora.
# Sourced from the Winogender schema (Rudinger et al., 2018).
FEMALE_OCCUPATIONS = [
    "nurse",
    "midwife",
    "receptionist",
    "librarian",
    "secretary",
    "housekeeper",
    "hairdresser",
    "bookkeeper",
    "teacher",
    "babysitter",
    "dietitian",
    "paralegal",
    "counselor",
    "dental hygienist",
    "clerk",
]

MALE_OCCUPATIONS = [
    "surgeon",
    "engineer",
    "pilot",
    "lawyer",
    "electrician",
    "mechanic",
    "firefighter",
    "architect",
    "judge",
    "sheriff",
    "manager",
    "physicist",
    "carpenter",
    "guard",
    "driver",
]

# Templates end naturally before a subject pronoun.
# Occupation token is the only gender cue — no names or pronouns in context.
TEMPLATES = [
    "The {occupation} said that",
    "The {occupation} reported that",
    "The {occupation} confirmed that",
    "The {occupation} indicated that",
    "The {occupation} mentioned that",
]

def _resolve_pronoun_token(model, pronoun: str) -> int:
    """
    Resolve a pronoun to a single token ID using the model's tokenizer.
    Raises a clear ValueError if the pronoun is not a single token in this
    model's vocabulary, rather than failing silently with a wrong ID.
    """
    tokens = model.to_tokens(pronoun, prepend_bos=False).squeeze(0)
    if tokens.shape[0] != 1:
        raise ValueError(
            f"Pronoun '{pronoun}' tokenises to {tokens.shape[0]} tokens in "
            f"model '{getattr(model.cfg, 'model_name', 'unknown')}'. "
            f"Gender bias task requires both pronouns to be single tokens."
        )
    return tokens[0].item()

def generate_gender_bias_data(
    n_samples: int = 128, output_path: Optional[str] = None, seed: int = 42, model=None
) -> pd.DataFrame:
    """
    Generate gender bias data using the occupation→pronoun coreference structure.

    Task definition (Winogender schema):
        clean     : "The {occupation} said that"   (ends before pronoun)
        corrupted : same template, opposite-gender occupation (counterfactual)
        correct   : gender-stereotyped pronoun token ID (" She" / " He")
        incorrect : opposite pronoun token ID (" He" / " She")

    Metric: logit(correct) - logit(incorrect) at the last token position.
    A positive score means the model correctly gender-codes the occupation.

    Dataset is balanced: n_samples//2 female-stereotyped, remainder male-stereotyped,
    then shuffled so the circuit must learn the bidirectional association.
    """
    if model is None:
        raise ValueError("Model required for gender bias data generation. No default model.")

    random.seed(seed)

    # Resolve once — fail fast with a clear error if model vocabulary is incompatible
    she_id = _resolve_pronoun_token(model, " She")
    he_id = _resolve_pronoun_token(model, " He")

    data = []
    n_female = n_samples // 2
    n_male = n_samples - n_female  # handles odd n_samples correctly

    for _ in range(n_female):
        template = random.choice(TEMPLATES)
        female_occ = random.choice(FEMALE_OCCUPATIONS)
        male_occ = random.choice(MALE_OCCUPATIONS)
        data.append(
            {
                "clean": template.format(occupation=female_occ),
                "corrupted": template.format(occupation=male_occ),
                "correct_idx": she_id,
                "incorrect_idx": he_id,
            }
        )

    for _ in range(n_male):
        template = random.choice(TEMPLATES)
        female_occ = random.choice(FEMALE_OCCUPATIONS)
        male_occ = random.choice(MALE_OCCUPATIONS)
        data.append(
            {
                "clean": template.format(occupation=male_occ),
                "corrupted": template.format(occupation=female_occ),
                "correct_idx": he_id,
                "incorrect_idx": she_id,
            }
        )

    random.shuffle(data)
    df = pd.DataFrame(data)

    if output_path:
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} gender bias examples to {output_path}")

    return df
