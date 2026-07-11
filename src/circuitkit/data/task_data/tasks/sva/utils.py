"""
SVA (Subject-Verb Agreement) Task Data Generation Utilities.
Number-contrastive structure: Linzen et al. (2016), Lakretz et al. (2021).
"""

import random
from typing import Optional

import pandas as pd

from .....utils.logging import get_logger

logger = get_logger(__name__)

# Singular/plural subject pairs. Chosen to be unambiguous and common.
SUBJECT_PAIRS = [
    ("dog", "dogs"),
    ("cat", "cats"),
    ("bird", "birds"),
    ("rabbit", "rabbits"),
    ("horse", "horses"),
    ("lion", "lions"),
    ("bear", "bears"),
    ("wolf", "wolves"),
    ("fox", "foxes"),
    ("deer", "deer"),  # same form — valid, model must rely on prior context
]

# (singular_verb, plural_verb) pairs with leading space for BPE tokenizers.
# All verified as single tokens in GPT-2's vocabulary.
VERB_PAIRS = [
    (" is", " are"),
    (" was", " were"),
    (" has", " have"),
]

# Templates end before the verb. {subject} is the only number cue.
TEMPLATES = [
    "The {subject}",
    "Usually the {subject}",
    "Often the {subject}",
    "Sometimes the {subject}",
]

def _resolve_verb_token(model, verb: str) -> int:
    """
    Resolve a verb string (with leading space) to a single token ID.
    Raises a clear ValueError if the verb is not a single token in this
    model's vocabulary, so mismatches surface immediately rather than
    producing silently wrong token IDs.
    """
    tokens = model.to_tokens(verb, prepend_bos=False).squeeze(0)
    if tokens.shape[0] != 1:
        raise ValueError(
            f"Verb '{verb}' tokenises to {tokens.shape[0]} tokens in "
            f"model '{getattr(model.cfg, 'model_name', 'unknown')}'. "
            f"SVA task requires all verbs to be single tokens."
        )
    return tokens[0].item()

def generate_sva_data(
    n_samples: int = 128,
    output_path: Optional[str] = None,
    seed: int = 42,
    model=None,
) -> pd.DataFrame:
    """
    Generate SVA data using a number-contrastive structure.

    Task definition (Linzen et al. 2016):
        clean     : "The {singular_subject}"   (context stops before verb)
        corrupted : same template, plural subject (number-contrastive patch)
        correct   : singular verb token ID (e.g. " runs")
        incorrect : plural verb token ID   (e.g. " run")

    Dataset is balanced: n_samples//2 singular-subject rows,
    remainder plural-subject rows, then shuffled.

    Both EAP and IBCircuit use the same CSV. EAP consumes all four columns;
    IBCircuit only consumes 'clean' and 'correct_idx'.
    """

    if model is None:
        raise ValueError("Model required for SVA data generation.")

    random.seed(seed)

    # Resolve and validate all verb tokens once — fail fast on incompatible models.
    resolved = []
    for sg_verb, pl_verb in VERB_PAIRS:
        sg_id = _resolve_verb_token(model, sg_verb)
        pl_id = _resolve_verb_token(model, pl_verb)
        resolved.append((sg_id, pl_id))

    data = []
    n_singular = n_samples // 2
    n_plural = n_samples - n_singular

    for _ in range(n_singular):
        template = random.choice(TEMPLATES)
        sg_subj, pl_subj = random.choice(SUBJECT_PAIRS)
        sg_id, pl_id = random.choice(resolved)
        data.append(
            {
                "clean": template.format(subject=sg_subj),
                "corrupted": template.format(subject=pl_subj),
                "correct_idx": sg_id,
                "incorrect_idx": pl_id,
            }
        )

    for _ in range(n_plural):
        template = random.choice(TEMPLATES)
        sg_subj, pl_subj = random.choice(SUBJECT_PAIRS)
        sg_id, pl_id = random.choice(resolved)
        data.append(
            {
                "clean": template.format(subject=pl_subj),
                "corrupted": template.format(subject=sg_subj),
                "correct_idx": pl_id,
                "incorrect_idx": sg_id,
            }
        )

    random.shuffle(data)
    df = pd.DataFrame(data)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} SVA examples to {output_path}")

    return df
