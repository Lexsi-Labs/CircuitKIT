"""
DoubleIO Dataset Generator

Generates DoubleIO prompts where BOTH the subject (S) and indirect object (IO)
names appear twice, challenging the IOI circuit's duplicate-token-inhibition algorithm.

Standard IOI:
  "When John and Mary went to the store, John gave a drink to ___"
  → S appears twice (John, John), IO appears once (Mary) → predict Mary

DoubleIO:
  "When John and Mary went to the store, Mary was happy. John gave a drink to ___"
  → S appears twice (John, John), IO ALSO appears twice (Mary, Mary) → predict Mary
  → The "remove duplicates" algorithm should FAIL here, yet GPT-2 succeeds (~89%)

This module reuses the IOI infrastructure (NAMES, PLACES, OBJECTS, gen_prompt_uniform)
but defines its own template set that inserts a clause making IO appear a second time.

Reference: "Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability"
           (arxiv 2411.16105, Dec 2024)
"""

import copy
import random
from typing import Dict, List

import numpy as np
import torch

# ── Reuse IOI's name/noun pools ──────────────────────────────────────────────
# These are imported from the IOI dataset module at registration time.
# We duplicate them here so the file is self-contained for review/testing.

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

# ── Filler clauses that naturally reintroduce the IO name ────────────────────
# Each clause is inserted between the first and second sentence of an IOI
# template. [IO] is replaced with the indirect-object name at generation time.
# These are designed to be semantically neutral — they add a second mention
# of IO without providing any new task-relevant signal.

IO_FILLER_CLAUSES = [
    "[IO] was happy.",
    "[IO] smiled.",
    "[IO] looked around.",
    "[IO] sat down.",
    "[IO] waited patiently.",
    "[IO] was excited.",
    "[IO] felt cheerful.",
    "[IO] nodded.",
]

# ── DoubleIO BABA Templates ─────────────────────────────────────────────────
# Structure: "[clause_1_with_B_and_A]. [IO_filler]. [clause_2_with_B_giving_to_A]"
# In BABA order: B (subject) appears first, then A (IO).
# The filler clause adds a second mention of A (IO), making both names appear 2x.

DOUBLE_IO_BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [IO_FILLER] [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [IO_FILLER] [B] wanted to give a [OBJECT] to [A]",
    "After [B] and [A] went to the [PLACE], [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [IO_FILLER] [B] decided to give it to [A]",
    "While [B] and [A] were working at the [PLACE], [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [IO_FILLER] [B] gave a [OBJECT] to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [IO_FILLER] [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [IO_FILLER] [B] gave it to [A]",
]

# ── DoubleIO ABBA Templates (swap first-clause name order) ──────────────────
# In ABBA order: A (IO) appears first in the opening clause, then B (subject).

DOUBLE_IO_ABBA_TEMPLATES = []
for _tmpl in DOUBLE_IO_BABA_TEMPLATES:
    _new = _tmpl
    # Swap [A] and [B] in the FIRST clause only (before [IO_FILLER])
    filler_pos = _new.index("[IO_FILLER]")
    first_half = _new[:filler_pos]
    second_half = _new[filler_pos:]
    # Swap A↔B in first half
    first_half = first_half.replace("[A]", "[__TEMP__]")
    first_half = first_half.replace("[B]", "[A]")
    first_half = first_half.replace("[__TEMP__]", "[B]")
    DOUBLE_IO_ABBA_TEMPLATES.append(first_half + second_half)


def gen_double_io_prompts(
    templates: List[str],
    names: List[str],
    nouns_dict: Dict[str, List[str]],
    filler_clauses: List[str],
    N: int,
    symmetric: bool = False,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate N DoubleIO prompts from the given templates.

    Each prompt dict has keys:
      - "text":  the full prompt string (final token = answer position)
      - "IO":    the indirect-object name (correct answer)
      - "S":     the subject name (incorrect answer)
      - "TEMPLATE_IDX": which template was used

    The [IO_FILLER] placeholder in each template is replaced with a randomly
    chosen filler clause, which itself has [IO] replaced with the IO name.
    """
    assert seed is not None
    random.seed(seed)

    prompts = []
    nb_gen = 0

    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)

        # Sample two distinct names
        name_io = ""
        name_s = ""
        while name_io == name_s:
            name_io = random.choice(names)
            name_s = random.choice(names)

        # Sample nouns
        nouns = {}
        for k in nouns_dict:
            nouns[k] = random.choice(nouns_dict[k])

        # Pick a filler clause and fill in the IO name
        filler = random.choice(filler_clauses)
        filler = filler.replace("[IO]", name_io)

        # Build the prompt
        prompt_text = temp
        prompt_text = prompt_text.replace("[IO_FILLER]", filler)
        prompt_text = prompt_text.replace("[A]", name_io)
        prompt_text = prompt_text.replace("[B]", name_s)
        for k, v in nouns.items():
            prompt_text = prompt_text.replace(k, v)

        prompts.append(
            {
                "text": prompt_text,
                "IO": name_io,
                "S": name_s,
                "TEMPLATE_IDX": temp_id,
                **{k: v for k, v in nouns.items()},
            }
        )
        nb_gen += 1

        # Optional symmetric pair (swap IO and S)
        if symmetric and nb_gen < N:
            sym_filler = filler.replace(name_io, name_s)
            sym_text = temp
            sym_text = sym_text.replace("[IO_FILLER]", sym_filler)
            sym_text = sym_text.replace("[A]", name_s)
            sym_text = sym_text.replace("[B]", name_io)
            for k, v in nouns.items():
                sym_text = sym_text.replace(k, v)
            prompts.append(
                {
                    "text": sym_text,
                    "IO": name_s,
                    "S": name_io,
                    "TEMPLATE_IDX": temp_id,
                    **{k: v for k, v in nouns.items()},
                }
            )
            nb_gen += 1

    return prompts


def gen_double_io_corrupted_prompts(
    prompts: List[Dict],
    names: List[str],
    seed: int = 42,
) -> List[Dict]:
    """
    Generate corrupted (patched) versions of DoubleIO prompts.

    Corruption strategy: replace the SECOND mention of S (S2) with a random
    name, mirroring the standard IOI corruption (S2 -> RAND). This breaks
    the duplicate-token signal that S2 inhibition heads rely on.

    Args:
        prompts: List of DoubleIO prompt dicts
        names: Pool of names to sample replacements from
        seed: Random seed

    Returns:
        List of corrupted prompt dicts (same structure as input)
    """
    assert seed is not None
    np.random.seed(seed)

    corrupted = []
    for prompt in prompts:
        p = copy.deepcopy(prompt)
        text_tokens = p["text"].split(" ")

        s_name = p["S"]
        # Find the LAST occurrence of S in the token list (= S2)
        s_positions = [i for i, tok in enumerate(text_tokens) if tok.rstrip(".,!?;:") == s_name]
        if len(s_positions) >= 2:
            s2_pos = s_positions[-1]
        elif len(s_positions) == 1:
            s2_pos = s_positions[0]
        else:
            # S not found as standalone token — skip corruption
            corrupted.append(p)
            continue

        # Pick a random replacement name
        rand_name = names[np.random.randint(len(names))]
        while rand_name == p["IO"] or rand_name == p["S"]:
            rand_name = names[np.random.randint(len(names))]

        # Preserve any trailing punctuation
        original_tok = text_tokens[s2_pos]
        trailing = ""
        if original_tok and original_tok[-1] in ".,!?;:":
            trailing = original_tok[-1]

        text_tokens[s2_pos] = rand_name + trailing
        p["text"] = " ".join(text_tokens)
        corrupted.append(p)

    return corrupted


def get_double_io_data_only(
    num_examples: int,
    device: str,
    model,
    seed: int = 42,
    prompt_type: str = "ABBA",
) -> Dict:
    """
    Generate DoubleIO data in the same format as IOI's get_ioi_data_only().

    Returns a dict with:
      - validation_data:        clean tokens  [N, seq_len-1]
      - validation_patch_data:  corrupted tokens [N, seq_len-1]
      - validation_labels:      correct answer token IDs [N]
      - validation_wrong_labels: incorrect answer token IDs [N]
      - end_idxs:               answer positions [N]

    Args:
        num_examples: Number of examples to generate
        device: Target device
        model: HookedTransformer model (required for tokenization)
        seed: Random seed
        prompt_type: "ABBA" or "BABA" (template ordering)
    """
    if model is None:
        raise ValueError("Model required for DoubleIO data generation.")

    # Select templates
    if prompt_type == "ABBA":
        templates = DOUBLE_IO_ABBA_TEMPLATES
    elif prompt_type == "BABA":
        templates = DOUBLE_IO_BABA_TEMPLATES
    elif prompt_type == "mixed":
        templates = DOUBLE_IO_BABA_TEMPLATES[:6] + DOUBLE_IO_ABBA_TEMPLATES[:6]
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    # Generate clean prompts
    prompts = gen_double_io_prompts(
        templates=templates,
        names=NAMES,
        nouns_dict={"[PLACE]": PLACES, "[OBJECT]": OBJECTS},
        filler_clauses=IO_FILLER_CLAUSES,
        N=num_examples,
        seed=seed,
    )

    # Generate corrupted prompts
    corrupted_prompts = gen_double_io_corrupted_prompts(prompts, NAMES, seed=seed + 1)

    # Tokenize
    clean_tokens_list = []
    corrupted_tokens_list = []
    io_tokens_list = []
    s_tokens_list = []

    for clean_p, corrupt_p in zip(prompts, corrupted_prompts):
        clean_toks = model.to_tokens(clean_p["text"], prepend_bos=False).squeeze(0)
        corrupt_toks = model.to_tokens(corrupt_p["text"], prepend_bos=False).squeeze(0)

        io_tok = model.to_tokens(f" {clean_p['IO']}", prepend_bos=False).squeeze(0)[0]
        s_tok = model.to_tokens(f" {clean_p['S']}", prepend_bos=False).squeeze(0)[0]

        clean_tokens_list.append(clean_toks)
        corrupted_tokens_list.append(corrupt_toks)
        io_tokens_list.append(io_tok)
        s_tokens_list.append(s_tok)

    # Pad to uniform length
    max_len = max(
        max(t.shape[0] for t in clean_tokens_list),
        max(t.shape[0] for t in corrupted_tokens_list),
    )
    pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0

    padded_clean = []
    padded_corrupt = []
    end_positions = []

    for clean_toks, corrupt_toks in zip(clean_tokens_list, corrupted_tokens_list):
        # Capture original length BEFORE padding
        original_clean_len = clean_toks.shape[0]

        # Pad clean
        if clean_toks.shape[0] < max_len:
            padding = torch.full(
                (max_len - clean_toks.shape[0],),
                pad_id,
                dtype=clean_toks.dtype,
                device=clean_toks.device,
            )
            clean_toks = torch.cat([clean_toks, padding])
        padded_clean.append(clean_toks)

        # Pad corrupted
        if corrupt_toks.shape[0] < max_len:
            padding = torch.full(
                (max_len - corrupt_toks.shape[0],),
                pad_id,
                dtype=corrupt_toks.dtype,
                device=corrupt_toks.device,
            )
            corrupt_toks = torch.cat([corrupt_toks, padding])
        padded_corrupt.append(corrupt_toks)

        end_positions.append(original_clean_len - 1)  # last real token position

    # Stack into tensors
    clean_batch = torch.stack(padded_clean).to(device)
    corrupt_batch = torch.stack(padded_corrupt).to(device)
    io_batch = torch.tensor(io_tokens_list, dtype=torch.long, device=device)
    s_batch = torch.tensor(s_tokens_list, dtype=torch.long, device=device)
    end_idxs = torch.tensor(end_positions, dtype=torch.long, device=device)

    # Remove the last token (label) from input — the model predicts it
    validation_data = clean_batch[:, :-1]
    validation_patch_data = corrupt_batch[:, :-1]
    validation_labels = io_batch
    end_idxs = torch.clamp(end_idxs - 1, min=0)

    return {
        "validation_data": validation_data,
        "validation_patch_data": validation_patch_data,
        "validation_labels": validation_labels,
        "validation_wrong_labels": s_batch,
        "end_idxs": end_idxs,
    }
