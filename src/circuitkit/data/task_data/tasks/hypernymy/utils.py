"""
Hypernymy Task Data Generation and Utilities
Similar to IOI and Greater-Than ACDC implementations
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .....utils.logging import get_logger

logger = get_logger(__name__)

# Hypernymy data - (hyponym, hypernym) pairs
HYPERNYMY_PAIRS = [
    # Animals
    ("dog", "animal"),
    ("cat", "animal"),
    ("bird", "animal"),
    ("fish", "animal"),
    ("lion", "animal"),
    ("tiger", "animal"),
    ("elephant", "animal"),
    ("bear", "animal"),
    ("wolf", "animal"),
    ("fox", "animal"),
    ("rabbit", "animal"),
    ("mouse", "animal"),
    ("horse", "animal"),
    ("cow", "animal"),
    ("pig", "animal"),
    ("sheep", "animal"),
    ("chicken", "animal"),
    ("duck", "animal"),
    ("goose", "animal"),
    ("turkey", "animal"),
    # Vehicles
    ("car", "vehicle"),
    ("truck", "vehicle"),
    ("bus", "vehicle"),
    ("motorcycle", "vehicle"),
    ("bicycle", "vehicle"),
    ("train", "vehicle"),
    ("airplane", "vehicle"),
    ("helicopter", "vehicle"),
    ("boat", "vehicle"),
    ("ship", "vehicle"),
    ("submarine", "vehicle"),
    ("rocket", "vehicle"),
    # Fruits
    ("apple", "fruit"),
    ("banana", "fruit"),
    ("orange", "fruit"),
    ("grape", "fruit"),
    ("strawberry", "fruit"),
    ("blueberry", "fruit"),
    ("cherry", "fruit"),
    ("peach", "fruit"),
    ("pear", "fruit"),
    ("plum", "fruit"),
    ("lemon", "fruit"),
    ("lime", "fruit"),
    ("watermelon", "fruit"),
    ("pineapple", "fruit"),
    ("mango", "fruit"),
    ("kiwi", "fruit"),
    # Vegetables
    ("carrot", "vegetable"),
    ("broccoli", "vegetable"),
    ("spinach", "vegetable"),
    ("lettuce", "vegetable"),
    ("tomato", "vegetable"),
    ("potato", "vegetable"),
    ("onion", "vegetable"),
    ("garlic", "vegetable"),
    ("pepper", "vegetable"),
    ("cucumber", "vegetable"),
    ("celery", "vegetable"),
    ("cabbage", "vegetable"),
    ("corn", "vegetable"),
    ("bean", "vegetable"),
    ("pea", "vegetable"),
    ("asparagus", "vegetable"),
    # Colors
    ("red", "color"),
    ("blue", "color"),
    ("green", "color"),
    ("yellow", "color"),
    ("orange", "color"),
    ("purple", "color"),
    ("pink", "color"),
    ("brown", "color"),
    ("black", "color"),
    ("white", "color"),
    ("gray", "color"),
    ("silver", "color"),
    ("gold", "color"),
    ("maroon", "color"),
    ("navy", "color"),
    ("teal", "color"),
    # Professions
    ("doctor", "profession"),
    ("teacher", "profession"),
    ("lawyer", "profession"),
    ("engineer", "profession"),
    ("nurse", "profession"),
    ("pilot", "profession"),
    ("chef", "profession"),
    ("artist", "profession"),
    ("musician", "profession"),
    ("writer", "profession"),
    ("scientist", "profession"),
    ("researcher", "profession"),
    ("manager", "profession"),
    ("director", "profession"),
    ("analyst", "profession"),
    ("consultant", "profession"),
    # Sports
    ("football", "sport"),
    ("basketball", "sport"),
    ("baseball", "sport"),
    ("soccer", "sport"),
    ("tennis", "sport"),
    ("golf", "sport"),
    ("swimming", "sport"),
    ("running", "sport"),
    ("cycling", "sport"),
    ("boxing", "sport"),
    ("wrestling", "sport"),
    ("hockey", "sport"),
    ("volleyball", "sport"),
    ("badminton", "sport"),
    ("handball", "sport"),
    ("rugby", "sport"),
    # instruments
    ("piano", "instrument"),
    ("guitar", "instrument"),
    ("violin", "instrument"),
    ("drums", "instrument"),
    ("trumpet", "instrument"),
    ("saxophone", "instrument"),
    ("flute", "instrument"),
    ("clarinet", "instrument"),
    ("cello", "instrument"),
    ("bass", "instrument"),
    ("harp", "instrument"),
    ("organ", "instrument"),
    # Furniture
    ("chair", "furniture"),
    ("table", "furniture"),
    ("bed", "furniture"),
    ("sofa", "furniture"),
    ("desk", "furniture"),
    ("bookshelf", "furniture"),
    ("cabinet", "furniture"),
    ("dresser", "furniture"),
    ("nightstand", "furniture"),
    ("ottoman", "furniture"),
    ("bench", "furniture"),
    ("stool", "furniture"),
    # Tools
    ("hammer", "tool"),
    ("screwdriver", "tool"),
    ("wrench", "tool"),
    ("pliers", "tool"),
    ("saw", "tool"),
    ("drill", "tool"),
    ("knife", "tool"),
    ("scissors", "tool"),
    ("ruler", "tool"),
    ("level", "tool"),
    ("chisel", "tool"),
    ("file", "tool"),
    # Buildings
    ("house", "building"),
    ("apartment", "building"),
    ("office", "building"),
    ("school", "building"),
    ("hospital", "building"),
    ("church", "building"),
    ("library", "building"),
    ("museum", "building"),
    ("theater", "building"),
    ("stadium", "building"),
    ("hotel", "building"),
    ("restaurant", "building"),
    # Clothing
    ("shirt", "clothing"),
    ("pants", "clothing"),
    ("dress", "clothing"),
    ("skirt", "clothing"),
    ("jacket", "clothing"),
    ("coat", "clothing"),
    ("sweater", "clothing"),
    ("hat", "clothing"),
    ("shoes", "clothing"),
    ("boots", "clothing"),
    ("socks", "clothing"),
    ("gloves", "clothing"),
]

HYPERNYMY_TEMPLATES = [
    "A {hyponym} is a type of {hypernym}.",
    "{hyponym} is a kind of {hypernym}.",
    "{hyponym} belongs to the category of {hypernym}.",
    "{hyponym} is classified as a {hypernym}.",
    "{hyponym} is an example of a {hypernym}.",
    "The {hyponym} is a {hypernym}.",
    "{hyponym} falls under the category of {hypernym}.",
    "{hyponym} is a member of the {hypernym} class.",
    "{hyponym} is a subset of {hypernym}.",
    "{hyponym} is a specific type of {hypernym}.",
]

@dataclass
class HypernymyData:
    """Container for hypernymy task data"""

    text: str
    hyponym: str
    hypernym: str
    is_correct: bool
    question_type: str

class HypernymyDataset:
    def __init__(self, n_samples: int = 1000, seed: int = 42, model=None):
        self.n_samples = n_samples
        self.seed = seed
        self.model = model

        if model is None:
            raise ValueError("Model required for HypernymyDataset.")

        random.seed(seed)
        self.data = self._generate_data()

    def _generate_data(self) -> List[HypernymyData]:
        """
        Generates raw data objects.
        Stores the canonical pair info which will be formatted into sentences in to_dataframe.
        """
        data = []
        for _ in range(self.n_samples):
            # 1. Pick a valid Pair
            hyponym, hypernym = random.choice(HYPERNYMY_PAIRS)

            # Store the data. We construct the specific text in to_dataframe
            # to ensure we handle the 'answer included' requirement consistently.
            data.append(
                HypernymyData(
                    text="",  # Placeholder, actual text built in to_dataframe
                    hyponym=hyponym,
                    hypernym=hypernym,
                    is_correct=True,
                    question_type="hyponym_to_hypernym",
                )
            )
        return data

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts raw data to DataFrame with tokenized labels.
        CLEAN: prompt only ("A dog is a type of") — model predicts the hypernym.
        CORRUPTED: different hyponym from a different category, same prompt structure.
        """
        rows = []
        for item in self.data:
            # --- 1. CLEAN SAMPLE (prompt only, no answer) ---
            clean_text = f"A {item.hyponym} is a type of"

            # --- 2. CORRUPTED SAMPLE ---
            # Different hyponym from a different category entirely.
            # The circuit must retrieve different category information,
            # making this a proper counterfactual for EAP/IBCircuit.
            while True:
                cf_hyponym, cf_hypernym = random.choice(HYPERNYMY_PAIRS)
                if cf_hypernym != item.hypernym:
                    break

            corrupted_text = f"A {cf_hyponym} is a type of"

            # --- 3. TOKENIZATION & SAFETY ---
            # Leading space because tokenizers treat " animal" differently from "animal"
            clean_target_str = f" {item.hypernym}"
            corrupted_target_str = f" {cf_hypernym}"

            try:
                clean_token = self.model.to_single_token(clean_target_str)
                corrupted_token = self.model.to_single_token(corrupted_target_str)

                rows.append(
                    {
                        "clean": clean_text,
                        "corrupted": corrupted_text,
                        "correct_idx": clean_token,
                        "incorrect_idx": corrupted_token,
                        "hyponym": item.hyponym,
                        "hypernym": item.hypernym,
                        "is_correct": True,
                        "question_type": item.question_type,
                        "label": clean_token,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(rows)

    def save_to_csv(self, filepath: str):

        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.debug(f"Saved {len(df)} hypernymy examples to {filepath}")

def generate_hypernymy_data(
    n_samples: int = 1000, output_path: Optional[str] = None, seed: int = 42, model=None
) -> pd.DataFrame:

    dataset = HypernymyDataset(n_samples=n_samples, seed=seed, model=model)
    df = dataset.to_dataframe()
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} hypernymy examples to {output_path}")
    return df
