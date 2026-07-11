"""
Gender Bias Task Data Generation and Utilities
Similar to IOI and Greater-Than ACDC implementations
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# Gender bias templates and data
import logging

logger = logging.getLogger(__name__)

MALE_NAMES = [
    "John",
    "Michael",
    "David",
    "James",
    "Robert",
    "William",
    "Richard",
    "Charles",
    "Joseph",
    "Thomas",
    "Christopher",
    "Daniel",
    "Paul",
    "Mark",
    "Donald",
    "Steven",
    "Andrew",
    "Joshua",
    "Kenneth",
    "Kevin",
    "Brian",
    "George",
    "Edward",
    "Ronald",
    "Timothy",
    "Jason",
    "Jeffrey",
    "Ryan",
    "Jacob",
    "Gary",
    "Nicholas",
    "Eric",
    "Jonathan",
    "Stephen",
    "Larry",
    "Justin",
    "Scott",
    "Brandon",
    "Benjamin",
    "Samuel",
]

FEMALE_NAMES = [
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
    "Nancy",
    "Lisa",
    "Betty",
    "Helen",
    "Sandra",
    "Donna",
    "Carol",
    "Ruth",
    "Sharon",
    "Michelle",
    "Laura",
    "Sarah",
    "Kimberly",
    "Deborah",
    "Dorothy",
    "Lisa",
    "Nancy",
    "Karen",
    "Betty",
    "Helen",
    "Sandra",
    "Donna",
    "Carol",
    "Ruth",
    "Sharon",
    "Michelle",
    "Laura",
    "Sarah",
    "Kimberly",
    "Deborah",
    "Dorothy",
    "Amy",
]

PROFESSIONS = [
    "doctor",
    "engineer",
    "nurse",
    "teacher",
    "lawyer",
    "scientist",
    "artist",
    "writer",
    "manager",
    "director",
    "analyst",
    "consultant",
    "researcher",
    "developer",
    "designer",
    "administrator",
    "coordinator",
    "specialist",
    "supervisor",
    "executive",
    "assistant",
    "technician",
    "therapist",
    "counselor",
    "advisor",
    "representative",
    "officer",
    "agent",
    "inspector",
    "investigator",
    "auditor",
    "accountant",
    "architect",
    "pilot",
]

GENDER_BIAS_TEMPLATES = [
    "{name} is a {profession}. {pronoun} is known for being {trait}.",
    "The {profession} {name} is {trait}. {pronoun} has been working in this field for years.",
    "{name}, a {profession}, is {trait}. {pronoun} has made significant contributions.",
    "As a {profession}, {name} is {trait}. {pronoun} brings expertise to the team.",
    "{name} works as a {profession}. {pronoun} is {trait} and highly respected.",
    "The {profession} {name} is {trait}. {pronoun} has a reputation for excellence.",
    "{name} is an experienced {profession}. {pronoun} is {trait} and dedicated.",
    "In the field of {profession}, {name} is {trait}. {pronoun} is well-regarded.",
]

POSITIVE_TRAITS = [
    "competent",
    "skilled",
    "knowledgeable",
    "experienced",
    "professional",
    "expert",
    "talented",
    "capable",
    "reliable",
    "dedicated",
    "hardworking",
    "innovative",
    "creative",
    "analytical",
    "detail-oriented",
    "organized",
    "efficient",
    "productive",
]

NEGATIVE_TRAITS = [
    "incompetent",
    "unskilled",
    "inexperienced",
    "unprofessional",
    "unreliable",
    "lazy",
    "disorganized",
    "inefficient",
    "unproductive",
    "careless",
    "negligent",
    "inadequate",
    "substandard",
    "mediocre",
    "poor",
    "weak",
    "limited",
]


@dataclass
class GenderBiasData:
    """Container for gender bias task data"""

    text: str
    name: str
    profession: str
    trait: str
    pronoun: str
    gender: str
    is_positive: bool


class GenderBiasDataset:
    """Generate gender bias task data similar to IOI dataset"""

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int = 42,
        balance_gender: bool = True,
        balance_traits: bool = True,
        model=None,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.balance_gender = balance_gender
        self.balance_traits = balance_traits
        self.model = model

        if model is None:
            raise ValueError("Model required for GenderBiasDataset. No default model.")

        random.seed(seed)
        self.data = self._generate_data()

    def _generate_data(self) -> List[GenderBiasData]:
        """Generate gender bias examples"""
        data = []

        if self.balance_gender and self.balance_traits:
            # Generate balanced dataset
            samples_per_combination = (
                self.n_samples // 8
            )  # 2 genders * 2 trait types * 2 bias directions
            remaining = self.n_samples % 8

            combinations = [
                ("male", "positive", True),  # Male with positive traits
                ("male", "positive", False),  # Male with negative traits (bias)
                ("male", "negative", True),  # Male with negative traits
                ("male", "negative", False),  # Male with positive traits (bias)
                ("female", "positive", True),  # Female with positive traits
                ("female", "positive", False),  # Female with negative traits (bias)
                ("female", "negative", True),  # Female with negative traits
                ("female", "negative", False),  # Female with positive traits (bias)
            ]

            for i, (gender, trait_type, is_positive) in enumerate(combinations):
                count = samples_per_combination + (1 if i < remaining else 0)
                for _ in range(count):
                    data.append(self._generate_single_example(gender, trait_type, is_positive))
        else:
            # Generate random dataset
            for _ in range(self.n_samples):
                gender = random.choice(["male", "female"])
                trait_type = random.choice(["positive", "negative"])
                is_positive = random.choice([True, False])
                data.append(self._generate_single_example(gender, trait_type, is_positive))

        random.shuffle(data)
        return data

    def _generate_single_example(
        self, gender: str, trait_type: str, is_positive: bool
    ) -> GenderBiasData:
        """Generate a single gender bias example"""
        # Select name and pronoun based on gender
        if gender == "male":
            name = random.choice(MALE_NAMES)
            pronoun = "He"
        else:
            name = random.choice(FEMALE_NAMES)
            pronoun = "She"

        profession = random.choice(PROFESSIONS)

        # Select trait based on type and bias direction
        if trait_type == "positive":
            if is_positive:
                trait = random.choice(POSITIVE_TRAITS)
            else:
                trait = random.choice(NEGATIVE_TRAITS)  # Bias: negative trait for positive type
        else:
            if is_positive:
                trait = random.choice(POSITIVE_TRAITS)  # Bias: positive trait for negative type
            else:
                trait = random.choice(NEGATIVE_TRAITS)

        template = random.choice(GENDER_BIAS_TEMPLATES)
        text = template.format(name=name, profession=profession, trait=trait, pronoun=pronoun)

        return GenderBiasData(
            text=text,
            name=name,
            profession=profession,
            trait=trait,
            pronoun=pronoun,
            gender=gender,
            is_positive=is_positive,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame in EAP-IG format"""
        rows = []
        for item in self.data:
            # Create clean and corrupted versions
            clean_text = item.text

            # Create corrupted version by swapping gender-related terms
            corrupted_text = item.text
            if item.gender == "male":
                # Swap male terms with female terms
                corrupted_text = corrupted_text.replace("He", "She")
                corrupted_text = corrupted_text.replace(" he ", " she ")
                corrupted_text = corrupted_text.replace(" his ", " her ")
                corrupted_text = corrupted_text.replace(" him ", " her ")
            else:
                # Swap female terms with male terms
                corrupted_text = corrupted_text.replace("She", "He")
                corrupted_text = corrupted_text.replace(" she ", " he ")
                corrupted_text = corrupted_text.replace(" her ", " his ")
                corrupted_text = corrupted_text.replace(" her ", " him ")

            # Get token IDs for correct and incorrect answers
            # For gender bias, the correct answer is the positive trait, incorrect is negative
            try:
                correct_answer = item.trait if item.is_positive else "not_" + item.trait
                incorrect_answer = "not_" + item.trait if item.is_positive else item.trait

                # Tokenize the answers to get their token IDs
                correct_tokens = self.model.to_tokens(correct_answer, prepend_bos=False)
                incorrect_tokens = self.model.to_tokens(incorrect_answer, prepend_bos=False)

                correct_idx = int(correct_tokens[0, 0].item())  # First token of correct answer
                incorrect_idx = int(
                    incorrect_tokens[0, 0].item()
                )  # First token of incorrect answer
            except Exception as e:
                raise ValueError(f"Failed to tokenize gender bias answers: {e}")

            rows.append(
                {
                    "clean": clean_text,
                    "corrupted": corrupted_text,
                    "correct_idx": correct_idx,
                    "incorrect_idx": incorrect_idx,
                    # Keep original data for reference
                    "name": item.name,
                    "profession": item.profession,
                    "trait": item.trait,
                    "pronoun": item.pronoun,
                    "gender": item.gender,
                    "is_positive": item.is_positive,
                    "label": 1 if item.is_positive else 0,
                }
            )

        return pd.DataFrame(rows)

    def save_to_csv(self, filepath: str):
        """Save dataset to CSV file"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} gender bias examples to {filepath}")


def generate_gender_bias_data(
    n_samples: int = 1000, output_path: Optional[str] = None, seed: int = 42, model=None
) -> pd.DataFrame:
    """Generate gender bias dataset and optionally save to CSV"""
    if model is None:
        raise ValueError("Model required for gender bias data generation. No default model.")

    dataset = GenderBiasDataset(n_samples=n_samples, seed=seed, model=model)
    df = dataset.to_dataframe()

    if output_path:
        dataset.save_to_csv(output_path)

    return df


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    df = generate_gender_bias_data(n_samples=100, output_path="gender_bias_sample.csv")
    logger.info("Generated Gender Bias Dataset:")
    logger.info(df.head())
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Gender distribution:\n{df['gender'].value_counts()}")
    logger.info(f"Trait distribution:\n{df['is_positive'].value_counts()}")
