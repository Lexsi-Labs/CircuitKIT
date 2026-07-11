import random
from typing import Dict, List, Optional

import numpy as np
import torch

# Adapted from the ACDC repository, acdc/ioi/ioi_dataset.py:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery (MIT). See THIRD_PARTY_LICENSES.md.
# Originates from the IOI study (Wang et al., 2022, arXiv:2211.00593).

NAMES = [
    "Mary",
    "John",
    "Jennifer",
    "James",
    "Alice",
    "Robert",
    "Patricia",
    "Michael",
    "Linda",
    "William",
    "Elizabeth",
    "David",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
]

TEMPLATES = [
    "Then, [A] and [B] went to the store. [B] gave a gift to [A]",
    "After [A] and [B] went to the park, [B] gave a gift to [A]",
    "When [A] and [B] arrived at the station, [B] gave a gift to [A]",
]


def gen_prompt_uniform(
    templates: List[str],
    names: List[str],
    N: int,
    symmetric: bool,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)

    prompt_list = []
    N // 2 if symmetric else N

    # Ensure we generate exactly N prompts
    while len(prompt_list) < N:
        template = random.choice(templates)
        name1, name2 = random.sample(names, 2)

        # Create clean prompt
        clean_prompt = template.replace("[A]", name1).replace("[B]", name2)
        prompt_list.append(
            {
                "text": clean_prompt,
                "IO": name1,
                "S": name2,
                "TEMPLATE_IDX": templates.index(template),
            }
        )

        if symmetric:
            # Create symmetric prompt
            sym_prompt = template.replace("[A]", name2).replace("[B]", name1)
            prompt_list.append(
                {
                    "text": sym_prompt,
                    "IO": name2,
                    "S": name1,
                    "TEMPLATE_IDX": templates.index(template),
                }
            )

    return prompt_list[:N]


class IOIDataset:
    def __init__(
        self,
        prompt_type: str,
        N: int,
        tokenizer,
        prompts: Optional[List[Dict]] = None,
        prepend_bos: bool = True,
        seed: int = 0,
    ):
        assert prompt_type == "ABBA"
        self.prompt_type = prompt_type
        self.N = N
        self.tokenizer = tokenizer

        if prompts is None:
            # Symmetrically generate prompts for ABBA and BABA patterns
            self.ioi_prompts = gen_prompt_uniform(
                templates=TEMPLATES,
                names=NAMES,
                N=N,
                symmetric=True,
                seed=seed,
            )
        else:
            assert self.N == len(prompts)
            self.ioi_prompts = prompts

        self.sentences = [prompt["text"] for prompt in self.ioi_prompts]

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.ioi_prompts
        ]
        self.toks = torch.Tensor(self.tokenizer(texts, padding=True).input_ids).type(torch.int)

        self.io_tokenIDs = [
            self.tokenizer.encode(" " + prompt["IO"])[0] for prompt in self.ioi_prompts
        ]
        self.s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.ioi_prompts
        ]

    def gen_flipped_prompts(self, flip_key: str, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        flipped_prompts = []
        for prompt in self.ioi_prompts:
            new_prompt = dict(prompt)
            if flip_key == "S":  # Flip S to a random name
                rand_name = random.choice(NAMES)
                while rand_name == new_prompt["IO"] or rand_name == new_prompt["S"]:
                    rand_name = random.choice(NAMES)
                # Replace S in both occurrences
                new_prompt["text"] = new_prompt["text"].replace(new_prompt["S"], rand_name)
                new_prompt["S"] = rand_name

            elif flip_key == "IO":  # Flip IO to a random name
                rand_name = random.choice(NAMES)
                while rand_name == new_prompt["IO"] or rand_name == new_prompt["S"]:
                    rand_name = random.choice(NAMES)
                # Replace IO in both occurrences
                new_prompt["text"] = new_prompt["text"].replace(new_prompt["IO"], rand_name)
                new_prompt["IO"] = rand_name

            flipped_prompts.append(new_prompt)

        return IOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,  # Pass the generated prompts in
            seed=seed,
        )
