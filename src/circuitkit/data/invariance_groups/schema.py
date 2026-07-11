"""
Invariance-contract data structures for circuit evaluation.

An InvarianceGroup represents a base example paired with typed variants,
each annotated with an explicit InvarianceContract stating what must hold
across the transformation (label invariance, position invariance, etc.).

This extends the Contrast Sets / CheckList lineage from task evaluation to
circuit evaluation: instead of asking "does the model's prediction hold?", we
ask "does the discovered circuit hold?" — and the contract specifies which
properties the circuit should preserve.

Serialisation: every group and variant serialises to/from plain dicts so
groups can be written to HuggingFace datasets with standard Croissant schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class VariantType(str, Enum):
    """Typed transformation families for circuit evaluation."""

    BASE = "base"
    PARAPHRASE = "paraphrase"  # Semantic-preserving surface rewrite
    ENTITY_SWAP = "entity_swap"  # Named-entity substitution
    ROLE_SWAP = "role_swap"  # Participant/role reversal
    DISTRACTOR = "distractor"  # Irrelevant context insertion
    TOKEN_SWAP = "token_swap"  # Token-level substitution
    NEGATION = "negation"  # Logical negation
    VOICE_SWAP = "voice_swap"  # Active ↔ passive


@dataclass
class InvarianceContract:
    """
    Explicit contract for what must hold across a transformation.

    A circuit is said to satisfy the contract for a group if:
    - label_invariant=True → the circuit's answer on every variant
      matches the base answer
    - position_invariant=True → the circuit's critical token positions
      are the same across base and variant
    - length_matched=True → the variant prompt has the same token count
      as the base (required for position-wise patching)

    These fields are used by Pillar 4 (Robustness) to determine which
    failures are genuine circuit failures vs data-preparation artifacts.
    """

    label_invariant: bool = True
    position_invariant: bool = False
    length_matched: bool = True
    # Optional: which specific fields must be invariant (e.g. ['answer_token'])
    invariant_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvarianceContract":
        return cls(
            label_invariant=d.get("label_invariant", True),
            position_invariant=d.get("position_invariant", False),
            length_matched=d.get("length_matched", True),
            invariant_fields=d.get("invariant_fields", []),
        )


# Default contracts per variant type (can be overridden per task)
DEFAULT_CONTRACTS: Dict[VariantType, InvarianceContract] = {
    VariantType.BASE: InvarianceContract(
        label_invariant=True, position_invariant=True, length_matched=True
    ),
    VariantType.PARAPHRASE: InvarianceContract(
        label_invariant=True, position_invariant=False, length_matched=False
    ),
    VariantType.ENTITY_SWAP: InvarianceContract(
        label_invariant=True, position_invariant=True, length_matched=True
    ),
    VariantType.ROLE_SWAP: InvarianceContract(
        label_invariant=False, position_invariant=False, length_matched=True
    ),
    VariantType.DISTRACTOR: InvarianceContract(
        label_invariant=True, position_invariant=False, length_matched=False
    ),
    VariantType.TOKEN_SWAP: InvarianceContract(
        label_invariant=True, position_invariant=True, length_matched=True
    ),
    VariantType.NEGATION: InvarianceContract(
        label_invariant=False, position_invariant=False, length_matched=False
    ),
    VariantType.VOICE_SWAP: InvarianceContract(
        label_invariant=True, position_invariant=False, length_matched=False
    ),
}


@dataclass
class InvarianceVariant:
    """
    One transformed variant of a base example.

    Fields:
        variant_type: Which transformation produced this variant.
        prompt: The full input prompt string.
        answer: Expected output token/string.
        contract: Explicit invariance contract for this variant.
        transformation_params: Reproducibility metadata (entity substitution
            map, paraphrase model ID, etc.).
        qc_passed: Whether this variant passed label-invariance QC.
        length_delta: Token-count difference vs the base prompt
            (0 = length-matched, positive = variant is longer).
    """

    variant_type: VariantType
    prompt: str
    answer: str
    contract: InvarianceContract
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    qc_passed: bool = True
    length_delta: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["variant_type"] = self.variant_type.value
        d["contract"] = self.contract.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvarianceVariant":
        return cls(
            variant_type=VariantType(d["variant_type"]),
            prompt=d["prompt"],
            answer=d["answer"],
            contract=InvarianceContract.from_dict(d.get("contract", {})),
            transformation_params=d.get("transformation_params", {}),
            qc_passed=d.get("qc_passed", True),
            length_delta=d.get("length_delta", 0),
        )


@dataclass
class InvarianceGroup:
    """
    A base example paired with its typed, contracted variants.

    One row in the released dataset. The group_id is stable across dataset
    versions so leaderboard submissions can be matched back to groups.

    Usage:
        group = InvarianceGroup(
            group_id="ioi_g0042",
            task="ioi",
            base_prompt="...",
            base_answer="Mary",
            variants=[...],
        )
        # Serialise to HF row
        row = group.to_hf_row()
        # Or write to JSON
        group.to_json(path)
    """

    group_id: str
    task: str
    base_prompt: str
    base_answer: str
    variants: List[InvarianceVariant] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_variants_by_type(self, vtype: VariantType) -> List[InvarianceVariant]:
        return [v for v in self.variants if v.variant_type == vtype]

    def passed_qc(self) -> bool:
        return all(v.qc_passed for v in self.variants)

    def variant_types(self) -> List[VariantType]:
        return [v.variant_type for v in self.variants]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "task": self.task,
            "base_prompt": self.base_prompt,
            "base_answer": self.base_answer,
            "variants": [v.to_dict() for v in self.variants],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InvarianceGroup":
        return cls(
            group_id=d["group_id"],
            task=d["task"],
            base_prompt=d["base_prompt"],
            base_answer=d["base_answer"],
            variants=[InvarianceVariant.from_dict(v) for v in d.get("variants", [])],
            metadata=d.get("metadata", {}),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "InvarianceGroup":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_hf_rows(self) -> List[Dict[str, Any]]:
        """
        Flatten the group into one HuggingFace dataset row per variant
        (including the base). Each row is self-contained and can be
        written directly to a HF Dataset.

        Row schema:
            group_id, task, variant_type, prompt, answer,
            label_invariant, position_invariant, length_matched,
            qc_passed, length_delta, transformation_params (JSON str),
            base_prompt (repeated for diff computation).
        """
        rows = []
        base_row = {
            "group_id": self.group_id,
            "task": self.task,
            "variant_type": VariantType.BASE.value,
            "prompt": self.base_prompt,
            "answer": self.base_answer,
            "label_invariant": True,
            "position_invariant": True,
            "length_matched": True,
            "qc_passed": True,
            "length_delta": 0,
            "transformation_params": "{}",
            "base_prompt": self.base_prompt,
        }
        rows.append(base_row)

        for v in self.variants:
            rows.append(
                {
                    "group_id": self.group_id,
                    "task": self.task,
                    "variant_type": v.variant_type.value,
                    "prompt": v.prompt,
                    "answer": v.answer,
                    "label_invariant": v.contract.label_invariant,
                    "position_invariant": v.contract.position_invariant,
                    "length_matched": v.contract.length_matched,
                    "qc_passed": v.qc_passed,
                    "length_delta": v.length_delta,
                    "transformation_params": json.dumps(v.transformation_params),
                    "base_prompt": self.base_prompt,
                }
            )

        return rows


def new_group_id(task: str, index: int) -> str:
    """Generate a stable, human-readable group ID."""
    return f"{task}_g{index:05d}"
