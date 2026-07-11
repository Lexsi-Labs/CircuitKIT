"""TemplateStrategy — builds contrastive pairs from user-defined placeholder templates.

Supports two pairing modes:
  explicit   — the CSV row contains all column values for both clean and
               corrupt sides (e.g. country, capital, other_country, other_capital).
  auto_peer  — the CSV row contains only clean-side columns; the corrupt side
               is sourced from a randomly chosen peer record's values mapped
               via the other_* convention.
"""

from __future__ import annotations

import random
from typing import Set, Any, Dict, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, NormalizedDataset, DatasetShape
from .base import (
    CorruptionResult,
    CorruptionStrategy,
    LengthContract,
    register_strategy,
)
from .template_utils import (
    build_pair_from_templates,
    detect_peer_columns,
    parse_placeholders,
    validate_placeholders_against_columns,
)

_REQUIRED_SPEC_KEYS = {"clean_prompt", "corrupt_prompt", "clean_answer", "corrupt_answer"}


@register_strategy("template")
class TemplateStrategy(CorruptionStrategy):
    """Build clean/corrupt pairs via template substitution on CSV column values."""

    name = "template"
    description = (
        "Template-based clean/corrupt pairing using user-defined placeholder templates."
    )
    length_contract = LengthContract.PRESERVE  # user controls both sides

    def __init__(
        self,
        template_spec: Optional[Dict[str, str]] = None,
        pairing_mode: str = "explicit",
    ) -> None:
        """
        Args:
            template_spec: dict with keys clean_prompt, corrupt_prompt,
                           clean_answer, corrupt_answer.
            pairing_mode:  "explicit" or "auto_peer".
        """
        if template_spec is None:
            # Allow zero-arg construction required by the strategy registry
            # (get_strategy("template")()). Real usage must supply spec.
            self._spec: Dict[str, str] = {}
            self._all_placeholders: List[str] = []
            self._peer_map: Dict[str, str] = {}
            self._pairing_mode = pairing_mode
            return

        missing = _REQUIRED_SPEC_KEYS - set(template_spec.keys())
        if missing:
            raise ValueError(
                f"template_spec is missing required keys: {sorted(missing)}"
            )
        if pairing_mode not in ("explicit", "auto_peer"):
            raise ValueError(
                f"pairing_mode must be 'explicit' or 'auto_peer', got {pairing_mode!r}"
            )

        self._spec = dict(template_spec)
        self._pairing_mode = pairing_mode

        # Pre-parse placeholders and peer map at construction time.
        all_pholds: set[str] = set()
        for tmpl in self._spec.values():
            all_pholds.update(parse_placeholders(tmpl))
        self._all_placeholders = sorted(all_pholds)
        self._peer_map, _ = detect_peer_columns(self._all_placeholders)

    # ------------------------------------------------------------------
    # CorruptionStrategy interface
    # ------------------------------------------------------------------

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        pool: Optional[List[ContrastiveRecord]] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> CorruptionResult:
        """Generate corrupt prompt/answer by resolving templates against row values.

        Expects ``record.meta["_template_values"]`` to contain the raw CSV row
        as a dict of column → value strings.
        """
        if not self._spec:
            return CorruptionResult(
                corrupt_prompt=None,
                corrupt_answer=None,
                notes="TemplateStrategy constructed without a template_spec.",
                succeeded=False,
            )

        row: Optional[Dict[str, str]] = record.meta.get("_template_values")
        if row is None:
            return CorruptionResult(
                corrupt_prompt=None,
                corrupt_answer=None,
                notes="record.meta['_template_values'] is missing; cannot apply template.",
                succeeded=False,
            )

        try:
            if self._pairing_mode == "explicit":
                _, corrupt_prompt, _, corrupt_answer = build_pair_from_templates(
                    self._spec, row, row
                )

            else:  # auto_peer
                if pool is None:
                    return CorruptionResult(
                        corrupt_prompt=None,
                        corrupt_answer=None,
                        notes="auto_peer mode requires pool; none supplied.",
                        succeeded=False,
                    )
                rng = rng or random.Random()
                candidates = [
                    r for r in pool if r.record_id != record.record_id
                ]
                if not candidates:
                    return CorruptionResult(
                        corrupt_prompt=None,
                        corrupt_answer=None,
                        notes="auto_peer: no candidate peers found in pool.",
                        succeeded=False,
                    )
                peer = rng.choice(candidates)
                peer_row: Dict[str, str] = peer.meta.get("_template_values", {})
                # Build corrupt_values: remap other_X → peer's X value
                corrupt_values = dict(row)
                for other_key, base_key in self._peer_map.items():
                    if base_key in peer_row:
                        corrupt_values[other_key] = peer_row[base_key]

                _, corrupt_prompt, _, corrupt_answer = build_pair_from_templates(
                    self._spec, row, corrupt_values
                )

        except (KeyError, ValueError) as exc:
            return CorruptionResult(
                corrupt_prompt=None,
                corrupt_answer=None,
                notes=str(exc),
                succeeded=False,
            )

        return CorruptionResult(
            corrupt_prompt=corrupt_prompt,
            corrupt_answer=corrupt_answer,
        )

    def apply_to_dataset(
        self,
        ds: NormalizedDataset,
        *,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> NormalizedDataset:
        """Apply template strategy to every record, passing pool for auto_peer mode."""
        pool = ds.records if self._pairing_mode == "auto_peer" else None
        rng = rng or random.Random()

        new_records = [
            self.apply(r, pool=pool, rng=rng) for r in ds.records
        ]
        return NormalizedDataset(
            name=ds.name,
            shape=DatasetShape.TEMPLATE,
            records=new_records,
            source=ds.source,
            schema_version=ds.schema_version,
            meta={**ds.meta, "_corruption": "template"},
        )

    def validate_against_columns(self, columns: List[str]) -> List[str]:
        """Return placeholder names not satisfied by the given column list."""
        return validate_placeholders_against_columns(self._spec, columns)


__all__ = ["TemplateStrategy"]
