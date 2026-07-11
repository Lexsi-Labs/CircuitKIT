"""resample — Zhang & Nanda 2023 resample-ablation strategy.

Pair each clean prompt with a *different* clean prompt drawn from the
same dataset as the corrupt counterfactual. The corrupt prompt is
in-distribution (it's a real example) but unrelated to the clean one,
so attribution measures specifically what differs between two valid
real inputs. This is the "resample ablation" baseline recommended by
Best-Practices-of-Activation-Patching ([arxiv:2309.16042]).

Length contract: UNKNOWN — two unrelated real prompts can have any
relative length.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, NormalizedDataset
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy


@register_strategy("resample")
class Resample(CorruptionStrategy):
    description = (
        "Pair each clean prompt with a different clean prompt from the "
        "same dataset (Zhang & Nanda 2023 resample ablation)."
    )
    length_contract = LengthContract.UNKNOWN

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        pool: Optional[List[ContrastiveRecord]] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        if not pool:
            return CorruptionResult(
                None,
                None,
                notes="resample requires `pool` kwarg with peer records",
                succeeded=False,
            )
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        candidates = [r for r in pool if r.record_id != record.record_id]
        if not candidates:
            return CorruptionResult(None, None, notes="empty pool", succeeded=False)
        peer = rng.choice(candidates)
        return CorruptionResult(
            corrupt_prompt=peer.clean_prompt,
            corrupt_answer=peer.clean_answer,
            notes=f"resampled from peer {peer.record_id}",
            succeeded=True,
        )

    def apply_to_dataset(
        self,
        ds: NormalizedDataset,
        rng: Optional[random.Random] = None,
    ) -> NormalizedDataset:
        """Convenience: apply resample to every record using the dataset itself as pool."""
        rng = rng or random.Random(0)
        new_records = []
        for r in ds.records:
            new_records.append(self.apply(r, rng=rng, pool=ds.records))
        return NormalizedDataset(
            name=ds.name,
            shape=ds.shape,
            records=new_records,
            source=ds.source,
            schema_version=ds.schema_version,
            meta={**ds.meta, "_corruption": self.name},
        )
