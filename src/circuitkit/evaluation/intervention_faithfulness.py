"""Intervention-Faithfulness (IF) metric.

Methodological contribution defined in the EMNLP 2026 submission
(Section 7.10, "An Intervention-Faithfulness metric"). Given a
circuit and a target intervention class, IF predicts whether that
circuit will produce a good downstream-intervention outcome, using
three cheap signals: Pillar-5 stability, Pillar-3 baseline ratio,
and a low-budget intervention probe.

Definition
----------
    IF(c, w) = alpha * J_5(c) + beta * R_3(c) + gamma * P(c, w)

subject to alpha + beta + gamma = 1 and alpha, beta, gamma >= 0.

The coefficients (alpha, beta, gamma) are fit by leave-one-wrapper-out
cross-validation across a grid of (method, wrapper) cells: each fold
fits the simplex on 6 of 7 wrappers and predicts the held-out
wrapper's intervention winner.

Usage
-----

>>> from circuitkit.evaluation.intervention_faithfulness import IF
>>>
>>> # cells: dict[(method, wrapper)] -> dict with keys
>>> #   {"stability": float, "baseline_ratio": float, "probe": float,
>>> #    "quality": float}
>>> if_metric = IF()
>>> result = if_metric.fit_loo(cells)
>>> print(result.if_accuracy, result.faith_accuracy)
>>> # IF coefficients per fold:
>>> for fold in result.folds:
...     print(fold.held_wrapper, fold.coeffs)

The probe value P(c, w) is computed by running the wrapper at a
reduced budget (smaller top-K, single seed, single batch); see
HYPOTHESES.md in the emnlp_experiments/ directory for non-leakage
constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class IFFold:
    held_wrapper: str
    coeffs: Tuple[float, float, float]
    true_winner: str
    pred_if: str
    pred_faithfulness_only: str
    pred_stability_only: str
    if_correct: bool
    faith_correct: bool
    stab_correct: bool


@dataclass
class IFResult:
    n_methods: int
    n_wrappers: int
    chance_top1: float
    if_accuracy: float
    faith_accuracy: float
    stab_accuracy: float
    folds: List[IFFold] = field(default_factory=list)


class IF:
    """Intervention-Faithfulness metric.

    Parameters
    ----------
    simplex_grid_step : float
        Step size for the coarse simplex search over (alpha, beta, gamma).
        Default 0.1, giving 66 candidate triples.
    """

    def __init__(self, simplex_grid_step: float = 0.1) -> None:
        self.simplex_grid_step = simplex_grid_step

    @staticmethod
    def _if_score(
        stab: float, ratio: float, probe: float, coeffs: Tuple[float, float, float]
    ) -> float:
        a, b, g = coeffs
        return a * stab + b * ratio + g * probe

    def _fit_simplex(self, train_cells, train_winners) -> Tuple[float, float, float]:
        """Coarse grid search over the simplex; minimizes the count
        of training-fold method mis-orderings."""
        step = self.simplex_grid_step
        steps = int(1 / step) + 1
        best, best_coeffs = None, (1 / 3, 1 / 3, 1 / 3)
        wrappers = sorted(train_winners.keys())
        methods = sorted({m for (m, _) in train_cells})

        for i in range(steps):
            for j in range(steps - i):
                k = steps - 1 - i - j
                if k < 0:
                    continue
                a, b, g = i * step, j * step, k * step
                err = 0
                for w in wrappers:
                    truth_winner = train_winners.get(w)
                    if truth_winner is None:
                        continue
                    scores = {
                        m: self._if_score(
                            train_cells[(m, w)]["stability"],
                            train_cells[(m, w)]["baseline_ratio"],
                            train_cells[(m, w)]["probe"],
                            (a, b, g),
                        )
                        for m in methods
                        if (m, w) in train_cells
                    }
                    pred = max(scores, key=scores.get)
                    if pred != truth_winner:
                        err += 1
                if best is None or err < best:
                    best, best_coeffs = err, (a, b, g)
        return best_coeffs

    def fit_loo(self, cells: Dict[Tuple[str, str], Dict[str, float]]) -> IFResult:
        """Leave-one-wrapper-out cross-validation across all wrappers."""
        methods = sorted({m for (m, _) in cells})
        wrappers = sorted({w for (_, w) in cells})

        # Per-wrapper ground-truth winners
        winners = {}
        for w in wrappers:
            scored = [(m, cells[(m, w)]["quality"]) for m in methods if (m, w) in cells]
            if scored:
                winners[w] = max(scored, key=lambda mv: mv[1])[0]

        folds: List[IFFold] = []
        for w_held in wrappers:
            train_cells = {(m, w): cells[(m, w)] for (m, w) in cells if w != w_held}
            train_winners = {w: winners[w] for w in winners if w != w_held}
            coeffs = self._fit_simplex(train_cells, train_winners)

            held_scores_if = {
                m: self._if_score(
                    cells[(m, w_held)]["stability"],
                    cells[(m, w_held)]["baseline_ratio"],
                    cells[(m, w_held)]["probe"],
                    coeffs,
                )
                for m in methods
                if (m, w_held) in cells
            }
            held_scores_faith = {
                m: cells[(m, w_held)].get("faithfulness", 0.0)
                for m in methods
                if (m, w_held) in cells
            }
            held_scores_stab = {
                m: cells[(m, w_held)].get("stability", 0.0) for m in methods if (m, w_held) in cells
            }

            pred_if = max(held_scores_if, key=held_scores_if.get)
            pred_faith = max(held_scores_faith, key=held_scores_faith.get)
            pred_stab = max(held_scores_stab, key=held_scores_stab.get)
            truth = winners.get(w_held, "")

            folds.append(
                IFFold(
                    held_wrapper=w_held,
                    coeffs=coeffs,
                    true_winner=truth,
                    pred_if=pred_if,
                    pred_faithfulness_only=pred_faith,
                    pred_stability_only=pred_stab,
                    if_correct=(pred_if == truth),
                    faith_correct=(pred_faith == truth),
                    stab_correct=(pred_stab == truth),
                )
            )

        n_methods = len(methods)
        n_wrappers = len(wrappers)
        return IFResult(
            n_methods=n_methods,
            n_wrappers=n_wrappers,
            chance_top1=1.0 / max(1, n_methods),
            if_accuracy=sum(f.if_correct for f in folds) / max(1, len(folds)),
            faith_accuracy=sum(f.faith_correct for f in folds) / max(1, len(folds)),
            stab_accuracy=sum(f.stab_correct for f in folds) / max(1, len(folds)),
            folds=folds,
        )


__all__ = ["IF", "IFFold", "IFResult"]
