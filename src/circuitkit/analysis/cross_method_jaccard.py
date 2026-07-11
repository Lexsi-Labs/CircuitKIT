"""Cross-method circuit comparison via top-K Jaccard.

Used by the EMNLP 2026 submission's Finding B (Section 5) to
quantify component-level disagreement across discovery methods
discovering circuits for the same task on the same model.

Example
-------
>>> from circuitkit.analysis.cross_method_jaccard import (
...     cross_method_jaccard,
... )
>>> # circuits: dict mapping method name to a CircuitScores artifact
>>> # (or a list of node names, or a dict node_name -> score)
>>> result = cross_method_jaccard(circuits, top_k=10)
>>> # result.matrix:   numpy-style nested list of pairwise Jaccards
>>> # result.top_nodes: dict method -> top_k node names
>>> # result.range:    (min, max) pairwise Jaccard over off-diagonal cells

The Jaccard at top-K is the standard descriptive statistic for the
"do methods agree on the circuit?" question reported in the EMNLP
paper, derived from the workshop result on GPT-2 IOI where the same
analysis showed Jaccard 0.11 to 1.00 across 8 EAP-family variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

CircuitLike = Union[Dict[str, float], List[str], object]


def _extract_top_k_nodes(circuit: CircuitLike, top_k: int) -> List[str]:
    """Coerce a circuit-like input into a top-K node-name list.

    Accepts:
      - dict[str, float]: node_name -> score; top-K by absolute score
      - list[str]: assumed to be a pre-ranked list; truncated to top-K
      - object with `.node_scores` attribute (e.g. CircuitScores): dict-like
    """
    if hasattr(circuit, "node_scores"):
        node_scores = circuit.node_scores
    else:
        node_scores = circuit
    if isinstance(node_scores, dict):
        ranked = sorted(node_scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        return [n for n, _ in ranked[:top_k]]
    if isinstance(node_scores, list):
        return list(node_scores[:top_k])
    raise TypeError(f"Unsupported circuit type for jaccard: {type(circuit)!r}")


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class CrossMethodJaccardResult:
    methods: List[str]
    top_k: int
    matrix: List[List[float]]
    top_nodes: Dict[str, List[str]]
    range: Tuple[float, float]
    n_pairs: int


def cross_method_jaccard(
    circuits: Dict[str, CircuitLike],
    top_k: int = 10,
) -> CrossMethodJaccardResult:
    """Compute the symmetric pairwise Jaccard matrix between method
    circuits at top-K nodes.

    Parameters
    ----------
    circuits : dict
        Maps method name to either a CircuitScores artifact, a dict of
        node_name to score, or a pre-ranked list of node names.
    top_k : int
        Truncation depth for each method's circuit. Default 10 to match
        the EMNLP paper's protocol.

    Returns
    -------
    CrossMethodJaccardResult
        Includes the pairwise Jaccard matrix, the top-K node sets per
        method, and the off-diagonal range (min, max).
    """
    methods = sorted(circuits.keys())
    n = len(methods)
    top_nodes = {m: _extract_top_k_nodes(circuits[m], top_k) for m in methods}
    matrix = [[1.0] * n for _ in range(n)]
    off_diag: List[float] = []
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i <= j:
                continue
            j_val = _jaccard(top_nodes[mi], top_nodes[mj])
            matrix[i][j] = j_val
            matrix[j][i] = j_val
            off_diag.append(j_val)
    rng = (min(off_diag), max(off_diag)) if off_diag else (0.0, 0.0)
    return CrossMethodJaccardResult(
        methods=methods,
        top_k=top_k,
        matrix=matrix,
        top_nodes=top_nodes,
        range=rng,
        n_pairs=len(off_diag),
    )


__all__ = ["cross_method_jaccard", "CrossMethodJaccardResult"]
