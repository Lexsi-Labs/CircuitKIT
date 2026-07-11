"""
Faithfulness Evaluation Pillars

Implements the 6-pillar faithfulness framework for evaluating circuit quality:

1. Causal Patching: Does the circuit explain model behavior?
2. Ablation: Does the circuit support learned behavior?
3. Stability: Is the circuit robust to distribution shift?
4. Robustness: Does the circuit withstand input corruptions?
5. Baseline Comparison: How does the circuit compare to baselines?
6. Generalization: Does the circuit transfer to related tasks?
"""

from .ablation import Pillar2_Ablation
from .baselines import Pillar5_Baselines
from .causal_patching import Pillar1_CausalPatching
from .generalization import Pillar6_Generalization
from .robustness import Pillar4_Robustness
from .stability import Pillar3_Stability

__all__ = [
    "Pillar1_CausalPatching",
    "Pillar2_Ablation",
    "Pillar3_Stability",
    "Pillar4_Robustness",
    "Pillar5_Baselines",
    "Pillar6_Generalization",
]
