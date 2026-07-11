"""
Circuit evaluation module.

Entry points:
    evaluate_graph(model, graph, dataloader, metrics, intervention)
        → Core evaluation function. Use this directly.
        intervention='patching' → Pillar 1 (causal patching)
        intervention='zero'     → Pillar 2 (ablation)

    run_full_faithfulness(model, graph, dataloader, metric)
        → Runs all 6 pillars.

    Pillar1_CausalPatching, Pillar2_Ablation
        → Thin wrappers around evaluate_graph(). Prefer evaluate_graph() directly.
"""

from .checkpoint_benchmark import (
    compare_base_vs_intervened,
    export_and_benchmark,
    run_lm_eval,
    verify_checkpoint_weights,
)
from .evaluate import evaluate_baseline, evaluate_graph
from .full import run_full_faithfulness
from .hf_checkpoint import (
    benchmark_on_checkpoint,
    is_compressed_tensors_checkpoint,
    is_quantized_checkpoint,
    load_compressed_tensors_checkpoint,
    load_quantized_checkpoint,
    save_compressed_tensors_checkpoint,
    save_pruned_checkpoint,
    save_quantized_checkpoint,
)
from .intervention_faithfulness import IF, IFFold, IFResult
from .master_grid import MasterGrid, MasterGridCell
from .pillars import Pillar1_CausalPatching, Pillar2_Ablation
from .report import FaithfulnessReport
from .reports import (
    ComprehensiveEvaluationReport,
    RobustnessReport,
    StabilityReport,
    StabilityRobustnessReport,
)
from .transfer import TransferMatrix
from .transfer_analysis import TransferMatrixAnalyzer
from .transfer_visualizer import TransferMatrixVisualizer

__all__ = [
    "evaluate_graph",
    "evaluate_baseline",
    "FaithfulnessReport",
    "Pillar1_CausalPatching",
    "Pillar2_Ablation",
    "run_full_faithfulness",
    "StabilityReport",
    "RobustnessReport",
    "StabilityRobustnessReport",
    "ComprehensiveEvaluationReport",
    "TransferMatrix",
    "TransferMatrixVisualizer",
    "TransferMatrixAnalyzer",
    # EMNLP 2026, Section 7.10 (Pivot 2: Intervention-Faithfulness metric)
    "IF",
    "IFFold",
    "IFResult",
    # EMNLP 2026, Section 7.8 (Pivot 3: master 5x7 method x wrapper grid)
    "MasterGrid",
    "MasterGridCell",
    # HF checkpoint export + lm-eval (for post-intervention benchmark eval)
    "save_pruned_checkpoint",
    "save_quantized_checkpoint",
    "load_quantized_checkpoint",
    "is_quantized_checkpoint",
    "save_compressed_tensors_checkpoint",
    "load_compressed_tensors_checkpoint",
    "is_compressed_tensors_checkpoint",
    "benchmark_on_checkpoint",
    "export_and_benchmark",
    "compare_base_vs_intervened",
    "run_lm_eval",
    "verify_checkpoint_weights",
]
