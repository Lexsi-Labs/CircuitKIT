"""Hyperparameter ranges: one source of truth for what is *valid* and what is *sensible*.

Two tiers, borrowed from how mature libraries separate "wrong" from "unusual"
(scikit-learn's ``Interval`` constraints raise on invalid values; Hugging Face's
``GenerationConfig.validate`` logs "minor issues" and only raises on ``strict=True``):

* **Valid** – values outside this range can never work (``ig_steps=0``, a negative
  sparsity, an unknown pillar name). They raise :class:`HyperparameterError`, a
  ``ValueError``, with a message that names the parameter and the allowed range.
* **Sensible** – values inside the valid range that earlier runs or the literature
  show to be risky (very few examples, pruning past the point where models
  collapse). They emit a :class:`HyperparameterWarning` once per call site. Pass
  ``validation: {strict: true}`` in a dict/YAML config, or set the
  ``CIRCUITKIT_STRICT_HPARAMS=1`` environment variable, to turn warnings into errors.

Only values the caller set explicitly are warned about, so a default never triggers a
warning. Only ranges backed by evidence (CircuitKIT's benchmarks or a cited paper) are enforced
as warnings; each entry records its confidence and the reason, and the same registry
renders the reference page ``docs/reference/hyperparameters.md``.

This module is deliberately pure Python (no torch / TransformerLens import) so that it
can run during config validation and in the docs build.

The warning class derives from ``Warning`` (not ``UserWarning``), so the blanket
``UserWarning`` filter that ``circuitkit.api`` installs does not hide it. Silence the warnings
with the standard machinery, e.g.::

    import warnings
    from circuitkit.utils.hparams import HyperparameterWarning
    warnings.filterwarnings("ignore", category=HyperparameterWarning)
"""

from __future__ import annotations

import math
import numbers
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "HParam",
    "HParamIssue",
    "HPARAMS",
    "HyperparameterError",
    "HyperparameterWarning",
    "PILLARS_BASIC",
    "PILLARS_VALID",
    "STRICT_ENV_VAR",
    "check_value",
    "check_rules",
    "collect_config_values",
    "describe",
    "is_strict",
    "render_markdown",
    "validate",
    "validate_config_hparams",
]

STRICT_ENV_VAR = "CIRCUITKIT_STRICT_HPARAMS"

# Pillar names, defined once. ``PILLARS_BASIC`` is what the user-facing evaluation entry
# points (``Pipeline.evaluate``, ``ck.faithfulness``) run by default; ``PILLARS_VALID`` is
# every name ``run_full_faithfulness`` accepts and what it runs when ``pillars=None``.
PILLARS_BASIC: Tuple[str, ...] = ("patching", "ablation")
PILLARS_VALID: Tuple[str, ...] = (
    "patching",
    "ablation",
    "baselines",
    "robustness",
    "stability",
    "generalization",
    "intervention_reliability",
)

# What the ACDC backend sweeps when ``tao_bases`` / ``tao_exps`` are not given.
_ACDC_DEFAULT_BASES = (1, 5)
_ACDC_DEFAULT_EXPS = (-5, -4, -3, -2)

# The algorithms whose attribution reads ``ig_steps``.
_IG_ALGORITHMS = ("eap-ig", "eap-ig-activations", "eap-gp")


class HyperparameterError(ValueError):
    """A hyperparameter is outside its *valid* range (or, in strict mode, its sensible one)."""


class HyperparameterWarning(Warning):
    """A hyperparameter is valid but outside the range earlier runs found sensible.

    Deliberately a direct subclass of ``Warning`` and not of ``UserWarning``:
    ``circuitkit.api`` installs ``warnings.filterwarnings("ignore", category=UserWarning)`` at
    import time, which would silence every ``UserWarning`` subclass in normal use.
    """


@dataclass(frozen=True)
class HParam:
    """One hyperparameter: its valid range, its sensible range, and why."""

    key: str  # "<section>.<name>", unique
    summary: str
    kind: str  # int | float | enum | list_enum | list_num | list_int
    surfaces: str = ""  # where a user sets it (docs only)
    default: Any = None  # compared with DEFAULT_CONFIG when ``config_path`` is set
    default_note: str = ""  # shown instead of ``default`` when it depends on the surface
    config_path: Tuple[str, ...] = ()  # location in the dict/YAML config, if any
    applies_to: Tuple[str, ...] = ()  # algorithms the parameter is read by; () = all
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    valid_min_open: bool = False
    valid_max_open: bool = False
    valid_values: Tuple[Any, ...] = ()
    sensible_min: Optional[float] = None
    sensible_max: Optional[float] = None
    scale: str = "linear"  # linear | log (how the sensible range should be read)
    checked_by: str = "docs"  # hparams | existing | docs
    confidence: str = "medium"  # high | medium
    sensible_warns: bool = True  # False: the sensible range is documented, not warned on
    why: str = ""

    @property
    def section(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.key.split(".", 1)[1]

    @property
    def warns(self) -> bool:
        has_range = self.sensible_min is not None or self.sensible_max is not None
        return self.checked_by == "hparams" and self.sensible_warns and has_range


@dataclass(frozen=True)
class HParamIssue:
    key: str
    value: Any
    severity: str  # "error" | "warning"
    message: str


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
# Only ranges with a citable basis are listed. Ranges that rest on a single anecdote or on
# intuition (steering coefficients, weight steering, knowledge editing, LoRA healing) are not
# published here until they have been validated.
_SPECS: Tuple[HParam, ...] = (
    # ------------------------------------------------------------- discovery
    HParam(
        key="discovery.num_examples",
        summary="Examples used to estimate attribution scores.",
        kind="int",
        surfaces="dict `discovery.data_params.num_examples` · flat/Pipeline `n_examples` · CLI `--num-examples`",
        default=128,
        config_path=("discovery", "data_params", "num_examples"),
        valid_min=1,
        sensible_min=32,
        sensible_max=1024,
        scale="log",
        checked_by="hparams",
        why=(
            "Scores from very few examples are noisy: in CircuitKIT's IOI benchmarks the top-30% "
            "node sets of three seeds overlapped only ~0.6-0.7 (Jaccard) at 128 examples, and the "
            "EAP-IG, ACDC and CD-T papers all work with ~100 examples. Cost grows linearly, so "
            "beyond ~1000 there is little to gain."
        ),
    ),
    HParam(
        key="discovery.batch_size",
        summary="Examples per attribution batch.",
        kind="int",
        surfaces="dict `discovery.batch_size` · flat/Pipeline `batch_size` · CLI `--batch-size`",
        default=4,
        config_path=("discovery", "batch_size"),
        valid_min=1,
        checked_by="hparams",
        why=(
            "Memory-scaled, so there is no fixed sensible range: in CircuitKIT's benchmarks 2 "
            "worked for 3-4B models and 16 for models up to 1.5B. `batch_size=1` is always safe, "
            "just slower."
        ),
    ),
    HParam(
        key="discovery.ig_steps",
        summary="Integration steps for the EAP-IG family (eap-ig, eap-ig-activations, eap-gp).",
        kind="int",
        surfaces="dict `discovery.ig_steps` · CLI `--ig-steps` · flat/Pipeline via `**kw`",
        default=3,
        default_note="3 (5 for eap-gp)",
        config_path=("discovery", "ig_steps"),
        applies_to=_IG_ALGORITHMS,
        valid_min=1,
        checked_by="hparams",
        confidence="high",
        why=(
            "Ignored by every other algorithm. One step is plain EAP. The EAP-IG paper "
            "(Hanna et al., 2024) tested 2 to 50 steps: 2 is unfaithful on some tasks, every "
            "value above 2 is similarly faithful. The authors used 5 to leave a margin and tested "
            "GPT-2 small on three tasks. Cost grows linearly with steps. EAP-GP reads "
            "the same key and defaults to 5 (its paper's k) when the key is not set."
        ),
    ),
    HParam(
        key="discovery.tao_bases",
        summary="ACDC threshold grid: tau = base x 10^exp for every (base, exp) pair.",
        kind="list_num",
        surfaces="dict `discovery.tao_bases`",
        default_note="[1, 5] (ACDC backend default)",
        config_path=("discovery", "tao_bases"),
        applies_to=("acdc",),
        valid_min=0,
        valid_min_open=True,
        checked_by="hparams",
        why=(
            "ACDC sweeps a log-spaced grid of thresholds and keeps one circuit per tau. "
            "Published ACDC circuits use tau between about 4e-3 and 1e-1 (KL divergence)."
        ),
    ),
    HParam(
        key="discovery.tao_exps",
        summary="Exponents of the ACDC threshold grid (see `tao_bases`).",
        kind="list_int",
        surfaces="dict `discovery.tao_exps`",
        default_note="[-5, -4, -3, -2] (ACDC backend default)",
        config_path=("discovery", "tao_exps"),
        applies_to=("acdc",),
        checked_by="hparams",
        why="A threshold of 1 or more keeps almost every edge, so such a grid point is flagged.",
    ),
    HParam(
        key="discovery.num_epochs",
        summary="Training epochs for the IBCircuit mask.",
        kind="int",
        surfaces="dict `discovery.num_epochs`",
        default=1000,
        config_path=("discovery", "num_epochs"),
        applies_to=("ibcircuit",),
        valid_min=1,
        sensible_min=500,
        sensible_max=1500,
        sensible_warns=False,
        checked_by="hparams",
        why=(
            "IBCircuit learns its mask by gradient descent, so it needs a real training budget. "
            "The existing IBCircuit check still warns below 100 epochs; the upstream IBCircuit "
            "repo trains 1300 epochs and CircuitKIT's runs use 1000."
        ),
    ),
    # ---------------------------------------------------------------- pruning
    HParam(
        key="pruning.target_sparsity",
        summary="Fraction of heads/MLPs removed (the circuit keeps the rest).",
        kind="float",
        surfaces="dict `pruning.target_sparsity` · flat/Pipeline/CLI `sparsity`",
        default=0.3,
        config_path=("pruning", "target_sparsity"),
        valid_min=0.0,
        valid_max=1.0,
        sensible_max=0.4,
        checked_by="hparams",
        why=(
            "At 30% sparsity accuracy retention ranged from 0.65 (random) to 0.99 (Taylor) on "
            "Llama-3.2-3B and Gemma-3-4B. At 50% every selector tried collapsed (perplexity 365 "
            "to 220,000) without recovery fine-tuning."
        ),
    ),
    # ------------------------------------------------------------- evaluation
    HParam(
        key="eval.num_examples",
        summary="Held-out examples used by the faithfulness pillars.",
        kind="int",
        surfaces="dict `eval.num_examples` · flat `n_examples` · Pipeline.evaluate `n_examples`",
        default=256,
        config_path=("eval", "num_examples"),
        valid_min=1,
        sensible_min=100,
        sensible_max=1000,
        scale="log",
        checked_by="hparams",
        why=(
            "Faithfulness is a ratio of two noisy averages. The EAP-IG paper evaluates on 100 "
            "examples; CircuitKIT's benchmarks use 300 held-out examples."
        ),
    ),
    HParam(
        key="eval.n_stability_runs",
        summary="Independent re-discoveries compared by the stability pillar.",
        kind="int",
        surfaces="dict `eval.n_stability_runs` · Pipeline.evaluate `n_stability_runs`",
        default=3,
        config_path=("eval", "n_stability_runs"),
        valid_min=1,
        sensible_min=3,
        sensible_max=10,
        checked_by="hparams",
        why=(
            "Stability is the overlap between re-discovered circuits. With 1 run there is nothing "
            "to compare, so the pillar reports a perfect overlap of 1.0; 2 runs give a single "
            "pairwise overlap. Seed-to-seed variance is large: in a seed replicate of CircuitKIT's "
            "benchmarks, patch faithfulness flipped sign in 3 of 5 cells. Each run repeats "
            "discovery, so cost grows linearly."
        ),
    ),
    HParam(
        key="eval.pillars",
        summary="Faithfulness pillars to compute.",
        kind="list_enum",
        surfaces="dict `eval.pillars` · flat `pillars` · Pipeline.evaluate `pillars`",
        default_note=(
            "`patching` + `ablation` (Pipeline, flat API, `circuitkit run`); every pillar for a dict "
            "config with `full_faithfulness_eval: true`; `\"all\"` runs every pillar"
        ),
        config_path=("eval", "pillars"),
        valid_values=PILLARS_VALID,
        checked_by="hparams",
        confidence="high",
        why=(
            "Names are checked up front so a typo fails before any discovery or evaluation work "
            "starts. Stability and generalization re-run discovery and are the expensive ones."
        ),
    ),
    HParam(
        key="benchmark.limit",
        summary="Cap on examples per lm-eval task.",
        kind="float",
        surfaces="flat `benchmark(limit=...)` · Pipeline.benchmark `limit` · CLI `--limit`",
        default_note="unset (full task)",
        valid_min=0,
        valid_min_open=True,
        checked_by="docs",
        why=(
            "lm-evaluation-harness documents `limit` as for testing (a value below 1 is a "
            "fraction of the task). Leave it unset for numbers you report."
        ),
    ),
    # ----------------------------------------------------------- quantization
    HParam(
        key="quantization.bits",
        summary="Bit-width of the quantized tier (llmcompressor backend).",
        kind="int",
        surfaces="flat `quantize(bits=...)` · Pipeline.quantize `bits` · CLI `--bits`",
        default_note="4 (Pipeline, CLI) · 3 (flat `ck.quantize`)",
        valid_values=(3, 4, 8),
        checked_by="existing",
        why=(
            "The llmcompressor backend accepts 3, 4 or 8 and raises otherwise. The quanto backend "
            "ignores `bits` (it uses qint2/qint4/qint8 tiers)."
        ),
    ),
    HParam(
        key="quantization.high_fraction",
        summary="Fraction of layers kept at high precision.",
        kind="float",
        surfaces="flat `quantize(high_fraction=...)` · Pipeline.quantize · CLI `--high-fraction`",
        default_note="0.3 (code) · 0.05 in CircuitKIT's benchmarks",
        valid_min=0.0,
        valid_max=1.0,
        sensible_min=0.05,
        sensible_max=0.3,
        sensible_warns=False,
        checked_by="hparams",
        why=(
            "With a 4-bit base, protecting 5% of layers kept 0.98 accuracy retention for every "
            "selector. At a 3-bit base the choice of selector only started to matter at 15% "
            "protected. The range itself is not warned on; a warning is raised only when "
            "`bits <= 3` and fewer than 15% of layers are protected (llmcompressor backend)."
        ),
    ),
)

HPARAMS: Dict[str, HParam] = {spec.key: spec for spec in _SPECS}
if len(HPARAMS) != len(_SPECS):  # pragma: no cover - guards the registry itself
    raise RuntimeError("duplicate hyperparameter key in circuitkit.utils.hparams")

_SECTION_TITLES = {
    "discovery": "Discovery",
    "pruning": "Pruning",
    "eval": "Evaluation",
    "benchmark": "Benchmarking",
    "quantization": "Quantization",
}


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def _num(value: Any) -> str:
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
        return f"{value:g}"
    return str(value)


def _describe_valid(spec: HParam, md: bool = False) -> str:
    if spec.valid_values:
        sep = " \\| " if md else " | "
        return sep.join(f"`{v}`" for v in spec.valid_values)
    lo, hi = spec.valid_min, spec.valid_max
    if spec.kind == "list_int":
        return "non-empty list of integers"
    if spec.kind == "list_num":
        bound = f"{'>' if spec.valid_min_open else '>='} {_num(lo)}" if lo is not None else "a number"
        return f"non-empty list, each {bound}"
    if lo is None and hi is None:
        return "finite number"
    left = "(" if spec.valid_min_open else "["
    right = ")" if spec.valid_max_open else "]"
    if hi is None:
        return f"{'>' if spec.valid_min_open else '>='} {_num(lo)}"
    if lo is None:
        return f"{'<' if spec.valid_max_open else '<='} {_num(hi)}"
    return f"{left}{_num(lo)}, {_num(hi)}{right}"


def _describe_sensible(spec: HParam) -> str:
    lo, hi = spec.sensible_min, spec.sensible_max
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"<= {_num(hi)}"
    if hi is None:
        return f">= {_num(lo)}"
    return f"{_num(lo)} to {_num(hi)}"


# --------------------------------------------------------------------------- #
# Checking                                                                    #
# --------------------------------------------------------------------------- #
def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    if isinstance(value, numbers.Integral):
        return True  # arbitrarily large ints are finite; do not route them through float()
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _scalar_issues(spec: HParam, value: Any, what: str) -> List[HParamIssue]:
    """Valid-range checks for one numeric value; ``what`` names it in the message."""
    wants_int = spec.kind in ("int", "list_int")
    kind_word = "an integer" if wants_int else "a finite number"
    # An integer parameter must be an integer type: ``3.0`` would pass a range check and then
    # fail in the backend (``range(1, steps + 1)``). ``bool`` is not a number here.
    if not _is_number(value) or (wants_int and not isinstance(value, numbers.Integral)) or not _is_finite(value):
        return [HParamIssue(spec.key, value, "error", f"{what} must be {kind_word}, got {value!r}.")]
    if spec.valid_values:
        if value not in spec.valid_values:
            allowed = ", ".join(str(v) for v in spec.valid_values)
            return [HParamIssue(spec.key, value, "error", f"{what} must be one of {allowed}, got {value!r}.")]
        return []
    lo, hi = spec.valid_min, spec.valid_max
    too_low = lo is not None and (value <= lo if spec.valid_min_open else value < lo)
    too_high = hi is not None and (value >= hi if spec.valid_max_open else value > hi)
    if too_low or too_high:
        return [
            HParamIssue(
                spec.key,
                value,
                "error",
                f"{what} must be in the valid range {_describe_valid(spec)}, got {value!r}.",
            )
        ]
    return []


def check_value(key: str, value: Any) -> List[HParamIssue]:
    """Check one value against its registry entry; returns errors and sensible-range warnings."""
    spec = HPARAMS[key]
    if value is None:
        return []

    if spec.kind in ("int", "float"):
        issues = _scalar_issues(spec, value, key)
        if issues:
            return issues
        low = spec.sensible_min is not None and value < spec.sensible_min
        high = spec.sensible_max is not None and value > spec.sensible_max
        if spec.warns and (low or high):
            return [
                HParamIssue(
                    key,
                    value,
                    "warning",
                    f"{key}={value!r} is outside the sensible range {_describe_sensible(spec)}. {spec.why}",
                )
            ]
        return []

    if spec.kind == "enum":
        if value not in spec.valid_values:
            allowed = ", ".join(str(v) for v in spec.valid_values)
            return [HParamIssue(key, value, "error", f"{key} must be one of {allowed}, got {value!r}.")]
        return []

    # list kinds. For pillars the string "all" means every pillar (the same spelling
    # Pipeline.evaluate and ck.faithfulness accept).
    if spec.kind == "list_enum" and isinstance(value, str) and value == "all":
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [HParamIssue(key, value, "error", f"{key} must be a list, got {value!r}.")]
    items = list(value)
    if spec.kind == "list_enum":
        bad = [v for v in items if v not in spec.valid_values]
        if bad:
            allowed = ", ".join(spec.valid_values)
            return [HParamIssue(key, value, "error", f"{key} has unknown entries {bad!r}; valid: {allowed}, or \"all\".")]
    if not items:
        return [HParamIssue(key, value, "error", f"{key} must be a non-empty list.")]
    if spec.kind == "list_enum":
        return []
    issues: List[HParamIssue] = []
    for item in items:
        issues.extend(_scalar_issues(spec, item, f"{key} entry"))
    return issues


def check_rules(values: Mapping[str, Any]) -> List[HParamIssue]:
    """Cross-parameter checks that a per-parameter range cannot express."""
    issues: List[HParamIssue] = []

    # ACDC: a threshold of 1 or more keeps (almost) every edge. A list the user did not set
    # takes the backend default, so ``tao_exps: [0, 1]`` alone is checked against bases [1, 5].
    bases, exps = values.get("discovery.tao_bases"), values.get("discovery.tao_exps")
    if bases is not None or exps is not None:
        try:
            base_list = list(_ACDC_DEFAULT_BASES if bases is None else bases)
            exp_list = list(_ACDC_DEFAULT_EXPS if exps is None else exps)
            taus = {float(b) * 10.0 ** int(e) for b in base_list for e in exp_list}
        except OverflowError:
            taus = {math.inf}  # e.g. tao_exps: [400]; far above 1, so it is reported below
        except (TypeError, ValueError):
            taus = set()  # malformed lists are reported by check_value
        big = sorted(t for t in taus if t >= 1)
        if big:
            issues.append(
                HParamIssue(
                    "discovery.tao_exps" if exps is not None else "discovery.tao_bases",
                    list(exp_list) if exps is not None else base_list,
                    "warning",
                    f"ACDC threshold grid contains tau >= 1 ({', '.join(_num(b) for b in big)}), "
                    "which keeps almost every edge. Published ACDC circuits use tau between "
                    "about 4e-3 and 1e-1.",
                )
            )

    # Quantization: a 3-bit base needs more protected layers than a 4-bit one.
    bits, high = values.get("quantization.bits"), values.get("quantization.high_fraction")
    backend = values.get("quantization.backend", "llmcompressor")
    if backend == "llmcompressor" and _is_number(bits) and _is_number(high) and bits <= 3 and high < 0.15:
        issues.append(
            HParamIssue(
                "quantization.high_fraction",
                high,
                "warning",
                f"quantization.bits={bits} with only {high:.0%} of layers protected: in "
                "CircuitKIT's benchmarks a 3-bit base only separated selectors once ~15% of layers "
                "stayed at high precision (4-bit with 5% protected kept 0.98 accuracy retention).",
            )
        )
    return issues


_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off", "")


def _as_flag(value: Any, source: str) -> bool:
    """Read a strict-mode setting. ``"false"`` (a quoted YAML scalar) must not count as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise HyperparameterError(f"{source} must be true or false, got {value!r}.")


def is_strict(strict: Optional[Any] = None, *, source: str = "strict") -> bool:
    """Resolve strict mode: an explicit argument wins, then the environment variable."""
    if strict is not None:
        return _as_flag(strict, source)
    return os.environ.get(STRICT_ENV_VAR, "").strip().lower() in _TRUE_WORDS


def validate(
    values: Mapping[str, Any],
    *,
    user_set: Optional[Iterable[str]] = None,
    strict: Optional[bool] = None,
    stacklevel: int = 2,
) -> List[HParamIssue]:
    """Validate ``{registry key: value}``.

    Invalid values raise :class:`HyperparameterError` (all problems in one message).
    Sensible-range problems emit :class:`HyperparameterWarning`, or raise too in strict
    mode. When ``user_set`` is given, only values whose key is in it are warned about, so
    defaults never produce noise.
    """
    issues: List[HParamIssue] = []
    for key, value in values.items():
        if key not in HPARAMS:
            continue
        try:
            issues.extend(check_value(key, value))
        except (TypeError, ValueError, OverflowError) as exc:  # an array, a huge int, ...
            issues.append(
                HParamIssue(key, value, "error", f"{key} could not be checked ({type(exc).__name__}: {exc}).")
            )
    issues.extend(check_rules(values))

    wanted = None if user_set is None else set(user_set)
    errors = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warning" and (wanted is None or i.key in wanted)]

    if is_strict(strict):
        errors.extend(warns)
        warns = []
    if errors:
        body = "\n  - ".join(i.message for i in errors)
        raise HyperparameterError(f"Invalid hyperparameter value(s):\n  - {body}")
    for issue in warns:
        warnings.warn(issue.message, HyperparameterWarning, stacklevel=stacklevel + 1)
    return warns


# --------------------------------------------------------------------------- #
# Dict / YAML config integration                                              #
# --------------------------------------------------------------------------- #
_MISSING = object()


def _lookup(mapping: Any, path: Sequence[str]) -> Any:
    cur = mapping
    for part in path:
        if not isinstance(cur, Mapping) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def collect_config_values(
    config: Mapping[str, Any], user_config: Optional[Mapping[str, Any]] = None
) -> Tuple[Dict[str, Any], set]:
    """Pull registry values out of a merged config.

    Returns ``(values, user_set)``. A parameter is skipped when the chosen algorithm does not
    read it. ``user_set`` is the subset the caller wrote themselves (present in ``user_config``,
    before defaults were merged in).
    """
    algo = str(_lookup(config, ("discovery", "algorithm")) or "").lower()
    values: Dict[str, Any] = {}
    user_set: set = set()
    for spec in HPARAMS.values():
        if not spec.config_path or (spec.applies_to and algo not in spec.applies_to):
            continue
        value = _lookup(config, spec.config_path)
        if value is _MISSING:
            continue
        values[spec.key] = value
        if user_config is not None and _lookup(user_config, spec.config_path) is not _MISSING:
            user_set.add(spec.key)
    return values, user_set


def validate_config_hparams(
    config: Mapping[str, Any], user_config: Optional[Mapping[str, Any]] = None
) -> List[HParamIssue]:
    """Validate the hyperparameters of a merged dict/YAML config.

    Strict mode comes from ``config["validation"]["strict"]`` or the environment variable.
    """
    values, user_set = collect_config_values(config, user_config)
    strict = _lookup(config, ("validation", "strict"))
    if strict is not _MISSING and strict is not None:
        strict = _as_flag(strict, "validation.strict")
    return validate(
        values,
        user_set=user_set if user_config is not None else None,
        strict=None if strict is _MISSING else strict,
        stacklevel=3,
    )


# --------------------------------------------------------------------------- #
# Documentation                                                               #
# --------------------------------------------------------------------------- #
def _checked_by(spec: HParam) -> str:
    if spec.checked_by == "hparams":
        return "valid-range error + sensible-range warning" if spec.warns else "valid-range error"
    return {"existing": "existing check in code", "docs": "documented only"}[spec.checked_by]


def describe(spec: HParam, md: bool = False) -> Dict[str, str]:
    """The human-readable cells for one entry (docs table with ``md=True``, CLI table otherwise)."""
    if spec.default_note:
        default = spec.default_note
    elif spec.default is not None:
        default = f"`{_num(spec.default)}`" if md else _num(spec.default)
    else:
        default = ""
    sensible = _describe_sensible(spec)
    if sensible and spec.scale == "log":
        sensible += " (log scale)"
    if sensible and not spec.sensible_warns:
        sensible += " (not warned on)"
    return {
        "default": default,
        "valid": _describe_valid(spec, md=md),
        "sensible": sensible or "—",
        "checked_by": _checked_by(spec),
        "confidence": spec.confidence,
    }


def render_markdown() -> str:
    """Render the registry as the tables on ``docs/reference/hyperparameters.md``."""
    out: List[str] = []
    for section, title in _SECTION_TITLES.items():
        specs = [s for s in HPARAMS.values() if s.section == section]
        if not specs:
            continue
        out.append(f"### {title}\n")
        out.append("| Parameter | Default | Valid | Sensible | Checked by | Confidence |")
        out.append("|---|---|---|---|---|---|")
        for s in specs:
            cells = describe(s, md=True)
            out.append(
                f"| `{s.name}` | {cells['default']} | {cells['valid']} | {cells['sensible']} | "
                f"{cells['checked_by']} | {cells['confidence']} |"
            )
        out.append("")
        for s in specs:
            where = f" Set via {s.surfaces}." if s.surfaces else ""
            applies = f" Read by: {', '.join(s.applies_to)}." if s.applies_to else ""
            out.append(f"**`{s.name}`**: {s.summary}{where}{applies} {s.why}\n")
    return "\n".join(out).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin CLI
    """``python -m circuitkit.utils.hparams [--write PATH | --check PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Render or check the hyperparameter reference page.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", metavar="PATH", help="Rewrite the generated block of PATH.")
    group.add_argument("--check", metavar="PATH", help="Exit 1 if PATH's generated block is stale.")
    args = parser.parse_args(argv)
    start, end = "<!-- hparams:start -->", "<!-- hparams:end -->"
    block = render_markdown()
    if not (args.write or args.check):
        print(block, end="")
        return 0
    path = args.write or args.check
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    new = f"{head}{start}\n\n{block}\n{end}{tail}"
    if args.write:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(new)
        return 0
    return 0 if new == text else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
