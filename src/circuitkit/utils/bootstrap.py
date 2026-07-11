"""
Bootstrap utility for computing confidence intervals on metrics.

Provides a reusable bootstrap loop for computing robust estimates of any metric
with confidence intervals via resampling.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


def bootstrap(
    metric_fn: Callable,
    data: List[Any],
    n_samples: int = 100,
    sample_size: Optional[int] = None,
    seed: int = 42,
    return_all: bool = False,
    ci: float = 0.95,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Bootstrap metric computation with resampling.

    Computes a metric on random resamples of the data and returns the
    distribution of metric values across bootstraps, along with statistics.

    Args:
        metric_fn (Callable): Function that takes a data subset (List[Any])
            and returns a scalar metric value (float).
        data (List[Any]): Full dataset to resample from.
        n_samples (int): Number of bootstrap samples. Defaults to 100.
        sample_size (Optional[int]): Size of each bootstrap sample.
            If None, defaults to len(data) (sampling with replacement).
        seed (int): Random seed for reproducibility. Defaults to 42.
        return_all (bool): If True, also return all bootstrap values in output.
            Defaults to False.
        ci (float): Confidence interval level (e.g., 0.95 for 95% CI).
            Defaults to 0.95.
        quiet (bool): If True, suppress progress bar. Defaults to False.

    Returns:
        Dict[str, Any] with keys:
            'mean': float - Mean of bootstrap metric values
            'std': float - Standard deviation of bootstrap values
            'ci_lower': float - Lower confidence interval bound
            'ci_upper': float - Upper confidence interval bound
            'median': float - Median of bootstrap values
            'min': float - Minimum bootstrap value
            'max': float - Maximum bootstrap value
            'all_values': Optional[np.ndarray] - All bootstrap values (if return_all=True)

    Raises:
        ValueError: If data is empty or sample_size > len(data) without replacement.
        TypeError: If metric_fn doesn't return a scalar.
    """
    if not data:
        raise ValueError("data cannot be empty")

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)
    if torch is not None:
        torch.manual_seed(seed)

    # Default sample size: same as full dataset (with replacement)
    if sample_size is None:
        sample_size = len(data)

    # Bootstrap loop
    bootstrap_values = []
    iterator = tqdm(range(n_samples), disable=quiet, desc="Bootstrap")

    for _ in iterator:
        # Sample indices with replacement
        indices = rng.choice(len(data), size=sample_size, replace=True)
        sample = [data[i] for i in indices]

        # Compute metric on this sample
        metric_value = metric_fn(sample)

        # Ensure we got a scalar
        if torch is not None and isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        elif not isinstance(metric_value, (int, float)):
            raise TypeError(f"metric_fn must return scalar, got {type(metric_value)}")

        bootstrap_values.append(float(metric_value))

    bootstrap_values = np.array(bootstrap_values)

    # Compute statistics
    ci_lower_percentile = (1 - ci) / 2 * 100
    ci_upper_percentile = (1 + ci) / 2 * 100

    result = {
        "mean": float(np.mean(bootstrap_values)),
        "std": float(np.std(bootstrap_values)),
        "ci_lower": float(np.percentile(bootstrap_values, ci_lower_percentile)),
        "ci_upper": float(np.percentile(bootstrap_values, ci_upper_percentile)),
        "median": float(np.median(bootstrap_values)),
        "min": float(np.min(bootstrap_values)),
        "max": float(np.max(bootstrap_values)),
    }

    if return_all:
        result["all_values"] = bootstrap_values

    return result


def bootstrap_metric_parallel(
    metric_fn: Callable,
    data_splits: List[List[Any]],
    aggregate_fn: Callable = lambda x: np.mean(x),
    n_samples: int = 100,
    seed: int = 42,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Bootstrap a metric that operates on multiple data splits (e.g., tasks).

    Useful for computing metrics like cross-task transfer where you need to
    aggregate across multiple tasks. Each bootstrap sample resamples from
    all splits independently.

    Args:
        metric_fn (Callable): Function that takes a data split and returns
            a scalar metric value.
        data_splits (List[List[Any]]): List of datasets, one per split/task.
        aggregate_fn (Callable): Function to combine per-split metrics into
            a single scalar. Defaults to np.mean.
        n_samples (int): Number of bootstrap samples. Defaults to 100.
        seed (int): Random seed. Defaults to 42.
        quiet (bool): Suppress progress bar. Defaults to False.

    Returns:
        Dict[str, Any]: Same keys as bootstrap(), representing the distribution
            of the aggregated metric across splits.
    """
    if not data_splits:
        raise ValueError("data_splits cannot be empty")

    rng = np.random.RandomState(seed)
    if torch is not None:
        torch.manual_seed(seed)

    bootstrap_values = []
    iterator = tqdm(range(n_samples), disable=quiet, desc="Bootstrap (parallel)")

    for _ in iterator:
        # Resample each split independently
        metrics = []
        for split in data_splits:
            indices = rng.choice(len(split), size=len(split), replace=True)
            sample = [split[i] for i in indices]
            metric_value = metric_fn(sample)

            if torch is not None and isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            metrics.append(float(metric_value))

        # Aggregate across splits
        aggregated = aggregate_fn(np.array(metrics))
        if isinstance(aggregated, np.ndarray):
            aggregated = float(aggregated)
        bootstrap_values.append(aggregated)

    bootstrap_values = np.array(bootstrap_values)

    result = {
        "mean": float(np.mean(bootstrap_values)),
        "std": float(np.std(bootstrap_values)),
        "ci_lower": float(np.percentile(bootstrap_values, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_values, 97.5)),
        "median": float(np.median(bootstrap_values)),
        "min": float(np.min(bootstrap_values)),
        "max": float(np.max(bootstrap_values)),
        "all_values": bootstrap_values,
    }

    return result
