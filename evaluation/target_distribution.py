"""Distribution diagnostics for normalized mechanism trajectory targets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from aging_models.aging_label_generator import AgingLabelGenerator


CANONICAL_MECHANISM_ORDER = AgingLabelGenerator.MECHANISM_ORDER
DEFAULT_NEAR_TOLERANCE = 1e-6
# For normalized [0, 1] targets, a variance at or below 1e-7 represents a
# practically compressed target range even when individual float values differ.
DEFAULT_LOW_VARIANCE_THRESHOLD = 1e-7


def _as_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float64)


def _validate_targets(values: Any) -> np.ndarray:
    array = _as_array(values)
    if array.ndim != 3 or array.shape[-1] != len(CANONICAL_MECHANISM_ORDER):
        raise ValueError(
            "Mechanism trajectories must have shape [num_nodes, horizon, 3]; "
            f"got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Mechanism trajectories must contain nodes and horizon steps")
    if not np.all(np.isfinite(array)):
        raise ValueError("Mechanism trajectories must contain only finite values")
    return array


def summarize_values(
    values: Any,
    *,
    near_tolerance: float = DEFAULT_NEAR_TOLERANCE,
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    """Return JSON-safe descriptive statistics with saturation diagnostics."""
    array = _as_array(values).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty target array")
    if near_tolerance <= 0:
        raise ValueError("near_tolerance must be positive")

    quantized = np.rint(array / near_tolerance).astype(np.int64)
    unique_count = int(np.unique(quantized).size)
    variance = float(np.var(array))
    near_zero_fraction = float(np.mean(np.abs(array) <= near_tolerance))
    near_one_fraction = float(np.mean(np.abs(array - 1.0) <= near_tolerance))
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "variance": variance,
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "approx_unique_count": unique_count,
        "approx_unique_fraction": float(unique_count / array.size),
        "near_zero_fraction": near_zero_fraction,
        "near_one_fraction": near_one_fraction,
        "low_variance": bool(variance <= low_variance_threshold),
        "near_constant": bool(unique_count <= 1),
        "near_zero_saturation": bool(near_zero_fraction >= 0.99),
        "near_one_saturation": bool(near_one_fraction >= 0.99),
        "near_zero_range": bool(float(np.percentile(array, 99)) <= 0.01),
    }


def summarize_increments(
    values: Any,
    *,
    near_tolerance: float = DEFAULT_NEAR_TOLERANCE,
) -> dict[str, Any]:
    """Summarize adjacent future-state increments for one mechanism."""
    array = _as_array(values)
    if array.ndim != 2:
        raise ValueError(f"Expected [num_nodes, horizon] mechanism values, got {array.shape}")
    if array.shape[1] < 2:
        return {"increments": [], "aggregate": None}

    deltas = array[:, 1:] - array[:, :-1]

    def summary(delta: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "std": float(np.std(delta)),
            "near_zero_fraction": float(np.mean(np.abs(delta) <= near_tolerance)),
            "positive_fraction": float(np.mean(delta > near_tolerance)),
            "negative_fraction": float(np.mean(delta < -near_tolerance)),
        }

    return {
        "increments": [
            {"from_horizon": index + 1, "to_horizon": index + 2, **summary(deltas[:, index])}
            for index in range(deltas.shape[1])
        ],
        "aggregate": summary(deltas),
    }


def analyze_mechanism_trajectories(
    targets: Any,
    *,
    mechanism_names: Sequence[str] = CANONICAL_MECHANISM_ORDER,
    near_tolerance: float = DEFAULT_NEAR_TOLERANCE,
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    """Analyze ``[num_nodes, horizon, 3]`` targets by mechanism and horizon."""
    array = _validate_targets(targets)
    names = tuple(str(name).lower() for name in mechanism_names)
    if names != CANONICAL_MECHANISM_ORDER:
        raise ValueError(
            f"Mechanism order must be {CANONICAL_MECHANISM_ORDER}; got {names}"
        )

    mechanisms = {}
    for mechanism_index, name in enumerate(names):
        values = array[:, :, mechanism_index]
        mechanisms[name] = {
            "overall": summarize_values(
                values,
                near_tolerance=near_tolerance,
                low_variance_threshold=low_variance_threshold,
            ),
            "by_horizon": [
                {
                    "horizon": horizon_index + 1,
                    **summarize_values(
                        values[:, horizon_index],
                        near_tolerance=near_tolerance,
                        low_variance_threshold=low_variance_threshold,
                    ),
                }
                for horizon_index in range(array.shape[1])
            ],
            "increments": summarize_increments(
                values, near_tolerance=near_tolerance
            ),
        }

    return {
        "shape": [int(size) for size in array.shape],
        "mechanism_order": list(names),
        "near_tolerance": float(near_tolerance),
        "low_variance_threshold": float(low_variance_threshold),
        "mechanisms": mechanisms,
    }


def compare_target_splits(
    split_targets: Mapping[str, Any],
    *,
    near_tolerance: float = DEFAULT_NEAR_TOLERANCE,
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    """Analyze named splits and expose compact per-mechanism distribution ranges."""
    if not split_targets:
        raise ValueError("At least one named split is required")
    analyses = {
        str(name): analyze_mechanism_trajectories(
            values,
            near_tolerance=near_tolerance,
            low_variance_threshold=low_variance_threshold,
        )
        for name, values in split_targets.items()
    }
    comparisons = {}
    for name in CANONICAL_MECHANISM_ORDER:
        summaries = {
            split: analysis["mechanisms"][name]["overall"]
            for split, analysis in analyses.items()
        }
        comparisons[name] = {
            "mean_range": float(max(item["mean"] for item in summaries.values()) - min(item["mean"] for item in summaries.values())),
            "std_range": float(max(item["std"] for item in summaries.values()) - min(item["std"] for item in summaries.values())),
            "variance_range": float(max(item["variance"] for item in summaries.values()) - min(item["variance"] for item in summaries.values())),
            "split_overall": summaries,
        }
    return {"splits": analyses, "comparison": comparisons}
