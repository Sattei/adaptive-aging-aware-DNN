"""Axis-aware metrics for normalized mechanism trajectory forecasts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from aging_models.aging_label_generator import AgingLabelGenerator


CANONICAL_MECHANISM_ORDER = AgingLabelGenerator.MECHANISM_ORDER


def _as_numpy(values: Any) -> np.ndarray:
    """Convert NumPy-compatible inputs, including detached torch tensors."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def _validate_inputs(
    predictions: Any,
    targets: Any,
    mechanism_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str]]:
    predictions_array = _as_numpy(predictions)
    targets_array = _as_numpy(targets)

    if predictions_array.shape != targets_array.shape:
        raise ValueError(
            "predictions and targets must have identical shapes; "
            f"got {predictions_array.shape} and {targets_array.shape}"
        )
    if predictions_array.ndim != 3:
        raise ValueError(
            "mechanism trajectories must have shape [num_nodes, horizon, 3]; "
            f"got rank {predictions_array.ndim} with shape {predictions_array.shape}"
        )
    if predictions_array.shape[2] != len(CANONICAL_MECHANISM_ORDER):
        raise ValueError(
            "mechanism trajectory final dimension must be 3 "
            f"({CANONICAL_MECHANISM_ORDER}); got {predictions_array.shape[2]}"
        )
    if predictions_array.shape[0] == 0:
        raise ValueError("mechanism trajectories must contain at least one node")
    if predictions_array.shape[1] < 1:
        raise ValueError("mechanism trajectories must contain at least one horizon step")

    names = (
        CANONICAL_MECHANISM_ORDER
        if mechanism_names is None
        else tuple(str(name).lower() for name in mechanism_names)
    )
    if names != CANONICAL_MECHANISM_ORDER:
        raise ValueError(
            "mechanism_names must use the canonical order "
            f"{CANONICAL_MECHANISM_ORDER}; got {names}"
        )

    return predictions_array, targets_array, names


def _safe_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return NaN when R² is undefined rather than coercing it to a score."""
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if (
        targets.size < 2
        or not np.all(np.isfinite(predictions))
        or not np.all(np.isfinite(targets))
        or np.all(targets == targets[0])
    ):
        return float("nan")
    return float(r2_score(targets, predictions))


def _summary(predictions: np.ndarray, targets: np.ndarray, include_r2: bool) -> dict[str, float]:
    prediction_values = np.asarray(predictions).reshape(-1)
    target_values = np.asarray(targets).reshape(-1)
    metrics = {
        "mae": float(mean_absolute_error(target_values, prediction_values)),
        "rmse": float(np.sqrt(mean_squared_error(target_values, prediction_values))),
    }
    if include_r2:
        metrics["r2"] = _safe_r2(prediction_values, target_values)
    return metrics


def compute_mechanism_trajectory_metrics(
    predictions: Any,
    targets: Any,
    mechanism_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate and axis-aware metrics for ``[M, H, 3]`` trajectories.

    The final axis is permanently ``(nbti, hci, tddb)``. Aggregate values use
    the legacy flatten-all-elements reduction, while detailed reductions retain
    horizon and mechanism axes until their respective calculations.
    """
    predictions_array, targets_array, names = _validate_inputs(
        predictions, targets, mechanism_names
    )

    per_mechanism = {
        name: _summary(predictions_array[:, :, index], targets_array[:, :, index], True)
        for index, name in enumerate(names)
    }
    per_horizon = [
        {
            "horizon": horizon_index + 1,
            **_summary(
                predictions_array[:, horizon_index, :],
                targets_array[:, horizon_index, :],
                False,
            ),
        }
        for horizon_index in range(predictions_array.shape[1])
    ]

    absolute_error = np.abs(predictions_array - targets_array)
    squared_error = np.square(predictions_array - targets_array)

    return {
        "aggregate": _summary(predictions_array, targets_array, True),
        "per_mechanism": per_mechanism,
        "per_horizon": per_horizon,
        "mae_by_horizon_mechanism": np.mean(absolute_error, axis=0).tolist(),
        "rmse_by_horizon_mechanism": np.sqrt(
            np.mean(squared_error, axis=0)
        ).tolist(),
    }
