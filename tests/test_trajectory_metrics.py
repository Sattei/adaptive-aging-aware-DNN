import math

import numpy as np
import pytest

from evaluation.trajectory_metrics import compute_mechanism_trajectory_metrics


def test_perfect_prediction_has_zero_error_on_every_axis():
    targets = np.arange(24, dtype=np.float64).reshape(4, 2, 3) / 24.0
    metrics = compute_mechanism_trajectory_metrics(targets.copy(), targets)

    assert metrics["aggregate"]["mae"] == 0.0
    assert metrics["aggregate"]["rmse"] == 0.0
    assert metrics["aggregate"]["r2"] == pytest.approx(1.0)
    for mechanism_metrics in metrics["per_mechanism"].values():
        assert mechanism_metrics["mae"] == 0.0
        assert mechanism_metrics["rmse"] == 0.0
        assert mechanism_metrics["r2"] == pytest.approx(1.0)
    assert all(item["mae"] == 0.0 and item["rmse"] == 0.0 for item in metrics["per_horizon"])
    assert np.array_equal(metrics["mae_by_horizon_mechanism"], np.zeros((2, 3)))
    assert np.array_equal(metrics["rmse_by_horizon_mechanism"], np.zeros((2, 3)))


def test_single_hci_error_preserves_mechanism_axis_order():
    targets = np.zeros((2, 3, 3), dtype=np.float64)
    predictions = targets.copy()
    predictions[:, :, 1] = 0.25

    metrics = compute_mechanism_trajectory_metrics(predictions, targets)

    assert metrics["per_mechanism"]["nbti"]["mae"] == 0.0
    assert metrics["per_mechanism"]["hci"]["mae"] == pytest.approx(0.25)
    assert metrics["per_mechanism"]["tddb"]["mae"] == 0.0
    assert np.array_equal(
        np.asarray(metrics["mae_by_horizon_mechanism"]),
        np.array([[0.0, 0.25, 0.0]] * 3),
    )


def test_single_horizon_error_preserves_horizon_axis():
    targets = np.zeros((3, 3, 3), dtype=np.float64)
    predictions = targets.copy()
    predictions[:, 1, :] = 0.5

    metrics = compute_mechanism_trajectory_metrics(predictions, targets)

    assert [item["horizon"] for item in metrics["per_horizon"]] == [1, 2, 3]
    assert [item["mae"] for item in metrics["per_horizon"]] == [0.0, 0.5, 0.0]
    assert np.array_equal(
        np.asarray(metrics["mae_by_horizon_mechanism"]),
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]),
    )


def test_known_metric_values_are_reduced_over_the_correct_axes():
    targets = np.zeros((1, 2, 3), dtype=np.float64)
    predictions = np.array([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])

    metrics = compute_mechanism_trajectory_metrics(predictions, targets)

    assert metrics["aggregate"]["mae"] == pytest.approx(2.5)
    assert metrics["aggregate"]["rmse"] == pytest.approx(math.sqrt(55.0 / 6.0))
    assert [metrics["per_mechanism"][name]["mae"] for name in ("nbti", "hci", "tddb")] == [1.5, 2.5, 3.5]
    assert [item["mae"] for item in metrics["per_horizon"]] == [1.0, 4.0]
    assert np.array_equal(
        np.asarray(metrics["mae_by_horizon_mechanism"]), predictions[0]
    )
    assert np.array_equal(
        np.asarray(metrics["rmse_by_horizon_mechanism"]), predictions[0]
    )


@pytest.mark.parametrize("horizon", [1, 3, 7])
def test_dynamic_horizon_preserves_return_shapes(horizon):
    targets = np.arange(2 * horizon * 3, dtype=np.float64).reshape(2, horizon, 3)
    metrics = compute_mechanism_trajectory_metrics(targets, targets)

    assert len(metrics["per_horizon"]) == horizon
    assert np.asarray(metrics["mae_by_horizon_mechanism"]).shape == (horizon, 3)
    assert np.asarray(metrics["rmse_by_horizon_mechanism"]).shape == (horizon, 3)


@pytest.mark.parametrize(
    ("predictions", "targets", "match"),
    [
        (np.zeros((1, 2, 3)), np.zeros((1, 3, 3)), "identical shapes"),
        (np.zeros((1, 3)), np.zeros((1, 3)), "must have shape"),
        (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), "final dimension must be 3"),
        (np.zeros((0, 1, 3)), np.zeros((0, 1, 3)), "at least one node"),
        (np.zeros((1, 0, 3)), np.zeros((1, 0, 3)), "at least one horizon step"),
    ],
)
def test_shape_validation_rejects_invalid_trajectory_inputs(predictions, targets, match):
    with pytest.raises(ValueError, match=match):
        compute_mechanism_trajectory_metrics(predictions, targets)


def test_constant_targets_return_nan_r2_without_crashing():
    targets = np.full((2, 2, 3), 0.5, dtype=np.float64)
    predictions = np.zeros_like(targets)

    metrics = compute_mechanism_trajectory_metrics(predictions, targets)

    assert math.isnan(metrics["aggregate"]["r2"])
    assert all(math.isnan(item["r2"]) for item in metrics["per_mechanism"].values())


def test_metrics_are_invariant_to_artificial_node_batch_boundaries():
    first_predictions = np.array([[[0.0, 0.1, 0.2]], [[0.3, 0.4, 0.5]]])
    first_targets = first_predictions * 0.1
    second_predictions = np.array([[[0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1]]])
    second_targets = second_predictions * 0.1

    predictions_from_batches = np.concatenate([first_predictions, second_predictions], axis=0)
    targets_from_batches = np.concatenate([first_targets, second_targets], axis=0)
    equivalent_combined_predictions = np.vstack([first_predictions, second_predictions])
    equivalent_combined_targets = np.vstack([first_targets, second_targets])

    metrics_from_batches = compute_mechanism_trajectory_metrics(
        predictions_from_batches, targets_from_batches
    )
    metrics_from_combined = compute_mechanism_trajectory_metrics(
        equivalent_combined_predictions, equivalent_combined_targets
    )

    assert metrics_from_batches == metrics_from_combined
