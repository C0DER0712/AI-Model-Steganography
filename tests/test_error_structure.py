import numpy as np
import torch

from scripts.evaluate import _error_pattern_metrics


def _metric_for_errors(errors: np.ndarray) -> dict[str, object]:
    original = torch.zeros(errors.size, dtype=torch.uint8)
    recovered = torch.from_numpy(errors.astype(np.uint8))
    return _error_pattern_metrics([original], [recovered])


def test_error_pattern_metrics_matches_iid_reference_for_random_errors():
    rng = np.random.default_rng(7)
    errors = (rng.random(100_000) < 0.15).astype(np.uint8)

    metrics = _metric_for_errors(errors)

    assert abs(metrics["adjacent_error_pair_rate"] - 0.15**2) < 0.002
    assert abs(metrics["probability_next_bit_is_error_given_error"] - 0.15) < 0.02
    run_metrics = metrics["error_run_lengths"]
    assert abs(run_metrics["mean"] - 1 / (1 - 0.15)) < 0.05


def test_error_pattern_metrics_flags_bursty_errors():
    errors = np.zeros(1_000, dtype=np.uint8)
    for start in range(0, 1_000, 100):
        errors[start:start + 10] = 1

    metrics = _metric_for_errors(errors)

    assert metrics["probability_next_bit_is_error_given_error"] > 0.8
    assert metrics["adjacent_error_pair_rate_excess_over_iid"] > 0.05
    assert metrics["error_run_lengths"]["mean"] == 10


def test_error_pattern_metrics_handles_zero_error_payload():
    metrics = _metric_for_errors(np.zeros(20, dtype=np.uint8))

    assert metrics["adjacent_error_pair_rate"] == 0.0
    assert metrics["error_run_lengths"]["count"] == 0
    assert metrics["per_position_error_rate_across_payload_runs"]["max"] == 0.0


def test_error_pattern_metrics_reports_repeatable_weak_position_bins():
    payloads = [torch.zeros(512, dtype=torch.uint8) for _ in range(8)]
    recovered = [payload.clone() for payload in payloads]
    for run in recovered:
        run[10:12] = 1

    metrics = _error_pattern_metrics(payloads, recovered)
    positions = metrics["per_position_error_rate_across_payload_runs"]

    assert positions["bin_error_rates"][5] == 1.0
    assert positions["split_half_stability"]["pearson_correlation_of_bin_rates"] > 0.99
    assert positions["split_half_stability"]["top_decile_bin_overlap_fraction"] == 1.0
