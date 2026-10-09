"""Tests for Population Stability Index monitoring."""

import numpy as np

from mena_mlops.monitoring.psi import population_stability_index


def test_identical_distributions_have_zero_psi() -> None:
    values = np.arange(100, dtype=float)
    assert population_stability_index(values, values) == 0.0


def test_shifted_distribution_has_positive_psi() -> None:
    reference = np.arange(100, dtype=float)
    current = reference + 100
    assert population_stability_index(reference, current) > 0.25
