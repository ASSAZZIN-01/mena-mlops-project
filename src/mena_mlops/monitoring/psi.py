"""Population Stability Index (PSI) calculations for monitoring features."""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI using reference quantile bins and stable proportions."""

    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("PSI requires non-empty reference and current values")
    if bins < 2:
        raise ValueError("PSI requires at least two bins")
    edges = np.unique(
        np.quantile(reference_values, np.linspace(0, 1, bins + 1))
    )
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference_values, bins=edges)[0]
    current_counts = np.histogram(current_values, bins=edges)[0]
    reference_share = np.maximum(
        reference_counts / reference_values.size, epsilon
    )
    current_share = np.maximum(current_counts / current_values.size, epsilon)
    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )
