"""Payoff-cutoff arithmetic and row-alignment checks without model training."""

import numpy as np
import pandas as pd

from members.zhenci.e7.evaluate_e7 import apply_payoff_cutoffs, fit_payoff_cutoffs


def test_nonup_flat_day_counts_as_zero_loss() -> None:
    train = pd.DataFrame({
        "stock_id": ["A", "A", "B"],
        "etl:adj_open": [100.0, 100.0, 100.0],
        "etl:adj_close": [110.0, 95.0, 100.0],
    })
    fitted, details = fit_payoff_cutoffs(train, pd.Series([0.01, -0.01, np.nan]), "toy")
    # The non-up loss mean is (0.05 + 0) / 2 = 0.025.
    assert np.isclose(fitted["pooled"], 0.025 / (0.1 + 0.025))
    assert details.loc[details.scope.eq("pooled"), "n_nonup"].iloc[0] == 2
    labels = apply_payoff_cutoffs(
        np.array([0.19, 0.21]), pd.Series(["A", "B"]), pd.Series([0.01, np.nan]),
        fitted, "pooled_payoff", {},
    )
    assert labels.tolist() == [0, 1]


def test_stock_cutoff_requires_training_estimate() -> None:
    train = pd.DataFrame({
        "stock_id": ["A", "A"],
        "etl:adj_open": [100.0, 100.0],
        "etl:adj_close": [110.0, 90.0],
    })
    fitted, _ = fit_payoff_cutoffs(train, pd.Series([0.01, -0.01]), "toy")
    try:
        apply_payoff_cutoffs(
            np.array([0.5]), pd.Series(["B"]), pd.Series([0.0]),
            fitted, "stock_payoff_shrunk_100", {},
        )
    except ValueError as exc:
        assert "no training payoff estimate" in str(exc)
    else:
        raise AssertionError("A stock absent from training must not be assigned a cutoff")
