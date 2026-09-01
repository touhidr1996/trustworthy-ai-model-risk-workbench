from __future__ import annotations

import numpy as np
import pandas as pd

from . import SEED


def generate_applications(n: int = 12000, seed: int = SEED) -> pd.DataFrame:
    """Create a non-sensitive, fictional application stream with documented bias mechanisms."""
    rng = np.random.default_rng(seed)
    month = np.arange(n) // 500
    group = rng.choice(["group_a", "group_b"], size=n, p=[0.58, 0.42])
    income = rng.lognormal(10.45 + 0.012 * month, 0.42, n)
    debt_ratio = np.clip(rng.beta(2.2, 5.0, n) + 0.035 * (month >= 18), 0, 0.95)
    history_months = np.clip(rng.gamma(4, 18, n), 1, 360)
    late_payments = rng.poisson(0.55 + 1.8 * debt_ratio)
    requested_amount = rng.lognormal(9.0, 0.5, n)
    employment_years = np.clip(rng.gamma(2.5, 2.4, n), 0, 35)
    latent = -1.6 + 3.4 * debt_ratio + 0.24 * late_payments - 0.004 * history_months
    latent += 0.000025 * requested_amount - 0.000010 * income - 0.06 * employment_years
    latent += rng.normal(0, 0.65, n)
    default = rng.binomial(1, 1 / (1 + np.exp(-latent)))
    return pd.DataFrame({"event_id": [f"sim-{i:06d}" for i in range(n)], "month": month, "group": group, "income": income.round(2), "debt_ratio": debt_ratio.round(4), "history_months": history_months.round(1), "late_payments": late_payments, "requested_amount": requested_amount.round(2), "employment_years": employment_years.round(1), "default": default})


FEATURES = ["income", "debt_ratio", "history_months", "late_payments", "requested_amount", "employment_years"]


def temporal_split(df: pd.DataFrame, fraction: float = 0.25):
    cutoff = int(df["month"].quantile(1 - fraction))
    train, test = df[df.month < cutoff].copy(), df[df.month >= cutoff].copy()
    assert train.month.max() < test.month.min()
    return train, test
