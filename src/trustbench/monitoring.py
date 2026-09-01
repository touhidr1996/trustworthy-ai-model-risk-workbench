from __future__ import annotations

import numpy as np


def psi(reference, current, bins: int = 10) -> float:
    reference, current = np.asarray(reference), np.asarray(current)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    b = np.clip(np.histogram(current, edges)[0] / len(current), 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def counterfactual_consistency(model, X, feature: str, delta: float) -> dict[str, float]:
    base = model.predict_proba(X)[:, 1]
    changed = X.copy()
    changed[feature] = np.maximum(0, changed[feature] + delta)
    shifted = model.predict_proba(changed)[:, 1]
    return {"mean_absolute_probability_change": float(np.mean(np.abs(shifted - base))), "p95_absolute_probability_change": float(np.quantile(np.abs(shifted - base), 0.95))}
