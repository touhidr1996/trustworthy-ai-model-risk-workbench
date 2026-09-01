from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y, p, bins: int = 10) -> float:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.cut(frame.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    stats = frame.groupby("bin", observed=True).agg(rate=("y", "mean"), confidence=("p", "mean"), n=("y", "size")).dropna()
    return float(((stats.n / len(frame)) * (stats.rate - stats.confidence).abs()).sum())


def group_audit(y, p, group, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    pred = np.asarray(p) >= threshold
    for label in sorted(pd.unique(np.asarray(group))):
        mask = np.asarray(group) == label
        yy, pp, dd = np.asarray(y)[mask], np.asarray(p)[mask], pred[mask]
        positive = yy == 1
        negative = ~positive
        rows.append({"group": label, "n": int(mask.sum()), "selection_rate": float(dd.mean()), "true_positive_rate": float(dd[positive].mean()), "false_positive_rate": float(dd[negative].mean()), "brier": float(brier_score_loss(yy, pp)), "ece": expected_calibration_error(yy, pp)})
    return pd.DataFrame(rows)


def summary_metrics(y, p) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)), "ece": expected_calibration_error(y, p)}


def selective_metrics(y, p, low: float = 0.4, high: float = 0.6) -> dict[str, float]:
    p, y = np.asarray(p), np.asarray(y)
    decided = (p < low) | (p > high)
    pred = p[decided] >= 0.5
    return {"coverage": float(decided.mean()), "review_rate": float(1 - decided.mean()), "decided_error_rate": float(np.mean(pred != y[decided]))}
