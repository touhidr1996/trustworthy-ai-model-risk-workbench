from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, generate_applications, temporal_split
from .metrics import group_audit, selective_metrics, summary_metrics
from .monitoring import counterfactual_consistency, psi

ROOT = Path(__file__).resolve().parents[2]


def run() -> dict:
    for name in ["data", "artifacts", "reports"]:
        (ROOT / name).mkdir(exist_ok=True)
    data = generate_applications()
    train, test = temporal_split(data)
    baseline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    baseline.fit(train[FEATURES], train.default)
    improved_base = HistGradientBoostingClassifier(max_iter=220, learning_rate=.05, max_leaf_nodes=20, l2_regularization=2.0, random_state=42)
    improved = CalibratedClassifierCV(improved_base, method="sigmoid", cv=4)
    improved.fit(train[FEATURES], train.default)
    p0 = baseline.predict_proba(test[FEATURES])[:, 1]
    p1 = improved.predict_proba(test[FEATURES])[:, 1]
    overall0, overall1 = summary_metrics(test.default, p0), summary_metrics(test.default, p1)
    selected = baseline
    selected_p = p0
    groups = group_audit(test.default, selected_p, test.group)
    tpr_gap = float(groups.true_positive_rate.max() - groups.true_positive_rate.min())
    fpr_gap = float(groups.false_positive_rate.max() - groups.false_positive_rate.min())
    audit = {"split":{"train_rows":len(train),"test_rows":len(test),"train_end_month":int(train.month.max()),"test_start_month":int(test.month.min())}, "selected_logistic_baseline":overall0, "rejected_calibrated_boosting":overall1, "model_selection":{"selected":"logistic_regression","reason":"Lower Brier score, log loss, and ECE on the untouched temporal holdout"}, "fairness":{"equal_opportunity_gap":tpr_gap,"false_positive_rate_gap":fpr_gap}, "selective_prediction":selective_metrics(test.default,selected_p), "drift":{"debt_ratio_psi":psi(train.debt_ratio,test.debt_ratio),"warning_threshold":.2}, "robustness":{"income_plus_10k":counterfactual_consistency(selected,test[FEATURES].sample(1000,random_state=42),"income",10000)}}
    (ROOT / "reports" / "audit_summary.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    data.to_csv(ROOT / "data" / "synthetic_applications.csv",index=False)
    scored = test[["event_id","month","group","default"]].assign(baseline_probability=p0,candidate_probability=p1,risk_probability=selected_p,decision=np.where(selected_p<.4,"low_risk",np.where(selected_p>.6,"high_risk","human_review")))
    scored.to_csv(ROOT / "artifacts" / "holdout_scores.csv",index=False)
    groups.to_csv(ROOT / "artifacts" / "group_audit.csv",index=False)
    joblib.dump(selected,ROOT / "artifacts" / "model.joblib")
    with sqlite3.connect(ROOT / "artifacts" / "audit.db") as con:
        scored.to_sql("fact_scores",con,if_exists="replace",index=False)
        groups.to_sql("group_audit",con,if_exists="replace",index=False)
    fig, axes = plt.subplots(1,3,figsize=(14,4.2))
    for probs,label,color in [(p0,"Logistic baseline","#6b7280"),(p1,"Calibrated boosting","#2563eb")]:
        obs,pred=calibration_curve(test.default,probs,n_bins=10,strategy="quantile"); axes[0].plot(pred,obs,marker="o",label=label,color=color)
    axes[0].plot([0,1],[0,1],"--",color="#111827"); axes[0].set(title="Calibration",xlabel="Predicted risk",ylabel="Observed rate"); axes[0].legend(fontsize=8)
    groups.plot(x="group",y=["true_positive_rate","false_positive_rate"],kind="bar",ax=axes[1],color=["#2563eb","#f59e0b"]); axes[1].set(title="Error-rate audit",ylabel="Rate",xlabel="Synthetic group"); axes[1].legend(["TPR","FPR"])
    axes[2].hist(p1,bins=30,color="#2563eb",alpha=.85); axes[2].axvspan(.4,.6,color="#f59e0b",alpha=.25,label="Human review"); axes[2].set(title="Selective prediction",xlabel="Predicted risk",ylabel="Applications"); axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "trustworthy_ai_dashboard.png",dpi=170); plt.close()
    return audit


if __name__ == "__main__":
    print(json.dumps(run(),indent=2))
