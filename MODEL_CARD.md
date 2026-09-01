# Model card

## Intended use

Demonstrate a model-risk workflow on synthetic data: temporal validation, calibration, group metrics, selective prediction, perturbation testing, drift monitoring, and audit artifacts. It is not a lending model and has no real users or production deployment.

## Models

The baseline is standardized logistic regression. A histogram-gradient-boosting candidate uses four-fold sigmoid calibration. The complex candidate performs worse on the untouched temporal holdout, so the workbench rejects it and serializes logistic regression. Protected-group labels are excluded from model features and used only for auditing. This exclusion alone does not guarantee fairness because other fields can act as proxies.

## Evaluation

The latest simulated months form an untouched temporal holdout. ROC-AUC measures ranking; Brier score, log loss, and ECE measure probability quality. TPR/FPR gaps describe group error disparity. A 0.40–0.60 review band reports coverage and decided-case error rather than silently automating uncertain cases.

## Limitations and governance

Synthetic labels cannot validate fairness in real populations. Group metrics can hide intersectional harms, thresholds encode value judgments, and calibration can decay. Real deployment would require legal review, adverse-action explanations, appeal and correction channels, accessibility testing, outcome monitoring, data-quality contracts, independent validation, security controls, and human accountability.
