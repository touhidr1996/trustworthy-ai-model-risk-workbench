import numpy as np
from trustbench.metrics import expected_calibration_error, group_audit, selective_metrics
from trustbench.monitoring import psi


def test_perfect_calibration_has_small_error():
    p=np.tile([.1,.3,.7,.9],1000); y=np.random.default_rng(42).binomial(1,p)
    assert expected_calibration_error(y,p)<.04


def test_group_audit_returns_both_groups():
    out=group_audit([0,1,0,1],[.1,.8,.6,.4],["a","a","b","b"])
    assert set(out.group)=={"a","b"}
    assert out.filter(regex="rate").to_numpy().min()>=0


def test_selective_prediction_accounting():
    result=selective_metrics([0,0,1,1],[.1,.45,.55,.9])
    assert abs(result["coverage"]+result["review_rate"]-1)<1e-12


def test_psi_detects_shift():
    rng=np.random.default_rng(5); ref=rng.normal(size=3000)
    assert psi(ref,ref)<1e-10
    assert psi(ref,ref+1)>.2
