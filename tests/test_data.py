import pandas as pd
from trustbench.data import generate_applications, temporal_split


def test_generator_reproducible():
    pd.testing.assert_frame_equal(generate_applications(100,7),generate_applications(100,7))


def test_temporal_split_has_no_overlap():
    train,test=temporal_split(generate_applications(2000))
    assert train.month.max()<test.month.min()
