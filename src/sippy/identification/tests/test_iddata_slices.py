import numpy as np
import pandas as pd

from ..iddata import IDData


def build_df(n=50):
    idx = pd.date_range("2021-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "u": rng.normal(size=n),
            "y": rng.normal(size=n),
        },
        index=idx,
    )


def test_iddata_with_bad_slice_ffill():
    df = build_df(20)
    slices = {
        "s1": {"type": "bad", "isGlobal": False, "start": 5, "end": 10, "tags": ["u"]}
    }
    idd = IDData(df, ["u"], ["y"], slices=slices)

    # mask should be true for u in range, false for y
    mask = idd.get_bad_mask()
    assert mask.loc[df.index[5:10], "u"].all()
    assert not mask.loc[df.index[5:10], "y"].any()

    # data should be forward-filled and contain no NaNs
    assert not idd.input_data.isna().any().any()
    assert not idd.output_data.isna().any().any()


def test_iddata_with_interpolate_slice():
    df = build_df(20)
    slices = {
        "s1": {
            "type": "interpolate",
            "isGlobal": False,
            "start": 2,
            "end": 4,
            "tags": ["y"],
        }
    }
    idd = IDData(df, ["u"], ["y"], slices=slices)

    # mask should be true for y in range
    mask = idd.get_bad_mask()
    assert mask.loc[df.index[2:4], "y"].all()

    # interpolation should remove NaNs
    assert not idd.output_data.isna().any().any()


def test_drop_masked_any():
    df = build_df(20)
    slices = {"s1": {"type": "bad", "isGlobal": True, "start": 0, "end": 3, "tags": []}}
    idd = IDData(df, ["u"], ["y"], slices=slices)
    dropped = idd.drop_masked(any_col=True)
    # Expect 20-3 rows remaining
    assert dropped.n_samples == 17
