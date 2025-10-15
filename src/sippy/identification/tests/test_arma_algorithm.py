import numpy as np
import pytest

from sippy.identification.algorithms.arma import ARMAAlgorithm
from sippy.identification.base import StateSpaceModel


def _simulate_arma(na, nc, n_samples=400):
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=0.2, size=n_samples)
    y = np.zeros(n_samples)

    warmup = max(na, nc, 1) + 1
    for k in range(warmup, n_samples):
        ar_part = 0.0
        if na:
            idx = k - np.arange(1, na + 1)
            ar_part = np.dot(y[idx], 0.3 * np.ones(na))
        ma_part = 0.0
        if nc:
            idx = k - np.arange(1, nc + 1)
            ma_part = np.dot(noise[idx], 0.4 * np.ones(nc))
        y[k] = -ar_part + ma_part + noise[k]

    return y.reshape(1, -1)


def test_arma_identification_runs():
    y = _simulate_arma(na=2, nc=2)
    algo = ARMAAlgorithm()
    model = algo.identify(y=y, na=2, nc=2, tsample=1.0, max_iterations=10)

    assert isinstance(model, StateSpaceModel)
    assert model.A.shape[0] == model.A.shape[1]
    assert model.B.shape[1] == 0
    assert model.Yid.shape == y.shape


@pytest.mark.parametrize("orders", [(1, 1), (2, 1)])
def test_arma_order_variants(orders):
    na, nc = orders
    y = _simulate_arma(na=na, nc=nc)
    algo = ARMAAlgorithm()
    model = algo.identify(y=y, na=na, nc=nc, tsample=1.0, max_iterations=5)

    assert model.C.shape[0] == 1
    assert model.A.shape[0] >= max(na, 1)
