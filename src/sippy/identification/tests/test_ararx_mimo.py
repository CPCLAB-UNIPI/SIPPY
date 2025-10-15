import numpy as np
import pytest

from sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy.identification.base import StateSpaceModel


def _simulate_ararx_system(na, nb, nd, theta, n_samples=300):
    rng = np.random.default_rng(42)
    ny, nu = nb.shape
    u = rng.normal(scale=0.3, size=(nu, n_samples))
    y = np.zeros((ny, n_samples))
    w = np.zeros((ny, n_samples))
    e = rng.normal(scale=0.05, size=(ny, n_samples))

    # Coefficient templates per Section 4.1-like structure
    A = [0.2] * ny
    D = [0.2] * ny
    B = np.full((ny, nu), 0.4)

    warmup = int(max(np.max(na), np.max(nb + theta), np.max(nd), 1)) + 2
    for k in range(warmup, n_samples):
        for i in range(ny):
            s = 0.0
            for j in range(nu):
                order = nb[i, j]
                if order == 0:
                    continue
                idx = k - theta[i, j] - np.arange(1, order + 1)
                s += B[i, j] * np.sum(u[j, idx])

            w[i, k] = s
            for r in range(nd[i]):
                w[i, k] -= D[i] * w[i, k - (r + 1)]

            y[i, k] = w[i, k] + e[i, k]
            for r in range(na[i]):
                y[i, k] -= A[i] * y[i, k - (r + 1)]

    return y, u


def test_ararx_siso_runs():
    y, u = _simulate_ararx_system(
        na=np.array([2]),
        nb=np.array([[2]]),
        nd=np.array([1]),
        theta=np.array([[1]]),
    )

    algo = ARARXAlgorithm()
    model = algo.identify(
        y=y,
        u=u,
        na=2,
        nb=2,
        nd=1,
        theta=1,
        tsample=1.0,
        max_iterations=10,
    )

    assert isinstance(model, StateSpaceModel)
    assert model.A.shape[0] == model.A.shape[1]
    assert model.B.shape[1] == u.shape[0]
    assert model.Yid.shape == y.shape


def test_ararx_mimo_runs():
    na = np.array([2, 2])
    nb = np.array([[2, 1], [1, 2]])
    nd = np.array([1, 1])
    theta = np.zeros_like(nb)

    y, u = _simulate_ararx_system(na=na, nb=nb, nd=nd, theta=theta)

    algo = ARARXAlgorithm()
    model = algo.identify(
        y=y,
        u=u,
        na=na.tolist(),
        nb=nb.tolist(),
        nd=nd.tolist(),
        theta=theta.tolist(),
        tsample=1.0,
        max_iterations=10,
    )

    assert isinstance(model, StateSpaceModel)
    assert model.A.shape[0] == model.A.shape[1]
    assert model.B.shape[1] == u.shape[0]
    assert model.C.shape[0] == y.shape[0]
    assert model.Yid.shape == y.shape


@pytest.mark.parametrize("theta", [0, 1])
def test_ararx_delay_handling(theta):
    y, u = _simulate_ararx_system(
        na=np.array([1]),
        nb=np.array([[1]]),
        nd=np.array([1]),
        theta=np.array([[theta]]),
    )

    algo = ARARXAlgorithm()
    model = algo.identify(
        y=y,
        u=u,
        na=1,
        nb=1,
        nd=1,
        theta=theta,
        tsample=1.0,
        max_iterations=5,
    )

    assert model.Yid.shape == y.shape
