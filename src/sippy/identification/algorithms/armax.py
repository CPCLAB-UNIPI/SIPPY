"""ARMAX (AutoRegressive Moving Average with eXogenous inputs)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import IdentificationAlgorithm, StateSpaceModel
from .opt_support import (
    gen_mimo_id,
    gen_miso_id,
    armax_mimo_id,
    armax_miso_id,
)
from .ararx import (
    _normalize_matrix,
    _normalize_orders,
    _state_space_from_results,
    _state_space_from_single_result,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..iddata import IDData


class ARMAXAlgorithm(IdentificationAlgorithm):
    """ARMAX identification with master-compatible solvers."""

    def __init__(self, mode: str = "NLP") -> None:
        super().__init__()
        self.default_mode = mode.upper()

    def get_algorithm_name(self) -> str:
        return "ARMAX"

    def validate_parameters(self, **kwargs) -> bool:  # pragma: no cover - simple validation
        for name in ("na", "nb", "nc"):
            if name not in kwargs:
                continue
            value = kwargs[name]
            if isinstance(value, (int, np.integer)):
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
            else:
                arr = np.asarray(value, dtype=int)
                if np.any(arr <= 0):
                    raise ValueError(f"{name} entries must be positive")
        theta = kwargs.get("theta", kwargs.get("nk", 0))
        if isinstance(theta, (int, np.integer)):
            if theta < 0:
                raise ValueError("theta must be non-negative")
        else:
            arr = np.asarray(theta, dtype=int)
            if np.any(arr < 0):
                raise ValueError("theta entries must be non-negative")
        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        if iddata is not None:
            if y is not None or u is not None:
                raise ValueError("Provide either iddata or (y, u), not both")
            y = iddata.get_output_array()
            u = iddata.get_input_array()
            sample_time = getattr(iddata, "sample_time", kwargs.get("tsample", 1.0))
        else:
            if y is None or u is None:
                raise ValueError("ARMAX requires both input and output data")
            sample_time = kwargs.get("tsample", 1.0)

        y = np.atleast_2d(np.asarray(y, dtype=float))
        u = np.atleast_2d(np.asarray(u, dtype=float))

        ny, n_samples = y.shape
        nu, _ = u.shape
        if u.shape[1] != n_samples:
            raise ValueError("Input and output must share the same number of samples")

        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nc = kwargs.get("nc", 1)
        theta = kwargs.get("theta", kwargs.get("nk", 0))

        max_iterations = kwargs.get("max_iterations", 200)
        mode = kwargs.get("mode", kwargs.get("algorithm", self.default_mode)).upper()
        if mode == "RLLS":
            mode = "ILLS"
        if mode == "OPT":
            mode = "NLP"
        stability_margin = kwargs.get("stability_margin", kwargs.get("stab_marg", 1.0))
        enforce_stability = kwargs.get("stability_constraint", kwargs.get("stab_cons", False))

        if ny == 1:
            nb_vec = _normalize_matrix(nb, 1, nu, allow_zero=False).ravel()
            theta_vec = _normalize_matrix(theta, 1, nu, allow_zero=True).ravel()
            na_val = int(np.squeeze(na))
            nc_val = int(np.squeeze(nc))

            if mode == "ILLS":
                result = armax_miso_id(
                    y=y[0],
                    u=u,
                    na=na_val,
                    nb=nb_vec,
                    nc=nc_val,
                    theta=theta_vec,
                    max_iterations=max_iterations,
                )
            else:
                try:
                    result = gen_miso_id(
                        id_method="ARMAX",
                        y=y[0],
                        u=u,
                        na=na_val,
                        nb=nb_vec,
                        nc=nc_val,
                        nd=0,
                        nf=0,
                        theta=theta_vec,
                        max_iterations=max_iterations,
                        stability_margin=stability_margin,
                        enforce_stability=enforce_stability,
                    )
                except RuntimeError as exc:
                    raise RuntimeError("CasADi is required for ARMAX NLP identification") from exc
            return _state_space_from_single_result(result, nu, sample_time)

        na_vec = _normalize_orders(na, ny)
        nc_vec = _normalize_orders(nc, ny)
        nb_matrix = _normalize_matrix(nb, ny, nu, allow_zero=False)
        theta_matrix = _normalize_matrix(theta, ny, nu, allow_zero=True)

        if mode == "ILLS":
            results, _ = armax_mimo_id(
                y=y,
                u=u,
                na=na_vec,
                nb=nb_matrix,
                nc=nc_vec,
                theta=theta_matrix,
                sample_time=sample_time,
                max_iterations=max_iterations,
            )
        else:
            try:
                results, _ = gen_mimo_id(
                    id_method="ARMAX",
                    y=y,
                    u=u,
                    na=na_vec,
                    nb=nb_matrix,
                    nc=nc_vec,
                    nd=[0] * ny,
                    nf=[0] * ny,
                    theta=theta_matrix,
                    sample_time=sample_time,
                    max_iterations=max_iterations,
                    stability_margin=stability_margin,
                    enforce_stability=enforce_stability,
                )
            except RuntimeError as exc:
                raise RuntimeError("CasADi is required for ARMAX NLP identification") from exc

        return _state_space_from_results(results, nu, sample_time)
