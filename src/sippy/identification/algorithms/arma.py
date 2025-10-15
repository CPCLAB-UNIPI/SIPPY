"""ARMA (AutoRegressive Moving Average) identification algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import IdentificationAlgorithm, StateSpaceModel
from .opt_support import gen_miso_id
from .ararx import _state_space_from_single_result

if TYPE_CHECKING:  # pragma: no cover
    from ..iddata import IDData


class ARMAAlgorithm(IdentificationAlgorithm):
    """Time-series ARMA identification wrapped around shared helpers."""

    def get_algorithm_name(self) -> str:
        return "ARMA"

    def validate_parameters(self, **kwargs) -> bool:  # pragma: no cover - trivial
        na = kwargs.get("na", 1)
        nc = kwargs.get("nc", 1)
        if na <= 0:
            raise ValueError("AR order (na) must be positive")
        if nc <= 0:
            raise ValueError("MA order (nc) must be positive")
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
            sample_time = getattr(iddata, "sample_time", kwargs.get("tsample", 1.0))
        else:
            if y is None:
                raise ValueError("ARMA requires output data")
            sample_time = kwargs.get("tsample", 1.0)

        y = np.atleast_2d(np.asarray(y, dtype=float))
        if y.shape[0] != 1:
            raise ValueError("ARMA is defined for single-output (time-series) data")

        na = int(kwargs.get("na", 1))
        nc = int(kwargs.get("nc", 1))
        max_iterations = kwargs.get("max_iterations", 200)
        stability_margin = kwargs.get("stability_margin", kwargs.get("stab_marg", 1.0))
        enforce_stability = kwargs.get("stability_constraint", kwargs.get("stab_cons", False))

        empty_input = np.zeros((0, y.shape[1]))

        try:
            result = gen_miso_id(
                id_method="ARMA",
                y=y[0],
                u=empty_input,
                na=na,
                nb=[],
                nc=nc,
                nd=0,
                nf=0,
                theta=[],
                max_iterations=max_iterations,
                stability_margin=stability_margin,
                enforce_stability=enforce_stability,
            )
        except RuntimeError as exc:
            raise RuntimeError("CasADi is required for ARMA NLP identification") from exc

        return _state_space_from_single_result(result, nu=0, sample_time=sample_time)
