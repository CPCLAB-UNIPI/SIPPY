"""
Created on Wed Jul 26 2017

@author: Giuseppe Armenise
@updates: RBdC & MV
"""

from itertools import product
from typing import Literal
from warnings import warn

import control.matlab as cnt
import numpy as np

from .arx import ARX_id
from .functionset import information_criterion, rescale
from .io_opt import GEN_id
from .io_rls import GEN_RLS_id


def get_val_range(order_range: int | tuple[int, int]):
    if isinstance(order_range, int):
        order_range = (order_range, order_range + 1)
    min_val, max_val = order_range
    if min_val < 0:
        raise ValueError("Minimum value must be non-negative")
    return range(min_val, max_val + 1)


class SS_Model:
    def __init__(
        self,
        A,
        B,
        C,
        D,
        K,
        ts,
        Vn,
        x0=None,
        Q=None,
        R=None,
        S=None,
        A_K=None,
        B_K=None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Vn = Vn
        self.Q = Q  # not in parsim
        self.R = R  # not in parsim
        self.S = S  # not in parsim
        self.K = K

        self.n = A[:, 0].size
        self.G = cnt.ss(A, B, C, D, ts)
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((A[:, 0].size, 1))

        try:
            self.A_K = A - np.dot(K, C) if A_K is None else A_K
            self.B_K = B - np.dot(K, D) if B_K is None else B_K
        # TODO: find out why error is not raised
        except ValueError:
            self.A_K = []
            self.B_K = []

    @classmethod
    def _from_order(
        cls,
        y,
        u,
        f=20,
        weights="N4SID",
        method="AIC",
        orders=[1, 10],
        D_required=False,
        A_stability=False,
    ):
        pass


class IO_SISO_Model:
    def __init__(
        self,
        na: None | int,
        nb: None | int,
        nc: None | int,
        nd: None | int,
        nf: None | int,
        theta: None | int,
        ts: float,
        NUM: np.ndarray,
        DEN: np.ndarray,
        G: cnt.TransferFunction,
        H: cnt.TransferFunction,
        Vn,
        Yid,
        **kwargs,
    ):
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.ts = ts
        self.NUM = NUM
        self.DEN = DEN
        self.G = G
        self.H = H
        self.Vn = Vn
        self.Yid = Yid
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _from_order(
        cls,
        flag: Literal["arx", "rls", "opt"],
        id_method: Literal[
            "BJ",
            "GEN",
            "ARARX",
            "ARARMAX",
            "ARMA",
            "ARMAX",
            "ARX",
            "OE",
            "FIR",
        ],
        y,
        u,
        ts: float = 1.,
        na_ord: int | tuple[int, int] = (0, 5),
        nb_ord: int | tuple[int, int] = (1, 5),
        nc_ord: int | tuple[int, int] = (0, 5),
        nd_ord: int | tuple[int, int] = (0, 5),
        nf_ord: int | tuple[int, int] = (0, 5),
        delays: int | tuple[int, int] = (0, 5),
        ic_method: Literal["AIC", "AICc", "BIC"] = "AIC",
        max_iter: int = 200,
        st_m: float = 1.0,
        st_c: bool = False,
    ):
        if y.size != u.size:
            raise RuntimeError("y and u must have the same length")

        # order ranges
        na_range = get_val_range(na_ord)
        nb_range = get_val_range(nb_ord)
        nc_range = get_val_range(nc_ord)
        nd_range = get_val_range(nd_ord)
        nf_range = get_val_range(nf_ord)
        theta_range = get_val_range(delays)

        if nb_range[0] <= 0:
            raise ValueError(
                f"Lower bound of nb must be strictly positive integer. Got {nb_range[0]}"
            )

        ystd, y = rescale(y)
        Ustd, u = rescale(u)
        IC_old = np.inf

        na, nb, nc, nd, nf, theta = (
            na_range[1],
            nb_range[1],
            nc_range[1],
            nd_range[1],
            nf_range[1],
            theta_range[1],
        )
        for i_a, i_b, i_c, i_d, i_f, i_t in product(
            na_range, nb_range, nc_range, nd_range, nf_range, theta_range
        ):
            if flag == "opt":
                _, _, _, _, Vn, y_id = GEN_id(
                    id_method,
                    y,
                    u,
                    i_a,
                    i_b,
                    i_c,
                    i_d,
                    i_f,
                    i_t,
                    max_iter,
                    st_m,
                    st_c,
                )
            elif flag == "rls":
                _, _, _, _, Vn, y_id = GEN_RLS_id(
                    id_method,
                    y,
                    u,
                    i_a,
                    i_b,
                    i_c,
                    i_d,
                    i_f,
                    i_t,
                )
            elif flag == "arx":
                _, _, _, _, Vn, y_id = ARX_id(y, u, i_a, i_b, i_t)
            IC = information_criterion(
                i_a + i_b + i_c + i_d + i_f,
                y.size - max(i_a, i_b + i_t, i_c, i_d, i_f),
                Vn * 2,
                ic_method,
            )
            if IC < IC_old:
                IC_old = IC
                (
                    na,
                    nb,
                    nc,
                    nd,
                    nf,
                    theta,
                ) = i_a, i_b, i_c, i_d, i_f, i_t

        # rerun identification
        if flag == "opt":
            NUM, DEN, NUMH, DENH, Vn, y_id = GEN_id(
                id_method,
                y,
                u,
                na,
                nb,
                nc,
                nd,
                nf,
                theta,
                max_iter,
                st_m,
                st_c,
            )
        elif flag == "rls":
            NUM, DEN, NUMH, DENH, Vn, y_id = GEN_RLS_id(
                id_method,
                y,
                u,
                na,
                nb,
                nc,
                nd,
                nf,
                theta,
            )
        elif flag == "arx":
            NUM, DEN, NUMH, DENH, Vn, y_id = ARX_id(y, u, na, nb, theta)

        Yid = np.atleast_2d(y_id) * ystd

        # rescale NUM coeff
        if id_method != "ARMA" and nb and theta:
            NUM[theta : nb + theta] = NUM[theta : nb + theta] * ystd / Ustd

        # FdT
        G = cnt.tf(NUM, DEN, ts)
        H = cnt.tf(NUMH, DENH, ts)

        if G is None or H is None:
            raise RuntimeError("tf could not be created")
        poles_G = max(np.abs(cnt.poles(G)))
        poles_H = max(np.abs(cnt.poles(H)))
        check_st_H = np.zeros(1) if id_method == "OE" else poles_H
        if poles_G > 1.0 or check_st_H > 1.0:
            warn("One of the identified system is not stable")
            if st_c is True:
                raise RuntimeError(f"Infeasible solution: the stability constraint has been violated, since the maximum pole is {max(poles_H, poles_G)} \
                        ... against the imposed stability margin {st_m}")
            else:
                warn(
                    f"Consider activating the stability constraint. The maximum pole is {max(poles_H, poles_G)}  "
                )

        return cls(na, nb, nc, nd, nf, theta, ts, NUM, DEN, G, H, Vn, Yid)


class IO_MIMO_Model(IO_SISO_Model):
    def __init__(
        self,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        ts,
        numerator,
        denominator,
        numerator_H,
        denominator_H,
        G,
        H,
        Vn,
        Yid,
        **kwargs,
    ):
        super().__init__(
            na,
            nb,
            nc,
            nd,
            nf,
            theta,
            ts,
            numerator,
            denominator,
            G,
            H,
            Vn,
            Yid,
            **kwargs,
        )
        self.numerator_H = numerator_H
        self.denominator_H = denominator_H
