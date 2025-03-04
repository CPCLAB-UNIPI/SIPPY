"""
Created on Fri Jul 28 2017

@author: Giuseppe Armenise
"""

from warnings import warn

import control.matlab as cnt
import numpy as np

from .functionset import mean_square_error, rescale
from .utils import get_val_range


def ARMAX_MISO_id(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    theta: np.ndarray,
    max_iter: int,
):
    nb = np.array(nb)
    theta = np.array(theta)
    u = np.atleast_2d(u)
    ylength = y.size
    y_std, y = rescale(y)
    [udim, ulength] = u.shape
    eps = np.zeros(y.size)
    Reached_max = False
    # checking dimension
    if nb.size != udim:
        raise RuntimeError(
            " nb must be a matrix, whose dimensions must be equal to yxu"
        )
    #        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf,Reached_max
    elif theta.size != udim:
        raise RuntimeError("theta matrix must have yxu dimensions")
    #        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf,Reached_max
    else:
        nbth = nb + theta
        U_std = np.zeros(udim)
        for j in range(udim):
            U_std[j], u[j] = rescale(u[j])
        # TODO: if run as MIMO, this should be fixed
        val = max(na, np.max(nbth), nc)
        # max predictable dimension
        N = ylength - val
        # regressor matrix
        phi = np.zeros(na + np.sum(nb[:]) + nc)
        PHI = np.zeros((N, na + np.sum(nb[:]) + nc))
        for k in range(N):
            phi[0:na] = -y[k + val - 1 :: -1][0:na]
            for nb_i in range(udim):
                phi[
                    na + np.sum(nb[0:nb_i]) : na + np.sum(nb[0 : nb_i + 1])
                ] = u[nb_i, :][val + k - 1 :: -1][
                    theta[nb_i] : nb[nb_i] + theta[nb_i]
                ]
            PHI[k, :] = phi
        Vn = np.inf
        Vn_old = np.inf
        # coefficient vector
        THETA = np.zeros(na + np.sum(nb[:]) + nc)
        ID_THETA = np.identity(THETA.size)
        lambdak = 0.5
        iterations = 0
        while (Vn_old > Vn or iterations == 0) and iterations < max_iter:
            THETA_old = THETA
            Vn_old = Vn
            iterations = iterations + 1
            for i in range(N):
                PHI[i, na + np.sum(nb[:]) : na + np.sum(nb[:]) + nc] = eps[
                    val + i - 1 :: -1
                ][0:nc]
            THETA = np.dot(np.linalg.pinv(PHI), y[val::])
            Vn = (np.linalg.norm(y[val::] - np.dot(PHI, THETA), 2) ** 2) / (
                2 * N
            )
            THETA_new = THETA
            lambdak = 0.5
            while Vn > Vn_old:
                THETA = np.dot(ID_THETA * lambdak, THETA_new) + np.dot(
                    ID_THETA * (1 - lambdak), THETA_old
                )
                # Model Output
                # y_id0 = np.dot(PHI,THETA)
                Vn = (
                    np.linalg.norm(y[val::] - np.dot(PHI, THETA), 2) ** 2
                ) / (2 * N)

                if lambdak < np.finfo(np.float32).eps:
                    THETA = THETA_old
                    Vn = Vn_old
                lambdak = lambdak / 2.0
            eps[val::] = y[val::] - np.dot(PHI, THETA)
            # adding non-identified outputs
            y_id = np.hstack((y[:val], np.dot(PHI, THETA))) * y_std

        if iterations >= max_iter:
            warn("Reached maximum iterations")
            Reached_max = True
        denominator = np.zeros((udim, val + 1))
        numerator_H = np.zeros((1, val + 1))
        numerator_H[0, 0] = 1.0
        numerator_H[0, 1 : nc + 1] = THETA[na + np.sum(nb[:]) : :]
        denominator[:, 0] = np.ones(udim)
        numerator = np.zeros((udim, val))
        for k in range(udim):
            THETA[na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])] = (
                THETA[na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])]
                * y_std
                / U_std[k]
            )
            numerator[k, theta[k] : theta[k] + nb[k]] = THETA[
                na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])
            ]
            denominator[k, 1 : na + 1] = THETA[0:na]
        return (
            numerator,
            denominator,
            numerator,
            denominator,
            Vn,
            y_id,
            Reached_max,
        )


class Armax:
    def __init__(
        self,
        G: cnt.TransferFunction,
        H: cnt.TransferFunction,
        *order_bounds: tuple[int, int],
        Vn,
        Yid,
        method="AIC",
    ):
        """Armax model class.

        The AutoRegressive-Moving-Average with eXogenous inputs model is computed based on a
        recursive lest-square regression between the input data (U) and the measured output data
        (Y). As Y is noisy in practice, a white noise (E) is identified within the model.
        This model is designed to deal with potential thetas between U and Y.

        The following equations summarize the equations involved in the model:

        Y = G.U + H.E

        G = B / A
        H = C / A

        A = 1 + a_1*z^(-1) + ... + a_na*z^(-na)
        B = b_1*z^(-1-theta) + ... + b_nb*z^(-nb-theta)
        C = c_1*z^(-1) + ... + c_nc*z^(-nc)

        .. seealso:: https://ieeexplore.ieee.org/abstract/document/8516791


        :param na_bounds: extended range of the order of the common denominator
        :type na_bounds: list of two ints
        :param nb_bounds: extended range of the order of the G numerator
        :type nb_bounds: list of two ints
        :param nc_bounds: extended range of the order of the H numerator
        :type nc_bounds: list of two ints
        :param theta_bounds: extended range of the discrete theta in B
        :type theta_bounds: list of two ints
        :param dt: sampling time of discretized data y and u
        :type dt: float
        :param method: Method used of to attribute a performance to the model
        :type method: string
        :param max_iter: maximum numbers of iterations to find the best fit
        :type max_iter: int
        """
        for order in order_bounds:
            if isinstance(order, list | tuple):
                if not all(isinstance(x, int) for x in order):
                    raise ValueError(
                        "wrong arguments passed to define an armax model"
                    )

        self.orders = [get_val_range(order) for order in order_bounds]

        self.method = method

        self.G = G
        self.H = H
        self.Vn = Vn
        self.Yid = Yid

    def __repr__(self):
        return f"Armax({', '.join(f'[{min(order)}, {max(order)}]' for order in self.orders)}, {self.method})"

    def __str__(self):
        return (
            "Armax model:\n"
            "- Params:\n"
            f"  orders: {', '.join(f'[{min(order)}, {max(order)}]' for order in self.orders)} \n"
            f"  dt: {self.G.dt} \n"
            f"  method: {self.method} \n"
            "- Output:\n"
            f"  G: {self.G} \n"
            f"  H: {self.H} \n"
            f"  Vn: {self.Vn} \n"
            f"  Model Output: {self.Yid} \n"
        )

    @staticmethod
    def _identify(
        y: np.ndarray,
        u: np.ndarray,
        na: int,
        nb: int | np.ndarray,
        nc: int,
        theta: int | np.ndarray,
        max_iter: int,
        **_,
    ):
        """Identify

        Given model order as parameter, the recursive algorithm looks for the best fit in less
        than max_iter steps.

        At each step, the algorithm performs a least-square regression. If the

        :param y: Measured data
        :type y: Array of float
        :param u: Input data
        :type u: Array of float (same shape as y)
        :param na: order of the common denominator
        :type na: int
        :param nb: order of the numerator of G
        :type nb: int
        :param nc: order of the numerator of H
        :type nc: int
        :param theta: discrete theta expressed as a number of shifted indices between u and y
        :type theta: int
        :param max_iter: maximum numbers of iterations to find the best fit
        :type max_iter: int
        :return numerator: Numerator of G
        :rtype numerator: float array
        :return denominator: Denominator of G
        :rtype denominator: float array
        :return numerator_H: Numerator of H
        :rtype numerator_H: float array
        :return denominator_H: Denominator of H
        :rtype denominator_H: float array
        :return Vn: variance between y and the estimated output data (X*beta_hat)
        :rtype Vn: float
        :return Yid: estimated output data (X*beta_hat)
        :rtype Yid: float array
        :return max_reached: Whether the algorithm reached maximum number of iterations or not.
        :rtype max_reached: boolean
        """
        if isinstance(nb, np.ndarray):
            nb = int(np.sum(nb))
        if isinstance(theta, np.ndarray):
            theta = int(np.sum(theta))
        max_order = max(na, nb + theta, nc)
        sum_order = sum((na, nb, nc))

        # Define the usable measurements length, N, for the identification process
        N: int = y.size - max_order

        noise_hat = np.zeros(y.size)

        # Fill X matrix used to perform least-square regression: beta_hat = (X_T.X)^(-1).X_T.y
        X = np.zeros((N, sum_order))
        for i in range(N):
            X[i, 0:na] = -y[i + max_order - 1 :: -1][0:na]
            X[i, na : na + nb] = u[max_order + i - 1 :: -1][theta : nb + theta]

        Vn, Vn_old = np.inf, np.inf
        beta_hat = np.zeros(sum_order)
        I_beta = np.identity(beta_hat.size)
        iterations = 0
        # max_reached = False

        # Stay in this loop while variance has not converged or max iterations has not been
        # reached yet.
        while (Vn_old > Vn or iterations == 0) and iterations < max_iter:
            beta_hat_old = beta_hat
            Vn_old = Vn
            iterations = iterations + 1
            for i in range(N):
                X[i, na + nb : na + nb + nc] = noise_hat[
                    max_order + i - 1 :: -1
                ][0:nc]
            beta_hat = np.dot(np.linalg.pinv(X), y[max_order::])
            Vn = mean_square_error(y[max_order::], np.dot(X, beta_hat))

            # If solution found is not better than before, perform a binary search to find a better
            # solution.
            beta_hat_new = beta_hat
            interval_length = 0.5
            while Vn > Vn_old:
                beta_hat = np.dot(
                    I_beta * interval_length, beta_hat_new
                ) + np.dot(I_beta * (1 - interval_length), beta_hat_old)
                Vn = mean_square_error(y[max_order::], np.dot(X, beta_hat))

                # Stop the binary search when the interval length is minor than smallest float
                if interval_length < np.finfo(np.float32).eps:
                    beta_hat = beta_hat_old
                    Vn = Vn_old
                interval_length = interval_length / 2.0

            # Update estimated noise based on best solution found from currently considered
            # noise.
            noise_hat[max_order::] = y[max_order::] - np.dot(X, beta_hat)
            # adding non-identified outputs
            Yid = np.hstack((y[:max_order], np.dot(X, beta_hat)))

        if iterations >= max_iter:
            warn("[ARMAX_id] Reached maximum iterations.")
            # max_reached = True

        numerator = np.zeros(max_order)
        numerator[theta : nb + theta] = beta_hat[na : na + nb]

        denominator = np.zeros(max_order + 1)
        denominator[0] = 1.0
        denominator[1 : na + 1] = beta_hat[0:na]

        numerator_H = np.zeros(max_order + 1)
        numerator_H[0] = 1.0
        numerator_H[1 : nc + 1] = beta_hat[na + nb : :]

        denominator_H = np.zeros(max_order + 1)
        denominator_H[0] = 1.0
        denominator_H[1 : na + 1] = beta_hat[0:na]

        return numerator, denominator, numerator_H, denominator_H, Vn, Yid
