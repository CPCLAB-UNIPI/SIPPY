# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, division, print_function
import control.matlab as cnt
from builtins import object
import warnings

from .functionset import *
# from functionset import *


class Armax(object):
    def __init__(self, na_bounds, nb_bounds, nc_bounds, delay_bounds, dt,
                 method="AIC", max_iterations=100):
        """Armax model class.

        The AutoRegressive-Moving-Average with eXogenous inputs model is computed based on a
        recursive lest-square regression between the input data (U) and the measured output data
        (Y). As Y is noisy in practice, a white noise (E) is identified within the model.
        This model is designed to deal with potential delays between U and Y.

        The following equations summarize the equations involved in the model:

        Y = G.U + H.E

        G = B / A
        H = C / A

        A = 1 + a_1*z^(-1) + ... + a_na*z^(-na)
        B = b_1*z^(-1-delay) + ... + b_nb*z^(-nb-delay)
        C = c_1*z^(-1) + ... + c_nc*z^(-nc)

        .. seealso:: https://ieeexplore.ieee.org/abstract/document/8516791


        :param na_bounds: extended range of the order of the common denominator
        :type na_bounds: list of two ints
        :param nb_bounds: extended range of the order of the G numerator
        :type nb_bounds: list of two ints
        :param nc_bounds: extended range of the order of the H numerator
        :type nc_bounds: list of two ints
        :param delay_bounds: extended range of the discrete delay in B
        :type delay_bounds: list of two ints
        :param dt: sampling time of discretized data y and u
        :type dt: float
        :param method: Method used of to attribute a performance to the model
        :type method: string
        :param max_iterations: maximum numbers of iterations to find the best fit
        :type max_iterations: int
        """
        if not (isinstance(na_bounds, (int, np.ndarray, list, tuple)) and
                isinstance(nb_bounds, (int, np.ndarray, list, tuple)) and
                isinstance(nc_bounds, (int, np.ndarray, list, tuple)) and
                isinstance(delay_bounds, (int, np.ndarray, list, tuple)) and
                isinstance(dt, (int, float))):
            raise ValueError("wrong arguments passed to define an armax model")

        for param in (na_bounds, nb_bounds, nc_bounds, delay_bounds):
            if isinstance(param, (list, tuple)):
                if not all(isinstance(x, int) for x in param):
                    raise ValueError("wrong arguments passed to define an armax model")
        self.na_range = range(min(na_bounds), max(na_bounds)+1)
        self.nb_range = range(min(nb_bounds), max(nb_bounds) + 1)
        self.nc_range = range(min(nc_bounds), max(nc_bounds) + 1)
        self.delay_range = range(min(delay_bounds), max(delay_bounds) + 1)
        self.dt = float(dt)

        self.method = method
        self.max_iterations = max_iterations

        self.na = None
        self.nb = None
        self.nc = None
        self.delay = None

        self.G = None
        self.H = None
        self.Vn = None
        self.Yid = None
        self.max_reached = None
        

    def __repr__(self):
        na_bounds = [min(self.na_range), max(self.na_range)]
        nb_bounds = [min(self.nb_range), max(self.nb_range)]
        nc_bounds = [min(self.nc_range), max(self.nc_range)]
        delay_bounds = [min(self.delay_range), max(self.delay_range)]
        return "Armax({}, {}, {}, {}, {}, {}, {})".format(na_bounds, nb_bounds, nc_bounds,
                                                          delay_bounds, self.dt, self.method,
                                                          self.max_iterations)

    def __str__(self):
        return "Armax model:\n" \
               "- Params:\n" \
               "  na: {} ({}, {})\n" \
               "  nb: {} ({}, {})\n" \
               "  nc: {} ({}, {})\n" \
               "  delay: {} ({}, {})\n" \
               "  dt: {} \n" \
               "  method: {} \n" \
               "  max iterations: {} \n" \
               "- Output:\n" \
               "  G: {} \n" \
               "  H: {} \n" \
               "  Vn: {} \n" \
               "  Model Output: {} \n" \
               "  Max reached: {}".format(self.na, min(self.na_range), max(self.na_range),
                                          self.nb, min(self.nb_range), max(self.nb_range),
                                          self.nc, min(self.nc_range), max(self.nc_range),
                                          self.delay, min(self.delay_range), max(self.delay_range),
                                          self.dt,
                                          self.method,
                                          self.max_iterations,
                                          self.G,
                                          self.H,
                                          self.Vn,
                                          self.Yid,
                                          self.max_reached)

    @staticmethod
    def _identify(y, u, na, nb, nc, delay, max_iterations):
        """ Identify

        Given model order as parameter, the recursive algorithm looks for the best fit in less
        than max_iterations steps.

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
        :param delay: discrete delay expressed as a number of shifted indices between u and y
        :type delay: int
        :param max_iterations: maximum numbers of iterations to find the best fit
        :type max_iterations: int
        :return G_num: Numerator of G
        :rtype G_num: float array
        :return G_den: Denominator of G
        :rtype G_den: float array
        :return H_num: Numerator of H
        :rtype H_num: float array
        :return H_den: Denominator of H
        :rtype H_den: float array
        :return Vn: variance between y and the estimated output data (X*beta_hat)
        :rtype Vn: float
        :return Yid: estimated output data (X*beta_hat)
        :rtype Yid: float array
        :return max_reached: Whether the algorithm reached maximum number of iterations or not.
        :rtype max_reached: boolean
        """
        max_order = max(na, nb + delay, nc)
        sum_order = sum((na, nb, nc))

        # Define the usable measurements length, N, for the identification process
        N = y.size - max_order

        noise_hat = np.zeros(y.size)

        # Fill X matrix used to perform least-square regression: beta_hat = (X_T.X)^(-1).X_T.y
        X = np.zeros((N, sum_order))
        for i in range(N):
            X[i, 0:na] = -y[i + max_order - 1::-1][0:na]
            X[i, na:na + nb] = u[max_order + i - 1::-1][delay:nb + delay]

        Vn, Vn_old = np.inf, np.inf
        beta_hat = np.zeros(sum_order)
        I_beta = np.identity(beta_hat.size)
        iterations = 0
        max_reached = False

        # Stay in this loop while variance has not converged or max iterations has not been
        # reached yet.
        while (Vn_old > Vn or iterations == 0) and iterations < max_iterations:
            beta_hat_old = beta_hat
            Vn_old = Vn
            iterations = iterations + 1
            for i in range(N):
                X[i, na + nb:na + nb + nc] = noise_hat[max_order + i - 1::-1][0:nc]
            beta_hat = np.dot(np.linalg.pinv(X), y[max_order::])
            Vn = mean_square_error(y[max_order::], np.dot(X, beta_hat))

            # If solution found is not better than before, perform a binary search to find a better
            # solution.
            beta_hat_new = beta_hat
            interval_length = 0.5
            while Vn > Vn_old:
                beta_hat = np.dot(I_beta * interval_length, beta_hat_new) + \
                        np.dot(I_beta * (1 - interval_length), beta_hat_old)
                Vn = mean_square_error(y[max_order::], np.dot(X, beta_hat))

                # Stop the binary search when the interval length is minor than smallest float
                if interval_length < np.finfo(np.float32).eps:
                    beta_hat = beta_hat_old
                    Vn = Vn_old
                interval_length = interval_length / 2.

            # Update estimated noise based on best solution found from currently considered
            # noise.
            noise_hat[max_order::] = y[max_order::] - np.dot(X, beta_hat)
            # adding non-identified outputs
            y_id = np.hstack((y[:max_order], np.dot(X,beta_hat)))
            
        if iterations >= max_iterations:
            warnings.warn("[ARMAX_id] Reached maximum iterations.")
            max_reached = True

        G_num = np.zeros(max_order)
        G_num[delay:nb + delay] = beta_hat[na:na + nb]

        G_den = np.zeros(max_order + 1)
        G_den[0] = 1.
        G_den[1:na + 1] = beta_hat[0:na]

        H_num = np.zeros(max_order + 1)
        H_num[0] = 1.
        H_num[1:nc + 1] = beta_hat[na + nb::]

        H_den = np.zeros(max_order + 1)
        H_den[0] = 1.
        H_den[1:na + 1] = beta_hat[0:na]

        return G_num, G_den, H_num, H_den, Vn, y_id, max_reached

    def find_best_estimate(self, y, u):
        """ Find best estimate

        Find best ARMAX estimate, given measurements and input data.

        :param y: Measurements data
        :param u: Input data
        """
        if y.size != u.size:
            raise ValueError("y and u must have tha same length")

        y_std, y = rescale(y)
        u_std, u = rescale(u)

        if u_std == 0.:
            raise ValueError("model cannot be estimated based on a constant input signal")

        IC_old = np.inf
        G_num_opt, G_den_opt, H_num_opt, H_den_opt = np.NAN, np.NAN, np.NAN, np.NAN
        for na in self.na_range:
            for nb in self.nb_range:
                for nc in self.nc_range:
                    for delay in self.delay_range:
                        G_num, G_den, \
                            H_num, H_den, \
                            Vn, y_id, max_reached = Armax._identify(y, u, na, nb, nc, delay,
                                                                    self.max_iterations)
                        if max_reached is True:
                            warnings.warn("[ARMAX ID] Max reached for:na: {} | nb: {} | nc: {} | "
                                          "delay: {}".format(na, nb, nc, delay))
                        IC = information_criterion(na + nb + delay,
                                                   y.size - max(na, nb + delay, nc),
                                                   Vn, self.method)
                        if IC < IC_old:
                            self.na, self.nb, self.nc, self.delay, IC_old = na, nb, nc, delay, IC
                            G_num_opt, G_den_opt, H_num_opt, H_den_opt = G_num, G_den, H_num, H_den
                            self.Vn, self.Yid, self.max_reached = Vn, np.atleast_2d(y_id) * y_std, max_reached  

        G_num_opt[self.delay:self.nb + self.delay] = \
            G_num_opt[self.delay:self.nb + self.delay] * y_std / u_std

        self.G = cnt.tf(G_num_opt, G_den_opt, self.dt)
        self.H = cnt.tf(H_num_opt, H_den_opt, self.dt)
        print(self)

