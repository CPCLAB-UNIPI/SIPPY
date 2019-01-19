# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, division, print_function
import control as cnt
import sys
from builtins import object
import warnings

from .functionset import *


class Armax(object):
    def __init__(self, na_range, nb_range, nc_range, delay_range, dt,
                 method="AIC", max_iterations=100):
        if not (isinstance(na_range, (int, np.ndarray, list, tuple)) and
                isinstance(nb_range, (int, np.ndarray, list, tuple)) and
                isinstance(nc_range, (int, np.ndarray, list, tuple)) and
                isinstance(delay_range, (int, np.ndarray, list, tuple)) and
                isinstance(dt, (int, float))):
            raise ValueError("wrong arguments passed to define an armax model")

        for param in (na_range, nb_range, nc_range, delay_range):
            if isinstance(param, (list, tuple)):
                if not all(isinstance(x, int) for x in param):
                    raise ValueError("wrong arguments passed to define an armax model")

        self.na_range = na_range
        self.nb_range = nb_range
        self.nc_range = nc_range
        self.delay_range = delay_range
        self.dt = float(dt)

        self.method = method
        self.max_iterations = max_iterations

        self.na = None
        self.nb = None
        self.nc = None
        self.delay = None

        self.G = None
        self.H = None
        self.variance = None
        self.max_reached = None

    @staticmethod
    def _identify(y, u, na, nb, nc, delay, max_iterations):
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

        variance, variance_old = np.inf, np.inf
        beta_hat = np.zeros(sum_order)
        I_beta = np.identity(beta_hat.size)
        iterations = 0
        reached_max = False
        while (variance_old > variance or iterations == 0) and iterations < max_iterations:
            beta_hat_old = beta_hat
            variance_old = variance
            iterations = iterations + 1
            for i in range(N):
                X[i, na + nb:na + nb + nc] = noise_hat[max_order + i - 1::-1][0:nc]
            beta_hat = np.dot(np.linalg.pinv(X), y[max_order::])
            variance = old_div(mean_square_error(y[max_order::], np.dot(X, beta_hat)), 2)

            # If solution found is not better than before, perform a binary search to find a better
            # solution.
            beta_hat_new = beta_hat
            interval_length = 0.5
            while variance > variance_old:
                beta_hat = np.dot(I_beta * interval_length, beta_hat_new) + \
                        np.dot(I_beta * (1 - interval_length), beta_hat_old)
                variance = old_div(mean_square_error(y[max_order::], np.dot(X, beta_hat)), 2)

                # Stop the binary search when the interval length is minor than smallest float
                if interval_length < np.finfo(np.float32).eps:
                    beta_hat = beta_hat_old
                    variance = variance_old
                interval_length = interval_length / 2.

            # Update estimated noise based on best solution found from currently considered
            # noise.
            noise_hat[max_order::] = y[max_order::] - np.dot(X, beta_hat)
        if iterations >= max_iterations:
            warnings.warn("[ARMAX_id] Reached maximum iterations.")
            reached_max = True

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

        return G_num, G_den, H_num, H_den, variance, reached_max

    def find_best_estimate(self, y, u):
        na_Min = min(self.na_range)
        na_MAX = max(self.na_range) + 1
        nb_Min = min(self.nb_range)
        nb_MAX = max(self.nb_range) + 1
        theta_Min = min(self.delay_range)
        theta_Max = max(self.delay_range) + 1
        nc_Min = min(self.nc_range)
        nc_MAX = max(self.nc_range) + 1
        if (isinstance(na_Min + na_MAX + nb_Min + nb_MAX + theta_Min + theta_Max + nc_Min + nc_MAX, int)
           and na_Min >= 0 and nb_Min > 0 and nc_Min >= 0 and theta_Min >= 0) is False:
            raise ValueError("Error! na, nc, theta must be positive integers, "
                             "nb must be strictly positive integer")
        elif y.size != u.size:
            raise ValueError("Error! y and u must have tha same length")
        else:
            y_std, y = rescale(y)
            u_std, u = rescale(u)
            IC_old = np.inf
            for i in range(na_Min, na_MAX):
                for j in range(nb_Min, nb_MAX):
                    for k in range(theta_Min, theta_Max):
                        for l in range(nc_Min, nc_MAX):
                            _, _, _, _, variance, Reached_max = Armax._identify(y, u, i, j, l, k,
                                                                                self.max_iterations)
                            if Reached_max is True:
                                print("at Na=", i, " Nb=", j, " Nc=", l, " Delay:", k)
                                print("-------------------------------------")
                            IC = information_criterion(i + j + l, y.size - max(i, j + k, l), variance * 2,
                                                       self.method)
                            if IC < IC_old:
                                na_opt, nb_opt, nc_opt, delay_opt = i, j, l, k
                                IC_old = IC
            print("suggested orders are: Na=", na_opt, "; Nb=", nb_opt, "; Nc=", nc_opt, "Delay: ",
                  delay_opt)
            G_num, G_den, \
                H_num, H_den, \
                variance, max_reached = Armax._identify(y, u,
                                                        na_opt, nb_opt, nc_opt, delay_opt,
                                                        self.max_iterations)

            G_num[delay_opt:nb_opt + delay_opt] = G_num[delay_opt:nb_opt + delay_opt] * y_std / u_std

            G = cnt.tf(G_num, G_den, self.dt)
            H = cnt.tf(H_num, H_den, self.dt)
            self.na = na_opt
            self.nb = nb_opt
            self.nc = nc_opt
            self.delay = delay_opt
            self.G = G
            self.H = H
            self.max_reached = max_reached

            return na_opt, nb_opt, nc_opt, delay_opt, G, H, G_num, G_den, H_num, H_den, variance
