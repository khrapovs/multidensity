#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Distributions
==========================

Introduction
------------


References
----------


Examples
--------


"""
from __future__ import print_function, division

import numpy as np

from scipy.special import gamma
from scipy.optimize import minimize

__all__ = ['MultiDensity']


class MultiDensity(object):

    """Multidimensional density.

    Attributes
    ----------
    eta : array_like
        Degrees of freedom. :math:`2 < \eta < \infty`
    lam : array_like
        Asymmetry. :math:`0 < \lambda < \infty`
    data : array_like
        Data grid

    Methods
    -------
    from_theta
        Initialize individual parameters from theta
    marginals
        Marginal drobability density functions
    pdf
        Probability density function
    loglikelihood
        Log-likelihood function
    fit_mle
        Fit parameters with MLE

    """

    def __init__(self, eta=[10., 10], lam=[.5, 1.5], data=[0, 0]):
        """Initialize the class.

        Parameters
        ----------
        eta : array_like
            Degrees of freedom. :math:`2 < \eta < \infty`
        lam : array_like
            Asymmetry. :math:`0 < \lambda < \infty`
        data : array_like
            Data grid

        """
        self.eta = np.atleast_1d(eta)
        self.lam = np.atleast_1d(lam)
        self.data = np.atleast_2d(data)

    def const_a(self):
        """Compute a constant.

        Returns
        -------
        float

        """
        return gamma((self.eta - 1) / 2) / gamma(self.eta / 2) \
            * ((self.eta - 2) / np.pi) ** .5 * (self.lam - 1. / self.lam)

    def const_b(self):
        """Compute b constant.

        Returns
        -------
        float

        """
        return self.lam ** 2 + self.lam ** (-2) - 1 - self.const_a() ** 2

    def pdf(self, data=None):
        """Probability density function (PDF).

        Parameters
        ----------
        data : array_like
            Grid of point to evaluate PDF at.

            (k,) - one observation, k dimensions

            (T, k) - T observations, k dimensions

        Returns
        -------
        (T, ) array
            PDF values

        """
        if data is None:
            raise ValueError('No data given!')
        return np.prod(self.marginals(data), axis=1)

    def loglikelihood(self, theta=[10., 10, .5, 1.5]):
        """Log-likelihood function.

        Parameters
        ----------
        theta : array_like
            Density parameters

        Returns
        -------
        float
            Log-likelihood values. Same shape as the input.

        """
        if theta is None:
            raise ValueError('No parameter given!')
        self.from_theta(theta)
        return -np.log(self.pdf(self.data)).mean()

    def fit_mle(self, theta_start=None):
        """Fit parameters with MLE.

        Parameters
        ----------
        theta : array_like
            Density parameters

        Returns
        -------
        array
            Log-likelihood values. Same shape as the input.

        """
        ndim = self.data.shape[1]
        if theta_start is None:
            theta_start = self.theta_start(ndim)
        bound_eta = np.ones(ndim) * 2
        bound_lam = np.zeros(ndim)
        bounds = zip(np.concatenate((bound_eta, bound_lam)), 2 * ndim * [None])
        return minimize(self.loglikelihood, theta_start, method='Nelder-Mead',
                        bounds=list(bounds))
