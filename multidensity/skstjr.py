#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Skewed Student Distribution (Jondeau & Rockinger)
==============================================================

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

from .multidensity import MultiDensity

__all__ = ['SkStJR']


class SkStJR(MultiDensity):

    """Multidimensional density (Jondeau & Rockinger).

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
    theta_start
        Initialize parameter for optimization

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
        super(SkStJR, self).__init__(eta=eta, lam=lam, data=data)

    def from_theta(self, theta=[10., 10, .5, 1.5]):
        """Initialize individual parameters from theta.

        Parameters
        ----------
        theta : array_like
            Density parameters

        """
        params = len(theta) // 2
        self.eta = np.array(theta[:params])
        self.lam = np.array(theta[params:])

    def theta_start(self, ndim=2):
        """Initialize parameter for optimization.

        Parameters
        ----------
        ndim : int
            Number of dimensions

        Returns
        -------
        array
            Parameters in one vector

        """
        eta = np.ones(ndim) * 10
        lam = np.ones(ndim)
        return np.concatenate((eta, lam))

    def marginals(self, data=None):
        """Marginal drobability density functions.

        Parameters
        ----------
        data : array_like
            Grid of point to evaluate PDF at.

            (k,) - one observation, k dimensions

            (T, k) - T observations, k dimensions

        Returns
        -------
        (T, k) array
            marginal pdf values

        """
        if data is None:
            raise ValueError('No data given!')
        self.data = np.atleast_2d(data)
        ind = - np.sign(self.data + self.const_a() / self.const_b())
        kappa = (self.const_b() * self.data + self.const_a()) \
            * self.lam ** ind
        return 2 / (np.pi * (self.eta - 2)) ** .5 \
            * self.const_b() / (self.lam + 1. / self.lam) \
            * gamma((self.eta + 1) / 2) / gamma(self.eta / 2) \
            * (1 + kappa ** 2 / (self.eta - 2)) ** (- (self.eta + 1) / 2)
