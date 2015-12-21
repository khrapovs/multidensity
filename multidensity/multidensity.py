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
import matplotlib.pylab as plt
import seaborn as sns

from scipy.special import gamma

__all__ = ['MultiDensity']


class MultiDensity(object):

    def __init__(self, eta=[10., 10], lam=[.5, 1.5]):
        """Initialize the class.

        Parameters
        ----------
        eta : array_like
            Degrees of freedom. :math:`2 < \eta < \infty`
        lam : array_like
            Asymmetry. :math:`0 < \lambda < 1`

        """
        self.eta = np.array(eta)
        self.lam = np.array(lam)

    @classmethod
    def from_theta(cls, theta=[10., 10, .5, 1.5]):
        """Initialize the class from theta.

        Parameters
        ----------
        theta : array_like
            Density parameters

        """
        params = len(theta) // 2
        return cls(eta=theta[:params], lam=theta[params:])

    def __const_a(self):
        """Compute a constant.

        Returns
        -------
        a : float

        """
        return gamma((self.eta - 1) / 2) / gamma(self.eta / 2) \
            * ((self.eta - 2) / np.pi) ** .5 * (self.lam - 1. / self.lam)

    def __const_b(self):
        """Compute b constant.

        Returns
        -------
        b : float

        """
        return self.lam ** 2 + 1 / self.lam ** 2 - 1 - self.__const_a() ** 2

    def pdf(self, arg=[0, 0], theta=None):
        """Probability density function (PDF).

        Parameters
        ----------
        arg : array_like
            Grid of point to evaluate PDF at.

            (k,) - one observation, k dimensions

            (T, k) - T observations, k dimensions

        Returns
        -------
        array
            PDF values

        """
        if theta is not None:
            self = self.from_theta(theta)
        arg = np.atleast_2d(arg)
        ind = np.sign(arg + self.__const_a() / self.__const_b())
        kappa = (self.__const_b() * arg + self.__const_a()) * self.lam ** ind
        marginals = 2 / (np.pi * (self.eta - 1)) ** .5 \
            * self.__const_b() / (self.lam + 1. / self.lam) \
            * gamma((self.eta + 1) / 2) / gamma(self.eta / 2) \
            * (1 + kappa ** 2 / (self.eta - 2)) ** (- (self.eta + 1) / 2)
        return np.prod(marginals, axis=1)

    def loglikelihood(self, arg=[0, 0], theta=None):
        """Log-likelihood function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        Returns
        -------
        array
            Log-likelihood values. Same shape as the input.

        """
        if theta is not None:
            self = self.from_theta(theta)

        return -np.log(self.pdf(arg)).sum()
