#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Skewed Student (Bauwens & Laurent)
===============================================

"""
from __future__ import print_function, division

import numpy as np
import scipy.stats as scs

from scipy.special import gamma

from .multidensity import MultiDensity
from .mvst import MvSt

__all__ = ['SkStBL']


class SkStBL(MultiDensity):

    """Multidimensional density (Bauwens & Laurent).

    Attributes
    ----------
    eta : float
        Degrees of freedom. :math:`2 < \eta < \infty`
    lam : array_like
        Asymmetry. :math:`0 < \lambda < \infty`
    data : array_like
        Data grid

    Methods
    -------
    from_theta
        Initialize individual parameters from theta

    """

    def __init__(self, ndim=None, eta=10., lam=[.5, 1.5], data=[0, 0]):
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
        super(SkStBL, self).__init__(ndim=ndim, eta=eta, lam=lam, data=data)

    def get_name(self):
        return 'Bauwens & Laurent'

    def from_theta(self, theta=[10., .5, 1.5]):
        """Initialize individual parameters from theta.

        Parameters
        ----------
        theta : array_like
            Density parameters

        """
        self.eta = np.atleast_1d(theta[0])
        self.lam = np.atleast_1d(theta[1:])

    def bounds(self):
        """Parameter bounds.

        Returns
        -------
        list of tuples
            Bounds on each parameter

        """
        bound_eta = [2]
        bound_lam = np.zeros(self.ndim)
        return list(zip(np.concatenate((bound_eta, bound_lam)),
                   (1 + self.ndim) * [None]))

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
        eta = np.array([10])
        lam = np.ones(ndim)
        return np.concatenate((eta, lam))

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
        ndim = self.lam.size
        if data is None:
            raise ValueError('No data given!')
        self.data = np.atleast_2d(data)
        ind = - np.sign(self.data + self.const_a() / self.const_b())
        # (T, k) array
        kappa = (self.const_b() * self.data + self.const_a()) \
            * self.lam ** ind
        return (2 / (np.pi * (self.eta - 2)) ** .5) ** ndim \
            * gamma((self.eta + ndim) / 2) / gamma(self.eta / 2) \
            * (1 + np.sum(kappa * kappa, axis=1) / (self.eta - 2)) \
            ** (- (self.eta + ndim) / 2) \
            * np.prod(self.const_b() / (self.lam + 1. / self.lam))

    def rvs(self, size=10):
        """Simulate random variables.

        Parameters
        ----------
        size : int
            Number of data points

        Returns
        -------
        (size, ndim) array

        """
        mvst = MvSt(ndim=self.ndim, eta=self.eta)
        mvst_rvs = mvst.rvs(size=size)

        prob = np.tile(self.lam**2 / (1 + self.lam**2), (size, 1))
        bern_rvs = scs.bernoulli.rvs(prob)

        data = np.abs(mvst_rvs) \
            * (bern_rvs * self.lam - (1 - bern_rvs) / self.lam)
        return (data - self.const_a()) / self.const_b()
