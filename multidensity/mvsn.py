#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Skewed Normal Distribution (Azzalini & Capitanio)
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

from scipy.special import gamma, kv
from scipy.linalg import solve, det
from scipy.stats import invgamma, multivariate_normal
import scipy.stats as scs
import scipy.linalg as scl

from .multidensity import MultiDensity

__all__ = ['MvSN']


class MvSN(MultiDensity):

    """Multidimensional Skewed Normal Distribution (Azzalini & Capitanio).

    Attributes
    ----------
    lam : array_like
        Asymmetry. :math:`-\infty < \lambda < \infty`
    data : array_like
        Data grid

    Methods
    -------
    from_theta
        Initialize individual parameters from theta
    pdf
        Probability density function
    theta_start
        Initialize parameter for optimization

    """

    def __init__(self, ndim=2, lam=[.5, 1.5],
                 mu=None, sigma=None, data=[0, 0]):
        """Initialize the class.

        Parameters
        ----------
        lam : array_like
            Asymmetry. :math:`0 < \lambda < \infty`
        mu : array_like
            Constant in the mean. None for centered density.
        sigma : array_like
            Covariance matrix. None for standardized density.
        data : array_like
            Data grid

        """
        super(MvSN, self).__init__(lam=lam, data=data)
        self.ndim = np.array(lam).size
        self.mu = mu
        self.sigma = sigma

    def get_name(self):
        return 'Multivariate Skewed Normal'

    def from_theta(self, theta=[.5, 1.5]):
        """Initialize individual parameters from theta.

        Parameters
        ----------
        theta : array_like
            Density parameters

        """
        self.lam = np.array(theta)

    def theta_start(self, ndim=2):
        """Initialize parameter for optimization.

        Parameters
        ----------
        ndim : int
            Number of dimensions

        Returns
        -------
        int
            Parameters in one vector

        """
        return np.zeros(ndim)

    def const_mu(self):
        """Compute a constant.

        Returns
        -------
        (ndim, ) array

        """
        if self.mu is None:
            return np.zeros(self.ndim)
        else:
            return np.array(self.mu)

    def const_sigma(self):
        """Compute a constant.

        Returns
        -------
        (ndim, ndim) array

        """
        if self.sigma is None:
            return np.eye(self.ndim)
        else:
            return np.array(self.sigma)

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
        self.data = np.atleast_2d(data)
        omega = np.diag(self.const_sigma())**.5
        # (T, k) array
        norm_diff = np.sum((self.data - self.const_mu()) / omega * self.lam, 1)
        return 2 * scs.multivariate_normal.pdf(self.data, mean=self.const_mu(),
                                               cov=self.const_sigma()) \
            * scs.norm.cdf(norm_diff)

    def cdf(self, data=None):
        """Cumulative density function (CDF).

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
        self.data = np.atleast_2d(data)
        # (k, ) array
        omega = np.diag(self.const_sigma())**.5
        rho = self.const_sigma() / (omega[:, np.newaxis] * omega)
        norm_diff = (self.data - self.const_mu()) / omega
        norm_lam = scl.solve(rho, self.lam)
        delta = np.atleast_2d(norm_lam / (1 + np.sum(norm_lam * self.lam)))
        big_rho = np.bmat([[np.ones((1, 1)), delta], [delta.T, rho]])
        norm_diff_ext = np.hstack((np.zeros((self.data.shape[0], 1)),
                                   norm_diff))
        low = -10 * np.ones(1 + self.ndim)
        mean = np.zeros(1 + self.ndim)
        # (http://www.nhsilbert.net/source/2014/04/
        # multivariate-normal-cdf-values-in-python/)
        mvncdf = [2 * scs.mvn.mvnun(low, x, mean, big_rho)[0]
            for x in norm_diff_ext]
        if len(mvncdf) == 1:
            return mvncdf[0]
        else:
            return np.array(mvncdf)
