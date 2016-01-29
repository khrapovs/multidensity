#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Skewed Student Distribution (Azzalini & Capitanio)
===============================================================

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
import scipy.stats as scs
import scipy.linalg as scl

from .multidensity import MultiDensity
from .mvst import MvSt
from .mvsn import MvSN

__all__ = ['SkStAC']


class SkStAC(MultiDensity):

    """Multidimensional density (Azzalini & Capitanio).

    Attributes
    ----------
    eta : float
        Degrees of freedom. :math:`4 < \eta < \infty`
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

    def __init__(self, ndim=None, eta=10., lam=[.5, 1.5],
                 mu=None, sigma=None, data=None):
        """Initialize the class.

        Parameters
        ----------
        eta : float
            Degrees of freedom. :math:`2 < \eta < \infty`
        lam : array_like
            Asymmetry. :math:`0 < \lambda < \infty`
        mu : array_like
            Constant in the mean. None for centered density.
        sigma : array_like
            Covariance matrix. None for standardized density.
        data : array_like
            Data grid

        """
        super(SkStAC, self).__init__(ndim=ndim, eta=eta, lam=lam, data=data)
        self.mu = mu
        self.sigma = sigma

    def get_name(self):
        return 'Azzalini & Capitanio'

    def from_theta(self, theta=[10., .5, 1.5]):
        """Initialize individual parameters from theta.

        Parameters
        ----------
        theta : array_like
            Density parameters

        """
        self.eta = np.atleast_1d(theta[0])
        self.lam = np.atleast_1d(theta[1:])

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
        lam = np.zeros(ndim)
        return np.concatenate((eta, lam))

    def norm_sigma(self):
        """Correlation matrix.

        Returns
        -------
        (ndim, ndim) array

        """
        if self.sigma is None:
            ndim = self.lam.size
            return np.eye(ndim)
        else:
            omega = np.diag(self.sigma)**.5
            return self.sigma / (omega[:, np.newaxis] * omega)

    def const_delta(self):
        """Compute a constant.

        Returns
        -------
        (ndim, ) array

        """
        if self.sigma is None:
            return self.lam / (1 + np.sum(self.lam**2))**.5
        else:
            sigma = self.norm_sigma()
            factor, lower = scl.cho_factor(sigma, lower=True)
            norm = scl.solve_triangular(factor, self.lam, lower=lower)
            return sigma.dot(self.lam) / (1 + np.sum(norm**2))**.5

    def const_xi(self):
        """Compute a constant.

        Returns
        -------
        (ndim, ) array

        """
        return gamma((self.eta - 1) / 2) / gamma(self.eta / 2) \
            * (self.eta / np.pi)**.5 * self.const_delta()

    def const_mu(self):
        """Compute a constant.

        Returns
        -------
        (ndim, ) array

        """
        if self.mu is None:
            return - np.diag(self.const_sigma())**.5 * self.const_xi()
        else:
            return np.array(self.mu)

    def const_sigma(self):
        """Compute a constant.

        Returns
        -------
        (ndim, ndim) array

        """
        if self.sigma is None:
            ndim = self.lam.size
            return (1 - 2 / self.eta) * np.eye(ndim)
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
        ndim = self.lam.size
        if data is None:
            raise ValueError('No data given!')
        self.data = np.atleast_2d(data)
        # (T, k) array
        diff = self.data - self.const_mu()
        # (k, T) array
        diff_norm = scl.solve(self.const_sigma(), diff.T)
        # (T, ) array
        diff_sandwich = (diff.T * diff_norm).sum(0)
        mvst = MvSt(eta=self.eta, ndim=ndim)
        # (k, ) array
        omega = np.diag(self.const_sigma())
        # (T, ) array
        arg = (diff * (self.lam / omega)).sum(1) \
            * ((self.eta + ndim) / (diff_sandwich + self.eta))**.5
        df = self.eta + ndim
        return 2 * mvst.pdf(diff) * scs.t.cdf(arg, df=df)

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
        ndim = self.lam.size
        mvsn = MvSN(ndim=ndim, lam=self.lam,
                    mu=np.zeros(ndim), sigma=self.const_sigma())
        mvs_rvs = mvsn.rvs(size=size)
        igrv = scs.invgamma.rvs(self.eta / 2, scale=self.eta / 2, size=size)
        igrv = igrv[:, np.newaxis]
        return self.const_mu() + igrv ** .5 * mvs_rvs
