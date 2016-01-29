#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Student Distribution (Demarta & McNeil)
====================================================

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
from scipy.linalg import solve, det

from .skstdm import SkStDM

__all__ = ['MvSt']


class MvSt(SkStDM):

    """Multidimensional Student density (Demarta & McNeil).

    Attributes
    ----------
    eta : float
        Degrees of freedom. :math:`4 < \eta < \infty`
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

    def __init__(self, ndim=2, eta=10., mu=None, sigma=None, data=[0, 0]):
        """Initialize the class.

        Parameters
        ----------
        eta : float
            Degrees of freedom. :math:`2 < \eta < \infty`
        mu : array_like
            Constant in the mean. None for centered density.
        sigma : array_like
            Covariance matrix. None for standardized density.
        data : array_like
            Data grid

        """
        super(SkStDM, self).__init__(eta=eta, lam=np.zeros(ndim), data=data)
        self.ndim = ndim
        self.mu = mu
        self.sigma = sigma

    def get_name(self):
        return 'Multivariate Student'

    def from_theta(self, theta=10.):
        """Initialize individual parameters from theta.

        Parameters
        ----------
        theta : int
            Density parameters

        """
        self.eta = theta

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
        return 10

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
            return (1 - 2 / self.eta) * np.eye(self.ndim)
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
        diff_norm = solve(self.const_sigma(), diff.T)
        # (T, ) array
        diff_sandwich = (diff.T * diff_norm).sum(0)
        return ((np.pi * self.eta) ** ndim * det(self.const_sigma())) **.5 \
            * gamma((self.eta + self.ndim) / 2) / gamma(self.eta / 2) \
            * (1 + diff_sandwich / self.eta) ** (- (self.eta + ndim) / 2)
