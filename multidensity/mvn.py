#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Normal
===================

"""
from __future__ import print_function, division

import numpy as np

import numpy.linalg as npl

__all__ = ['MvN']


class MvN(object):

    """Multidimensional Normal Distribution.

    Attributes
    ----------
    data : array_like
        Data grid

    Methods
    -------
    from_theta
        Initialize individual parameters from theta
    cdf
        Cumulative density function (CDF)
    rvs
        Simulate random variables

    """

    def __init__(self):
        """Initialize the class.

        """
        pass

    def get_name(self):
        return 'Multivariate Normal'

    @staticmethod
    def pdf(data, mean=None, cov=None):
        """Probability density function (PDF).

        Parameters
        ----------
        data : (nobs, ndim) array
            Grid of point to evaluate PDF at.
        mean : (nobs, ndim) array
            Means for each data point
        cov : (nobs, ndim, ndim)
            Covariances for each data point

        Returns
        -------
        (nobs, ) array
            PDF values

        """
        nobs, ndim = data.shape
        if mean is None:
            mean = np.zeros(data.shape)
        if cov is None:
            cov = np.tile(np.eye(ndim), (nobs, 1, 1))

        # (nobs, ndim)
        diff = data - mean
        # (nobs, ndim) and (nobs, ndim, ndim)
        eigvalues, eigvec = npl.eigh(cov)
        # (nobs, )
        log_det_cov = np.log(eigvalues).sum(-1)
        # (nobs, ndim, ndim)
        cov_inv = np.multiply(eigvec, np.sqrt(1 / eigvalues)[:, np.newaxis, :])
        # (nobs, ndim)
        sandwich = np.matmul(diff[:, np.newaxis, :], cov_inv)
        # (nobs, )
        sandwich = np.square(sandwich).sum(-1).squeeze()
        return -.5 * (ndim * np.log(2 * np.pi) + log_det_cov + sandwich)
