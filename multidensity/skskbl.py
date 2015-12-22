#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Multivariate Skewed Student Distribution (Bauwens & Laurent)
============================================================

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
from scipy.optimize import minimize

from .multidensity import MultiDensity

__all__ = ['SkStBL']


class SkStBL(MultiDensity):

    """Multidimensional density (Bauwens & Laurent).

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

    def __init__(self, eta=10., lam=[.5, 1.5], data=[0, 0]):
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
        super(SkStBL, self).__init__(eta=eta, lam=lam, data=data)
