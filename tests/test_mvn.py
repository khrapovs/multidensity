#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for Multivariate Normal class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt
import scipy.stats as scs

from multidensity import MvN


class MvNTestCase(ut.TestCase):

    """Test MvN distribution class."""

    def test_pdf(self):
        """Test pdf."""

        ndim, nobs = 3, 10
        size = (nobs, ndim)
        data = np.random.normal(size=size)

        pdf = MvN.pdf(data)

        self.assertEqual(pdf.ndim, 1)

        norm_pdf = np.zeros(nobs)
        for obs in range(nobs):
            scs_ndpf = scs.multivariate_normal.pdf
            norm_pdf[obs] = scs_ndpf(data[obs], mean=np.zeros(ndim),
                                     cov=np.eye(ndim))

        npt.assert_array_almost_equal(np.exp(pdf), norm_pdf)

        mean = np.random.normal(size=size)
        pdf = MvN.pdf(data, mean=mean)

        norm_pdf = np.zeros(nobs)
        for obs in range(nobs):
            scs_ndpf = scs.multivariate_normal.pdf
            norm_pdf[obs] = scs_ndpf(data[obs], mean=mean[obs])

        npt.assert_array_almost_equal(np.exp(pdf), norm_pdf)

        mean = np.random.normal(size=size)

        cov = np.zeros((nobs, ndim, ndim))
        for obs in range(nobs):
            cov[obs] = np.corrcoef(np.random.normal(size=size).T)

        pdf = MvN.pdf(data, mean=mean, cov=cov)

        norm_pdf = np.zeros(nobs)
        for obs in range(nobs):
            scs_ndpf = scs.multivariate_normal.pdf
            norm_pdf[obs] = scs_ndpf(data[obs], mean=mean[obs], cov=cov[obs])

        npt.assert_array_almost_equal(np.exp(pdf), norm_pdf)


if __name__ == '__main__':
    ut.main()
