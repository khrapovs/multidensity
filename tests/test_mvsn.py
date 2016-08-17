#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for MultiDensity class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from multidensity import MvSN


class MvSNTestCase(ut.TestCase):

    """Test MvSN distribution class."""

    def test_init(self):
        """Test __init__."""

        skst = MvSN(ndim=3)

        self.assertIsInstance(skst.lam, np.ndarray)

        lam = [.5, 1.5]
        skst = MvSN(ndim=len(lam), lam=lam)

        npt.assert_array_equal(skst.lam, np.array(lam))

        mu, sigma = [.5, .4], np.ones((2, 2))
        skst = MvSN(ndim=len(lam), lam=lam, mu=mu, sigma=sigma)

        npt.assert_array_equal(skst.mu, np.array(mu))
        npt.assert_array_equal(skst.sigma, np.array(sigma))
        npt.assert_array_equal(skst.const_mu(), np.array(mu))
        npt.assert_array_equal(skst.const_sigma(), np.array(sigma))

        lam = [1.5, .5]
        skst.from_theta(np.array(lam))

        npt.assert_array_equal(skst.lam, np.array(lam))

        size = len(lam)
        data = np.random.normal(size=size)
        skst = MvSN(ndim=len(lam), data=data)

        npt.assert_array_equal(skst.data, np.atleast_2d(data))

    def test_dimensions(self):
        """Test dimensions."""

        lam = .5
        mvsn = MvSN(lam=lam, ndim=1)

        self.assertEqual(mvsn.ndim, 1)

    def test_pdf(self):
        """Test pdf."""

        lam = .5
        skst = MvSN(ndim=1, lam=lam)
        size = (10, 1)
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

        lam = [.5, 1.5, 2]
        skst = MvSN(ndim=len(lam), lam=lam)
        size = (10, len(lam))
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

    def test_cdf(self):
        """Test cdf."""

        lam = 1.5
        skst = MvSN(ndim=1, lam=lam)
        cdf = skst.cdf(np.zeros(1))

        self.assertIsInstance(cdf, float)

        lam = [1.5, .5]
        skst = MvSN(ndim=len(lam), lam=lam)
        cdf = skst.cdf(np.zeros(2) - 10)

        self.assertIsInstance(cdf, float)

    def test_quantile(self):
        """Test quantile."""

        lam = 1.5
        skst = MvSN(ndim=1, lam=lam)
        arg = -2.
        cdf = skst.cdf(arg)
        ppf = skst.ppf(cdf)

        self.assertAlmostEqual(ppf, arg)

        arg = -.1 * np.ones(3)
        cdf = skst.cdf_vec(arg)
        quantiles = skst.ppf_vec(cdf)

        npt.assert_array_almost_equal(arg, quantiles)

    def test_likelihood(self):
        """Test log-likelihood."""

        lam = [.5, 1.5, 2]
        theta = np.array(lam)
        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = MvSN(ndim=len(lam), lam=lam, data=data)
        logl1 = skst.likelihood(theta)
        logl2 = skst.likelihood(theta * 2)

        self.assertIsInstance(logl1, float)
        self.assertNotEqual(logl1, logl2)
        npt.assert_array_equal(skst.data, data)

    def test_rvs(self):
        """Test simulation."""

        lam = [.5, 1.5, 2]
        skst = MvSN(ndim=len(lam), lam=lam)
        size = 10
        rvs = skst.rvs(size=size)

        self.assertEqual(rvs.shape, (size, len(lam)))

    def test_param_array(self):
        """Test pdf."""

        ndim, nobs = 1, 10
        size = (nobs, ndim)
        lam = np.ones((nobs, ndim)) * .5
        skst = MvSN(ndim=ndim, lam=lam)
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, ndim)
        self.assertEqual(pdf.shape, (size[0], ))


if __name__ == '__main__':
    ut.main()
