#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for MultiDensity class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from multidensity import SkStBL


class SkStBLTestCase(ut.TestCase):

    """Test SkStBL distribution class."""

    def test_init(self):
        """Test __init__."""

        skst = SkStBL(ndim=3)

        self.assertIsInstance(skst.eta, np.ndarray)
        self.assertIsInstance(skst.lam, np.ndarray)

        eta, lam = 10, [.5, 1.5]
        skst = SkStBL(ndim=len(lam), eta=eta, lam=lam)

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        eta, lam = 15, [1.5, .5]
        skst.from_theta(np.concatenate((np.atleast_1d(eta), lam)))

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = SkStBL(ndim=len(lam), data=data)

        npt.assert_array_equal(skst.data, data)

    def test_pdf(self):
        """Test pdf."""

        eta, lam = 10, [.5, 1.5, 2]
        skst = SkStBL(ndim=len(lam), eta=eta, lam=lam)
        size = (10, len(lam))
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

    def test_cdf(self):
        """Test cdf."""

        eta, lam = 20, 1.5
        skst = SkStBL(ndim=1, eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(1))

        self.assertIsInstance(cdf, float)

        eta, lam = 20, [1.5, .5]
        skst = SkStBL(ndim=len(lam), eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(2) - 10)

        self.assertIsInstance(cdf, float)

    def test_quantile(self):
        """Test quantile."""

        eta, lam = 20, 1.5
        skst = SkStBL(ndim=1, eta=eta, lam=lam)
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

        eta, lam = 10, [.5, 1.5, 2]
        theta = np.concatenate((np.atleast_1d(eta), lam))
        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = SkStBL(ndim=len(lam), eta=eta, lam=lam, data=data)
        logl1 = skst.likelihood(theta)
        logl2 = skst.likelihood(theta * 2)

        self.assertIsInstance(logl1, float)
        self.assertNotEqual(logl1, logl2)
        npt.assert_array_equal(skst.data, data)

    def test_rvs(self):
        """Test simulation."""

        eta, lam = 100, [.5, 1.5, 2]
        skst = SkStBL(ndim=len(lam), eta=eta, lam=lam)
        size = 10
        rvs = skst.rvs(size=size)

        self.assertEqual(rvs.shape, (size, len(lam)))


if __name__ == '__main__':
    ut.main()
