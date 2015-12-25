#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for MultiDensity class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from multidensity import SkStJR, SkStBL, SkStDM


class SkStJRTestCase(ut.TestCase):

    """Test SkStJR distribution class."""

    def test_init(self):
        """Test __init__."""

        skst = SkStJR()

        self.assertIsInstance(skst.eta, np.ndarray)
        self.assertIsInstance(skst.lam, np.ndarray)

        eta, lam = [10, 15], [.5, 1.5]
        skst = SkStJR(eta=eta, lam=lam)

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        eta, lam = [15, 10], [1.5, .5]
        skst.from_theta(np.concatenate((eta, lam)))

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        size = (10, len(eta))
        data = np.random.normal(size=size)
        skst = SkStJR(data=data)

        npt.assert_array_equal(skst.data, data)

    def test_marginals(self):
        """Test marginals."""

        eta, lam = [10, 15, 10], [.5, 1.5, 2]
        skst = SkStJR(eta=eta, lam=lam)
        size = (10, len(eta))
        data = np.random.normal(size=size)
        marginals = skst.marginals(data)

        self.assertEqual(marginals.ndim, 2)
        self.assertEqual(marginals.shape, size)
        self.assertGreater(marginals.all(), 0)

    def test_pdf(self):
        """Test pdf."""

        eta, lam = [10, 15, 10], [.5, 1.5, 2]
        skst = SkStJR(eta=eta, lam=lam)
        size = (10, len(eta))
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

    def test_cdf(self):
        """Test cdf."""

        eta, lam = 20, 1.5
        skst = SkStJR(eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(1))

        self.assertIsInstance(cdf[0], float)

        eta, lam = [20, 5], [1.5, .5]
        skst = SkStJR(eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(2) - 10)

        self.assertIsInstance(cdf[0], float)

    def test_quantile(self):
        """Test quantile."""

        eta, lam = 20, 1.5
        skst = SkStJR(eta=eta, lam=lam)
        arg = -2.
        cdf = skst.cdf(arg)[0]
        ppf = skst.ppf(cdf)

        self.assertAlmostEqual(ppf, arg)

    def test_loglikelihood(self):
        """Test log-likelihood."""

        eta, lam = [10, 15, 10], [.5, 1.5, 2]
        theta = np.concatenate((eta, lam))
        size = (10, len(eta))
        data = np.random.normal(size=size)
        skst = SkStJR(eta=eta, lam=lam, data=data)
        logl1 = skst.loglikelihood(theta)
        logl2 = skst.loglikelihood(theta * 2)

        self.assertIsInstance(logl1, float)
        self.assertNotEqual(logl1, logl2)
        npt.assert_array_equal(skst.data, data)


class SkStBLTestCase(ut.TestCase):

    """Test SkStBL distribution class."""

    def test_init(self):
        """Test __init__."""

        skst = SkStBL()

        self.assertIsInstance(skst.eta, np.ndarray)
        self.assertIsInstance(skst.lam, np.ndarray)

        eta, lam = 10, [.5, 1.5]
        skst = SkStBL(eta=eta, lam=lam)

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        eta, lam = 15, [1.5, .5]
        skst.from_theta(np.concatenate((np.atleast_1d(eta), lam)))

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = SkStBL(data=data)

        npt.assert_array_equal(skst.data, data)

    def test_pdf(self):
        """Test pdf."""

        eta, lam = 10, [.5, 1.5, 2]
        skst = SkStBL(eta=eta, lam=lam)
        size = (10, len(lam))
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

    def test_cdf(self):
        """Test cdf."""

        eta, lam = 20, 1.5
        skst = SkStBL(eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(1))

        self.assertIsInstance(cdf[0], float)

        eta, lam = 20, [1.5, .5]
        skst = SkStBL(eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(2) - 10)

        self.assertIsInstance(cdf[0], float)

    def test_quantile(self):
        """Test quantile."""

        eta, lam = 20, 1.5
        skst = SkStBL(eta=eta, lam=lam)
        arg = -2.
        cdf = skst.cdf(arg)[0]
        ppf = skst.ppf(cdf)

        self.assertAlmostEqual(ppf, arg)

    def test_loglikelihood(self):
        """Test log-likelihood."""

        eta, lam = 10, [.5, 1.5, 2]
        theta = np.concatenate((np.atleast_1d(eta), lam))
        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = SkStBL(eta=eta, lam=lam, data=data)
        logl1 = skst.loglikelihood(theta)
        logl2 = skst.loglikelihood(theta * 2)

        self.assertIsInstance(logl1, float)
        self.assertNotEqual(logl1, logl2)
        npt.assert_array_equal(skst.data, data)


class SkStDMTestCase(ut.TestCase):

    """Test SkStDM distribution class."""

    def test_init(self):
        """Test __init__."""

        skst = SkStDM()

        self.assertIsInstance(skst.eta, np.ndarray)
        self.assertIsInstance(skst.lam, np.ndarray)

        eta, lam = 10, [.5, 1.5]
        skst = SkStDM(eta=eta, lam=lam)

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        mu, sigma = [.5, .4], np.ones((2, 2))
        skst = SkStDM(eta=eta, lam=lam, mu=mu, sigma=sigma)

        npt.assert_array_equal(skst.mu, np.array(mu))
        npt.assert_array_equal(skst.sigma, np.array(sigma))
        npt.assert_array_equal(skst.const_mu(), np.array(mu))
        npt.assert_array_equal(skst.const_sigma(), np.array(sigma))

        eta, lam = 15, [1.5, .5]
        skst.from_theta(np.concatenate((np.atleast_1d(eta), lam)))

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = SkStDM(data=data)

        npt.assert_array_equal(skst.data, data)

    def test_pdf(self):
        """Test pdf."""

        eta, lam = 30, .5
        skst = SkStDM(eta=eta, lam=lam)
        size = (10, 1)
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

        eta, lam = 30, [.5, 1.5, 2]
        skst = SkStDM(eta=eta, lam=lam)
        size = (10, len(lam))
        data = np.random.normal(size=size)
        pdf = skst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

    def test_cdf(self):
        """Test cdf."""

        eta, lam = 20, 1.5
        skst = SkStDM(eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(1))

        self.assertIsInstance(cdf[0], float)

        eta, lam = 20, [1.5, .5]
        skst = SkStDM(eta=eta, lam=lam)
        cdf = skst.cdf(np.zeros(2) - 10)

        self.assertIsInstance(cdf[0], float)

    def test_quantile(self):
        """Test quantile."""

        eta, lam = 20, 1.5
        skst = SkStDM(eta=eta, lam=lam)
        arg = -2.
        cdf = skst.cdf(arg)[0]
        ppf = skst.ppf(cdf)

        self.assertAlmostEqual(ppf, arg)

    def test_loglikelihood(self):
        """Test log-likelihood."""

        eta, lam = 100, [.5, 1.5, 2]
        theta = np.concatenate((np.atleast_1d(eta), lam))
        size = (10, len(lam))
        data = np.random.normal(size=size)
        skst = SkStDM(eta=eta, lam=lam, data=data)
        logl1 = skst.loglikelihood(theta)
        logl2 = skst.loglikelihood(theta * 2)

        self.assertIsInstance(logl1, float)
        self.assertNotEqual(logl1, logl2)
        npt.assert_array_equal(skst.data, data)

    def test_rvs(self):
        """Test simulation."""

        eta, lam = 100, [.5, 1.5, 2]
        skst = SkStDM(eta=eta, lam=lam)
        size = 10
        rvs = skst.rvs(size=size)

        self.assertEqual(rvs.shape, (size, len(lam)))


if __name__ == '__main__':
    ut.main()
