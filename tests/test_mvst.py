#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for MultiDensity class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from multidensity import MvSt


class MvStTestCase(ut.TestCase):

    """Test MvSt distribution class."""

    def test_init(self):
        """Test __init__."""

        mvst = MvSt(ndim=3)

        self.assertIsInstance(mvst.eta, np.ndarray)

        eta = 10
        mvst = MvSt(ndim=1, eta=eta)

        npt.assert_array_equal(mvst.eta, np.array(eta))

        mu, sigma = [.5, .4], np.ones((2, 2))
        mvst = MvSt(ndim=len(mu), eta=eta, mu=mu, sigma=sigma)

        npt.assert_array_equal(mvst.mu, np.array(mu))
        npt.assert_array_equal(mvst.sigma, np.array(sigma))
        npt.assert_array_equal(mvst.const_mu(), np.array(mu))
        npt.assert_array_equal(mvst.const_sigma(), np.array(sigma))

        eta = 15
        mvst.from_theta(eta)

        self.assertEqual(mvst.eta, eta)

        ndim = 2
        size = (10, ndim)
        data = np.random.normal(size=size)
        skst = MvSt(ndim=ndim, data=data)

        npt.assert_array_equal(skst.data, data)

    def test_pdf(self):
        """Test pdf."""

        eta = 30
        mvst = MvSt(ndim=1, eta=eta)
        size = (10, 1)
        data = np.random.normal(size=size)
        pdf = mvst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

        eta = 30
        ndim = 2
        mvst = MvSt(ndim=ndim, eta=eta)
        size = (10, ndim)
        data = np.random.normal(size=size)
        pdf = mvst.pdf(data)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0], ))

    def test_cdf(self):
        """Test cdf."""

        eta = 20
        ndim = 3
        mvst = MvSt(ndim=ndim, eta=eta)
        cdf = mvst.cdf(np.zeros(1))

        self.assertIsInstance(cdf, float)

        eta = 20
        mvst = MvSt(ndim=ndim, eta=eta)
        cdf = mvst.cdf(np.zeros(ndim) - 10)

        self.assertIsInstance(cdf, float)

    def test_quantile(self):
        """Test quantile."""

        eta = 20
        ndim = 1
        mvst = MvSt(ndim=ndim, eta=eta)
        arg = -2.
        cdf = mvst.cdf(arg)
        ppf = mvst.ppf(cdf)

        self.assertAlmostEqual(ppf, arg)

        arg = -.1 * np.ones(ndim)
        cdf = mvst.cdf_vec(arg)
        quantiles = mvst.ppf_vec(cdf)

        npt.assert_array_almost_equal(arg, quantiles)

    def test_loglikelihood(self):
        """Test log-likelihood."""

        eta = 100
        size = (10, 2)
        data = np.random.normal(size=size)
        mvst = MvSt(ndim=size[1], eta=eta, data=data)
        logl1 = mvst.loglikelihood(eta)
        logl2 = mvst.loglikelihood(eta * 2)

        self.assertIsInstance(logl1, float)
        self.assertNotEqual(logl1, logl2)
        npt.assert_array_equal(mvst.data, data)

    def test_rvs(self):
        """Test simulation."""

        eta = 100
        ndim = 2
        mvst = MvSt(ndim=ndim, eta=eta)
        size = 10
        rvs = mvst.rvs(size=size)

        self.assertEqual(rvs.shape, (size, ndim))


if __name__ == '__main__':
    ut.main()
