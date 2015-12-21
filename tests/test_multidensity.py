#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for MultiDensity class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from multidensity import MultiDensity


class MultiDensityTestCase(ut.TestCase):

    """Test MultiDensity distribution class."""

    def test_init(self):
        """Test __init__."""

        skst = MultiDensity()

        self.assertIsInstance(skst.eta, np.ndarray)
        self.assertIsInstance(skst.lam, np.ndarray)

        eta, lam = [10, 15], [.5, 1.5]
        skst = MultiDensity(eta=eta, lam=lam)

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

        skst = MultiDensity.from_theta(np.concatenate((eta, lam)))

        npt.assert_array_equal(skst.eta, np.array(eta))
        npt.assert_array_equal(skst.lam, np.array(lam))

    def test_pdf(self):
        """Test pdf."""

        skst = MultiDensity()
        pdf = skst.pdf()

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (1,))

        size = (10, 2)
        arg = np.random.normal(size=size)
        pdf = skst.pdf(arg)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0],))

        eta, lam = [10, 15, 10], [.5, 1.5, 2]
        theta = np.concatenate((eta, lam))
        size = (10, len(eta))
        arg = np.random.normal(size=size)
        pdf = skst.pdf(arg, theta=theta)

        self.assertEqual(pdf.ndim, 1)
        self.assertEqual(pdf.shape, (size[0],))


if __name__ == '__main__':
    ut.main()
