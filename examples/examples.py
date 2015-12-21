#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage examples

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from multidensity import MultiDensity


if __name__ == '__main__':

    skst = MultiDensity()
    print(skst.pdf())

    arg = [1, 1]
    print(skst.pdf(arg))

    arg = np.zeros((10, 2))
    print(skst.pdf(arg))