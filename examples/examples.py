#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage examples

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from multidensity import MultiDensity
from skewstudent import SkewStudent


def estimate_bivariate_mle():
    ndim = 2
    size = (1000, ndim)
    data = np.random.normal(size=size)
    eta, lam = 4, -.9
    skst = SkewStudent(eta=eta, lam=lam)
    data = skst.rvs(size=size)

    out = MultiDensity.fit_mle(data=data)
    print(out)

    mdens = MultiDensity()
    mdens.from_theta(out.x)

    fig, axes = plt.subplots(nrows=size[1], ncols=1)
    for innov, ax in zip(data.T, axes):
        sns.kdeplot(innov, ax=ax)

    lines = [ax.get_lines()[0].get_xdata() for ax in axes]
    lines = np.vstack(lines).T
    marginals = mdens.marginals(lines)

    for line, margin, ax in zip(lines.T, marginals.T, axes):
        ax.plot(line, margin)

    plt.show()


if __name__ == '__main__':

    estimate_bivariate_mle()
