#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage examples

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from multidensity import SkStJR, SkStBL, SkStDM
from skewstudent import SkewStudent


def estimate_bivariate_mle_bl():
    ndim = 2
    size = (10000, ndim)
    data = np.random.normal(size=size)
    eta, lam = 4, -.9
    skst = SkewStudent(eta=eta, lam=lam)
    data = skst.rvs(size=size)

    model = SkStBL(data=data)
    out = model.fit_mle()
    print(out)


def estimate_bivariate_mle_jr():
    ndim = 2
    size = (10000, ndim)
    data = np.random.normal(size=size)
    eta, lam = 4, -.9
    skst = SkewStudent(eta=eta, lam=lam)
    data = skst.rvs(size=size)

    model = SkStJR(data=data)
    out = model.fit_mle()
    print(out)

    model.from_theta(out.x)

    fig, axes = plt.subplots(nrows=size[1], ncols=1)
    for innov, ax in zip(data.T, axes):
        sns.kdeplot(innov, ax=ax, label='data')

    lines = [ax.get_lines()[0].get_xdata() for ax in axes]
    lines = np.vstack(lines).T
    marginals = model.marginals(lines)

    for line, margin, ax in zip(lines.T, marginals.T, axes):
        ax.plot(line, margin, label='fitted')
        ax.legend()

    plt.show()


def plot_bidensity():

    eta, lam = [20, 5], [1.5, .5]
    skst = SkStJR(eta=eta, lam=lam)
    skst.plot_bidensity()

    eta, lam = 20, [1.5, .5]
    skst = SkStBL(eta=eta, lam=lam)
    skst.plot_bidensity()

    eta, lam = 20, [1.5, -2]
    skst = SkStDM(eta=eta, lam=lam)
    skst.plot_bidensity()


if __name__ == '__main__':

#    estimate_bivariate_mle_bl()
#    estimate_bivariate_mle_jr()
    plot_bidensity()