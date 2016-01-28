#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage examples

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from multidensity import SkStJR, SkStBL, SkStDM, SkStAC, MvSt, MvSN
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

    lam = [1.5, -2]
    mvsn = MvSN(lam=lam)
    mvsn.plot_bidensity()

    eta = 20
    skst = MvSt(eta=eta)
    skst.plot_bidensity()

    eta, lam = [20, 5], [1.5, .5]
    skst = SkStJR(eta=eta, lam=lam)
    skst.plot_bidensity()

    eta, lam = 20, [1.5, .5]
    skst = SkStBL(eta=eta, lam=lam)
    skst.plot_bidensity()

    eta, lam = 20, [1.5, -2]
    skst = SkStDM(eta=eta, lam=lam)
    skst.plot_bidensity()

    eta, lam = 20, [1.5, -2]
    skst = SkStAC(eta=eta, lam=lam)
    skst.plot_bidensity()


def plot_bidensity_skstdm():

    eta, lam = 20, [1.5, -2]
    sigma = [[1, -.5], [-.5, 1]]
    sigma = None
    skst = SkStDM(eta=eta, lam=lam, sigma=sigma)
    skst.plot_bidensity()

    rvs = skst.rvs(size=int(1e4))
    sns.kdeplot(rvs, shade=True)
    plt.axis('square')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


def compute_cdf():
    eta, lam = [20, 5], [1.5, .5]
    skst = SkStJR(eta=eta, lam=lam)
    print(skst.cdf(np.zeros(2)))

    eta, lam = 20, [1.5, .5]
    skst = SkStBL(eta=eta, lam=lam)
    print(skst.cdf(np.zeros(2)))

    eta, lam = 100, [1.5, -2]
    skst = SkStDM(eta=eta, lam=lam)
    print(skst.cdf(np.zeros(2)))


def compute_univ_cdf():
    eta, lam = 20, 1.5
    skst = SkStJR(eta=eta, lam=lam)
    print(skst.cdf(np.zeros(1)))

    eta, lam = 20, 1.5
    skst = SkStBL(eta=eta, lam=lam)
    print(skst.cdf(np.zeros(1)))

    eta, lam = 100, 1.5
    skst = SkStDM(eta=eta, lam=lam)
    print(skst.cdf(np.zeros(1)+10))


def compute_quantile():
    eta, lam = 20, 1.5
    skst = SkStJR(eta=eta, lam=lam)
    cdf = skst.cdf(np.zeros(1) - 2)
    print(skst.ppf(cdf))

    eta, lam = 20, 1.5
    skst = SkStBL(eta=eta, lam=lam)
    cdf = skst.cdf(np.zeros(1) - 2)
    print(skst.ppf(cdf))

    eta, lam = 100, 1.5
    skst = SkStDM(eta=eta, lam=lam)
    cdf = skst.cdf(np.zeros(1) - 2)
    print(skst.ppf(cdf))


def likelihood(model_univ, model_mult, data):
    data_marg = skst_univ.pdf_vec(data)
    cdfs_marg = skst_univ.cdf_vec(data)
    quantiles = skst_univ.ppf_vec(cdfs_marg)
    cop_marg = skst_univ.pdf_vec(quantiles)
    copula_density = skst_mult.pdf(quantiles) / np.prod(cop_marg, axis=1)
    return np.log(copula_density) + np.log(np.prod(data_marg, axis=1))


def compute_copula_likelihood():
    eta, lam = 100, 1.5
    skst_univ = SkStDM(eta=eta, lam=lam)

    eta, lam = 100, [1.5, -2]
    skst_mult = SkStDM(eta=eta, lam=lam)

    data = np.random.normal(size=(10, 2))

    ll = likelihood(skst_univ, skst_mult, data)
    print(ll)


if __name__ == '__main__':

#    estimate_bivariate_mle_bl()
#    estimate_bivariate_mle_jr()
    plot_bidensity()
#    plot_bidensity_skstdm()
#    compute_cdf()
#    compute_univ_cdf()
#    compute_quantile()
#    compute_copula_likelihood()
