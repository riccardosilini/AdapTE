# -*- coding: utf-8 -*-

# Copyright (c) 2021 Riccardo Silini

"""Functions to compute pseudo transfer entropy (pTE).

This module provides a set of functions to compute pTE between different
time series.

Functions
---------------------

  * normalisa -- L2 normalization, can be replaced by the
    sklearn.preprocessing.normalize(*args) function
  * embed -- generates matrices containing segments of the original time
    series, depending on the embedding size chosen.
  * timeshifted -- creeates time shifted surrogates. The sign on the shift means
    that the time series that must be shifted is the independent one
  * pTE -- Computes the pseudo transfer entropy between time series.
  * AdapTE -- computes the maximum pseudo transfer entropy between time series, over different time lags and embeddings

Libraries required
---------------------
import numpy as np
import scipy.signal as sps
import pandas as pd
from collections import deque

"""
import numpy as np
import scipy.signal as sps
import pandas as pd
from collections import deque


def AdapTE(z, taus=None, dimEmbs=None, which=None, mode=None, surr=None, Nsurr=19):
    if taus is None:
        taus = [1]
    if dimEmbs is None:
        dimEmbs = [1]
    if mode is None:
        mode = 'driving'
    if isinstance(z, pd.DataFrame):
        z = z.to_numpy()

    pte_list = []
    surr_list = []
    for dimEmb in dimEmbs:
        print('embedding : ', dimEmb)
        for tau in taus:
            print('tau : ', tau)
            locals()['pte_' + str(tau)], locals()['surr_' + str(tau)] = pTE(z=z, tau=tau, dimEmb=dimEmb, which=which, mode=mode, surr=surr, Nsurr=Nsurr)

            pte_list.append(locals()['pte_' + str(tau)])
            surr_list.append(locals()['surr_' + str(tau)])

    adpte = np.maximum.reduce(pte_list)
    surrogate = np.maximum.reduce(surr_list)

    return adpte, surrogate


def normalisa(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def embed(x, embd, lag):
    N = len(x)
    hidx = np.arange(embd * lag, step=lag)
    vidx = np.arange(N - (embd - 1) * lag)
    vidx = vidx.T
    Nv = len(vidx)
    U = np.array([x, ] * embd)
    W = np.array([hidx, ] * Nv).T + np.array([vidx, ] * embd)
    u = np.zeros((embd, Nv))
    for i in range(embd):
        for j in range(Nv):
            u[i, j] = U[i, W[i, j]]
    return u.T


def timeshifted(timeseries, shift):
    ts = deque(timeseries)
    ts.rotate(shift)
    return np.asarray(ts)


def iaaft(x, maxiter=1000, atol=1e-8, rtol=1e-10):
    """Return iterative amplitude adjusted Fourier transform surrogates.
    this function have been taken from teh NoLiTSA package,
    Copyright (c) 2015-2016, Manu Mannattil.
    All rights reserved.

    Returns phase randomized, amplitude adjusted (IAAFT) surrogates with
    the same power spectrum (to a very high accuracy) and distribution
    as the original data using an iterative scheme (Schreiber & Schmitz
    1996).

    Parameters
    ----------
    x : array
        1-D real input array of length N containing the time series.
    maxiter : int, optional (default = 1000)
        Maximum iterations to be performed while checking for
        convergence.  The scheme may converge before this number as
        well (see Notes).
    atol : float, optional (default = 1e-8)
        Absolute tolerance for checking convergence (see Notes).
    rtol : float, optional (default = 1e-10)
        Relative tolerance for checking convergence (see Notes).

    Returns
    -------
    y : array
        Surrogate series with (almost) the same power spectrum and
        distribution.
    i : int
        Number of iterations that have been performed.
    e : float
        Root-mean-square deviation (RMSD) between the absolute squares
        of the Fourier amplitudes of the surrogate series and that of
        the original series.

    Notes
    -----
    To check if the power spectrum has converged, we see if the absolute
    difference between the current (cerr) and previous (perr) RMSDs is
    within the limits set by the tolerance levels, i.e., if abs(cerr -
    perr) <= atol + rtol*perr.  This follows the convention used in
    the NumPy function numpy.allclose().

    Additionally, atol and rtol can be both set to zero in which
    case the iterations end only when the RMSD stops changing or when
    maxiter is reached.
    """
    # Calculate "true" Fourier amplitudes and sort the series.
    ampl = np.abs(np.fft.rfft(x))
    sort = np.sort(x)

    # Previous and current error.
    perr, cerr = (-1, 1)

    # Start with a random permutation.
    t = np.fft.rfft(np.random.permutation(x))

    for i in range(maxiter):
        # Match power spectrum.
        s = np.real(np.fft.irfft(ampl * t / np.abs(t), n=len(x)))

        # Match distribution by rank ordering.
        y = sort[np.argsort(np.argsort(s))]

        t = np.fft.rfft(y)
        cerr = np.sqrt(np.mean((ampl ** 2 - np.abs(t) ** 2) ** 2))

        # Check convergence.
        if abs(cerr - perr) <= atol + rtol * abs(perr):
            break
        else:
            perr = cerr

    # Normalize error w.r.t. mean of the "true" power spectrum.
    return y, i, cerr / np.mean(ampl ** 2)


def pTE(z, tau=1, dimEmb=1, which=None, mode='driving', surr=None, Nsurr=19):
    """Returns pseudo transfer entropy.

    Parameters
    ----------
    z : array
        array of arrays, containing all the time series.
    tau : integer
        delay of the embedding.
    dimEMb : integer
        embedding dimension, or model order.
    which : list
        list of the time series of interest
    mode : string
        if 'driving' (default) it computes the influnce of the time series selected with the which variable to all the others
        if 'driven' it computes the influence of all time series to the selected ones with the which variable
    surr : string
        if 'ts' it computes the maximum value obtained using 19 times shifted
        surrogates
        if 'iaaft' it computes the maximum value obtained using 19 times shifted
        surrogates

    Returns
    -------
    pte : array
        array of arrays. The dimension is (# time series, # time series).
        The diagonal is 0, while the off diagonal term (i, j) corresponds
        to the pseudo transfer entropy from time series i to time series j.
    ptesurr : array
        array of arrays. The dimension is (# time series, # time series).
        The diagonal is 0, while the off diagonal term (i, j) corresponds
        to the pseudo transfer entropy from time series i to surrogate time
        series j.
    In case of surrogates it returns pte and the maximum value obtained with
    surrogares ptesurr
    """

    NN, T = np.shape(z)
    Npairs = NN * (NN - 1)
    pte = np.zeros((NN, NN))
    ptesurr = np.zeros((NN, NN))
    z = normalisa(sps.detrend(z))
    channels = np.arange(NN, step=1)
    channelsi = channels
    channelsj = channels

    if which != None:
        channelsi = channels[which]
        channelsj = channels
        if mode == 'driven':
            channelsi = channels
            channelsj = channels[which]


    for i in channelsi:
        EmbdDumm = embed(z[i], dimEmb + 1, tau)
        Xtau = EmbdDumm[:, :-1]
        for j in channelsj:
            if i != j:
                Yembd = embed(z[j], dimEmb + 1, tau)
                Y = Yembd[:, -1]
                Ytau = Yembd[:, :-1]
                XtYt = np.concatenate((Xtau, Ytau), axis=1)
                YYt = np.concatenate((Y[:, np.newaxis], Ytau), axis=1)
                YYtXt = np.concatenate((YYt, Xtau), axis=1)

                if dimEmb > 1:
                    ptedum = np.linalg.det(np.cov(XtYt.T)) * np.linalg.det(np.cov(YYt.T)) / (
                            np.linalg.det(np.cov(YYtXt.T)) * np.linalg.det(np.cov(Ytau.T)))
                else:
                    ptedum = np.linalg.det(np.cov(XtYt.T)) * np.linalg.det(np.cov(YYt.T)) / (
                            np.linalg.det(np.cov(YYtXt.T)) * np.cov(Ytau.T))

                pte[i, j] = 0.5 * np.log(ptedum)

    if surr != None:
        surrogate = np.zeros((NN, Nsurr, T))
        if surr == 'ts':
            for k in range(NN):
                for n in range(Nsurr):
                    surrogate[k, n] = timeshifted(z[k], -(n + dimEmb + 1))
        if surr == 'iaaft':
            for k in range(NN):
                for n in range(Nsurr):
                    surrogate[k, n], a, b = iaaft(z[k])
        for i in channelsi:
            EmbdDumm = embed(z[i], dimEmb + 1, tau)
            Xtau = EmbdDumm[:, :-1]
            for j in channelsj:
                if i != j:
                    ptedumold = float('-inf')
                    for n in range(Nsurr):
                        Yembd = embed(surrogate[j, n], dimEmb + 1, tau)
                        Y = Yembd[:, -1]
                        Ytau = Yembd[:, :-1]
                        XtYt = np.concatenate((Xtau, Ytau), axis=1)
                        YYt = np.concatenate((Y[:, np.newaxis], Ytau), axis=1)
                        YYtXt = np.concatenate((YYt, Xtau), axis=1)

                        if dimEmb > 1:
                            ptedum = np.linalg.det(np.cov(XtYt.T)) * np.linalg.det(np.cov(YYt.T)) / (
                                    np.linalg.det(np.cov(YYtXt.T)) * np.linalg.det(np.cov(Ytau.T)))
                        else:
                            ptedum = np.linalg.det(np.cov(XtYt.T)) * np.linalg.det(np.cov(YYt.T)) / (
                                    np.linalg.det(np.cov(YYtXt.T)) * np.cov(Ytau.T))
                        if ptedum > ptedumold:
                            ptedumold = ptedum
                    ptesurr[i, j] = 0.5 * np.log(ptedumold)
    if mode == 'driving':
        return pte[which], ptesurr[which]
    if mode == 'driven':
        return pte[:, which], ptesurr[:, which]