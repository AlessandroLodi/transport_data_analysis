"""Peak detection utilities."""

from __future__ import annotations

import numpy as np


def peakdet(v, delta, x=None):
    """Find local extrema separated by at least ``delta`` in amplitude.

    Parameters
    ----------
    v:
        One-dimensional signal values.
    delta:
        Positive scalar threshold used to distinguish a peak from noise.
    x:
        Optional coordinates for ``v``. Integer sample positions are used when
        omitted.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Arrays of ``(x, value)`` pairs for maxima and minima.
    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)
    x = np.asarray(x)

    if v.ndim != 1 or x.ndim != 1:
        raise ValueError("v and x must be one-dimensional")

    if len(v) != len(x):
        raise ValueError("v and x must have the same length")

    if not np.isscalar(delta):
        raise TypeError("delta must be a scalar")

    if delta <= 0:
        raise ValueError("delta must be positive")

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

    lookformax = True

    for i in range(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return _as_peak_array(maxtab), _as_peak_array(mintab)


def _as_peak_array(peaks):
    """Return a stable ``(n, 2)`` shape, including when no peaks are found."""
    return np.asarray(peaks, dtype=float).reshape(-1, 2)
