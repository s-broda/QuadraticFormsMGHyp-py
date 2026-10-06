"""Tail probability and expected shortfall of a GH quadratic form.

The loss is ``L = a0 + a'X + X'A X``, with
``X = mu + W gam + sqrt(W) C Z``, ``Z`` standard normal, and
``W ~ GIG(lam, chi, psi)``. Construct one ``QuadraticForm`` and call
``eval`` on each threshold grid. One call integrates once and returns
the cdf, the survival function, the upper partial moment, and the
expected shortfall.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from . import _native

try:
    __version__ = version("QuadraticFormsMGHyp")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["QuadraticForm", "__version__"]


def _scalar(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a real scalar" % name) from exc


def _as_vec(value, name, length=None):
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("%s must be a 1-d vector" % name)
    if length is not None and arr.shape[0] != length:
        raise ValueError("%s must have length %d" % (name, length))
    return arr


def _as_mat(value, d, name):
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (d, d):
            raise ValueError("%s must have shape (%d, %d)" % (name, d, d))
        return arr.reshape(-1)
    if arr.ndim == 1 and arr.shape[0] == d * d:
        return arr
    raise ValueError(
        "%s must be a (%d, %d) matrix or a flat vector of length %d"
        % (name, d, d, d * d)
    )


class QuadraticForm(object):
    """Quadratic form ``L = a0 + a'X + X'A X`` in a multivariate GH vector.

    ``X = mu + W gam + sqrt(W) C Z``, with ``Z`` standard normal and
    ``W ~ GIG(lam, chi, psi)``. ``gam`` is the skewness vector γ.
    ``a``, ``mu``, and ``gam`` have length ``d``. ``A`` and ``C`` are
    ``d × d``, either shaped ``(d, d)`` or flattened in row-major order.

    The constructor diagonalizes the form. Repeat ``eval`` on one
    instance when several threshold grids share that reduction. The
    object releases the C state when it is closed, when a ``with``
    block ends, or when it is collected.

    Parameters
    ----------
    a0, lam, chi, psi : float
        Constant term and the GIG parameters λ, χ, and ψ.
        ``chi = psi = inf`` is the Gaussian limit: the mixer is the
        constant 1, for any ``lam``. A low-rank Gaussian spectrum is
        integrated in panels.
    a, mu, gam : array_like
        Length-``d`` vectors.
    A, C : array_like
        ``d × d`` matrices.

    Attributes
    ----------
    dimension : int or None
        Length of ``a``. ``None`` after ``from_spectral``.
    ne : int or None
        Number of retained spectral terms. ``None`` until
        ``from_spectral``.
    """

    def __init__(self, a0, a, A, C, mu, gam, lam, chi, psi):
        a = _as_vec(a, "a")
        if a.shape[0] == 0:
            raise ValueError("a must be a non-empty vector")
        d = int(a.shape[0])
        A = _as_mat(A, d, "A")
        C = _as_mat(C, d, "C")
        mu = _as_vec(mu, "mu", d)
        gam = _as_vec(gam, "gam", d)
        self._cap = _native.create(
            _scalar(a0, "a0"),
            a,
            A,
            C,
            mu,
            gam,
            _scalar(lam, "lam"),
            _scalar(chi, "chi"),
            _scalar(psi, "psi"),
        )
        self.dimension = d
        self.ne = None

    @classmethod
    def from_spectral(cls, omega, d, e, c, k, kk, lam, chi, psi):
        """Build from the spectral reduction used by the integral.

        Parameters
        ----------
        omega, d, e : array_like
            Eigenvalues and the two coefficient vectors, same length.
        c, k, kk, lam, chi, psi : float
            The remaining scalar coefficients. ``kk`` is added back
            when the conditional expectation is formed.

        Returns
        -------
        QuadraticForm
        """
        omega = _as_vec(omega, "omega")
        d = _as_vec(d, "d", omega.shape[0])
        e = _as_vec(e, "e", omega.shape[0])
        self = cls.__new__(cls)
        self._cap = _native.create_spectral(
            omega,
            d,
            e,
            _scalar(c, "c"),
            _scalar(k, "k"),
            _scalar(kk, "kk"),
            _scalar(lam, "lam"),
            _scalar(chi, "chi"),
            _scalar(psi, "psi"),
        )
        self.dimension = None
        self.ne = int(omega.shape[0])
        return self

    def eval(self, x, threads=0):
        """Integrate once and return the distribution and both moments.

        Parameters
        ----------
        x : array_like
            One-dimensional thresholds.
        threads : int, optional
            Worker count on the scalar path. ``threads <= 0`` uses
            one worker per logical CPU. NIG, ψ = 0, and the general-GH series evaluate
            the vector on the calling thread. The Gaussian limit does too.
            A low-rank Gaussian spectrum is integrated in panels.
            The node count doubles until successive
            refinements agree. More than 24 thresholds are reduced to a Chebyshev grid first.

        Returns
        -------
        cdf, ccdf, pm, es : ndarray
            ``cdf[i] = P(L <= x[i])``, ``ccdf[i] = P(L > x[i])``,
            ``pm[i] = E[L 1_{L > x[i]}]``, and
            ``es[i] = E[L | L > x[i]]``. The cdf and the partial
            moment are formed from the survival function and the
            expected shortfall of this same pass.
        """
        if self._cap is None:
            raise RuntimeError("this QuadraticForm has been closed")
        x = _as_vec(x, "x")
        ccdf, es = _native.eval(self._cap, x, int(threads))
        return 1.0 - ccdf, ccdf, es * ccdf, es

    def close(self):
        """Release the C object. A later ``eval`` raises ``RuntimeError``."""
        self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __repr__(self):
        if self.dimension is not None:
            return "QuadraticForm(dimension=%d)" % self.dimension
        return "QuadraticForm(ne=%s)" % self.ne

    def __del__(self):
        try:
            self._cap = None
        except Exception:
            pass
