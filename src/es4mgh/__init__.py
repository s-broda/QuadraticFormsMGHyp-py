"""Tail probability and expected shortfall of a GH quadratic form."""

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from . import _native

try:
    __version__ = version("es4mgh")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["QuadraticForm"]


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
    """Quadratic form L = a0 + a'X + X'A X in a multivariate GH vector.

    X = mu + W * gam + sqrt(W) * C Z, with Z standard normal and
    W ~ GIG(lam, chi, psi). ``gam`` is the skewness vector gamma.
    Matrices are row-major.
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

        ``omega``, ``d``, and ``e`` are the eigenvalues and the two
        coefficient vectors. ``kk`` is added back when the conditional
        expectation is formed.
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
        """Return ``(ccdf, es)`` at the thresholds ``x``.

        ``ccdf[i] = P(L > x[i])`` and ``es[i] = E[L | L > x[i]]``.
        ``threads <= 0`` uses the performance-core count on Apple and
        the online CPU count elsewhere.
        """
        if self._cap is None:
            raise RuntimeError("this QuadraticForm has been closed")
        x = _as_vec(x, "x")
        return _native.eval(self._cap, x, int(threads))

    def close(self):
        """Release the C object. Later ``eval`` calls fail."""
        self._cap = None

    def __del__(self):
        try:
            self._cap = None
        except Exception:
            pass
