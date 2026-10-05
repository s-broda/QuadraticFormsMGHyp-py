"""Independent 96-node quadrature of the Gil-Pelaez integral.

Nodes come from numpy's Legendre rule. The map u = (v/(1-v))^2 and the
truncation bound are the same ones the library uses, so a discrepancy is
a discrepancy in the integral, not a different cutoff. The weight already
divides by u, and u is real, which is imag(exp(log theta) / u).
"""

import math

import numpy as np

from examples import _mm, lklam

INV_PI = 1.0 / math.pi
LOG2 = math.log(2.0)


def integration_ub(omega):
    """Truncation of the mapped interval, before the nearest pole."""
    magnitude = np.abs(np.asarray(omega, dtype=np.float64))
    magnitude = np.sort(magnitude[magnitude > 0.0])[::-1]
    epsabs = 1e-10
    ub = 1.0
    logsum = 0.0
    log_pi = math.log(math.pi)
    log4 = math.log(4.0)
    for index, entry in enumerate(magnitude, start=1):
        logsum += math.log(2.0 * entry)
        exponent = (-1.0 / index) * (
            2.0 * log_pi + 2.0 * math.log(index) + 2.0 * math.log(epsabs) - log4 + logsum
        )
        root = math.exp(0.5 * exponent)
        candidate = root / (1.0 + root)
        if candidate < ub:
            ub = candidate
    if not (ub > 1e-6 and ub < 1.0):
        ub = 0.99
    if ub > 0.999999:
        ub = 0.999999
    return ub


def _spectral(a0, a, quad, scale, mu, gam):
    symmetric = 0.5 * (quad + quad.T)
    gram = _mm(_mm(scale.T, symmetric), scale)
    gram = 0.5 * (gram + gram.T)
    omega, vectors = np.linalg.eigh(gram)
    mu_quad = _mm(symmetric, mu)
    gam_quad = _mm(symmetric, gam)
    mixed = _mm(scale, vectors)
    c_coef = float(a @ gam + 2.0 * mu_quad @ gam)
    kk = float(a0 + a @ mu + mu_quad @ mu)
    k_coef = float(gam_quad @ gam)
    d_coef = _mm(mixed.T, a + 2.0 * mu_quad)
    e_coef = _mm(mixed.T, 2.0 * gam_quad)
    return omega, d_coef, e_coef, c_coef, k_coef, kk


def _mapped_nodes(omega, nnode):
    ub = integration_ub(omega)
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(nnode)
    v = 0.5 * ub * (gauss_x + 1.0)
    span = 1.0 - v
    ratio = v / span
    u = ratio * ratio
    dudv = 2.0 * ratio / (span * span)
    weight = (0.5 * ub * gauss_w) * dudv / u
    return u, weight


def _sums(omega, d_coef, e_coef, c_coef, k_coef, lam, chi, psi, nnode):
    u, weight = _mapped_nodes(omega, nnode)
    s = 1j * u
    s2 = s * s
    nu = 1.0 / (1.0 - 2.0 * omega[:, None] * s[None, :])
    nu2 = nu * nu
    d2 = (d_coef * d_coef)[:, None]
    e2 = (e_coef * e_coef)[:, None]
    de = (d_coef * e_coef)[:, None]
    om = omega[:, None]
    t1 = np.sum(d2 * nu, axis=0)
    t2 = np.sum(e2 * nu, axis=0)
    t3 = np.sum(de * nu, axis=0)
    t4 = np.sum(np.log(nu), axis=0)
    a2p = np.sum(s * d2 * nu + s2 * d2 * om * nu2, axis=0)
    a1p = np.sum(s * e2 * nu + s2 * e2 * om * nu2, axis=0) + k_coef
    lrp = np.sum(2.0 * s * de * nu + 2.0 * s2 * de * om * nu2 + om * nu, axis=0) + c_coef
    return {
        "u": u,
        "weight": weight,
        "a2p": a2p,
        "a1p": a1p,
        "lrp": lrp,
        "chi_base": chi - s2 * t1,
        "psi_node": psi - 2.0 * (k_coef * s + 0.5 * s2 * t2),
        "lrho": s * c_coef + s2 * t3 + 0.5 * t4,
        "lam": lam,
        "lk2": lklam(lam, chi, psi),
        "psi": psi,
        "k_coef": k_coef,
        "e_coef": e_coef,
    }


def _anchor(omega, c_coef, k_coef, lam, chi, psi, lk2):
    first = lklam(lam + 1.0, chi, psi) - lk2
    second = lklam(lam + 2.0, chi, psi) - lk2
    return math.exp(second) * k_coef + math.exp(first) * (c_coef + float(np.sum(omega)))


def _imag_exp(log_value, weight):
    return np.sum(weight[:, None] * np.imag(np.exp(log_value)), axis=0)


def _integrate(prepared, q):
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    chi = prepared["chi_base"][:, None] + 1j * (2.0 * prepared["u"][:, None] * q[None, :])
    lrho = prepared["lrho"][:, None]
    lam = prepared["lam"]
    lk2 = prepared["lk2"]
    if prepared["psi"] == 0.0:
        log_chi = np.log(chi)
        def level(order):
            return order * log_chi + math.lgamma(-order) - order * LOG2 - lk2 + lrho
    else:
        z = np.sqrt(chi * prepared["psi_node"][:, None])
        log_k = -z + 0.5 * np.log(math.pi / (2.0 * z))
        ratio = np.log(chi / prepared["psi_node"][:, None])
        def level(order):
            bessel = log_k if abs(abs(order) - 0.5) < 1e-12 else log_k + np.log(1.0 + 1.0 / z)
            return LOG2 + 0.5 * order * ratio + bessel - lk2 + lrho
    lm0 = level(lam)
    lm1 = level(lam + 1.0)
    acc = np.exp(lm0) * prepared["a2p"][:, None] + np.exp(lm1) * prepared["lrp"][:, None]
    if np.any(prepared["a1p"] != 0.0):
        acc = acc + np.exp(level(lam + 2.0)) * prepared["a1p"][:, None]
    weight = prepared["weight"]
    ic = _imag_exp(lm0, weight)
    ip = np.sum(weight[:, None] * np.imag(acc), axis=0)
    return ic, ip


def gil_pelaez(problem, nnode=96):
    """Return (ccdf, es) for one example dict from tests.examples."""
    omega, d_coef, e_coef, c_coef, k_coef, kk = _spectral(
        problem["a0"], problem["a"], problem["A"], problem["C"], problem["mu"], problem["gam"]
    )
    prepared = _sums(
        omega, d_coef, e_coef, c_coef, k_coef,
        problem["lam"], problem["chi"], problem["psi"], nnode,
    )
    if not (
        prepared["psi"] == 0.0
        or (abs(problem["lam"] + 0.5) < 1e-12 and problem["chi"] > 0.0 and problem["psi"] > 0.0)
    ):
        raise NotImplementedError("the reference covers the NIG portfolio and the psi = 0 laws")
    ic, ip = _integrate(prepared, np.asarray(problem["x"], dtype=np.float64) - kk)
    m20 = _anchor(omega, c_coef, k_coef, problem["lam"], problem["chi"], problem["psi"], prepared["lk2"])
    ccdf = 0.5 + INV_PI * ic
    es = (0.5 * m20 + INV_PI * ip) / ccdf + kk
    return ccdf, es
