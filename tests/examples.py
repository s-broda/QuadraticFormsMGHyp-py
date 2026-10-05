"""Portfolio and two-stage least squares setups from QuadraticFormsMGHyp.

The builders follow test/es.jl and test/2sls.jl. They do not call Julia.
The boolean passed to the Black–Scholes routines is FinancialToolbox's
FlagIsCall: true means a call. In es.jl the comment says the opposite;
the call itself passes the vector as FlagIsCall.
"""

import math

import numpy as np

LOG2 = math.log(2.0)


def _norm_cdf(x):
    return math.erfc(-x / math.sqrt(2.0)) / 2.0


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _log_bessel_k(order, z):
    """log K_order(z) for a small half-integer order and z > 0."""
    n = int(round(abs(order) - 0.5))
    if n < 0 or n > 8 or abs(abs(order) - (n + 0.5)) > 1e-10:
        raise ValueError("Bessel K is only implemented for small half-integer orders")
    # K_{1/2}(z) = K_{-1/2}(z) = sqrt(pi/(2z)) exp(-z).
    k_prev = math.exp(-z + 0.5 * math.log(math.pi / (2.0 * z)))
    k_cur = k_prev
    nu = 0.5
    for _ in range(n):
        k_next = k_prev + (2.0 * nu / z) * k_cur
        k_prev = k_cur
        k_cur = k_next
        nu += 1.0
    return math.log(k_cur)


def lklam(lam, chi, psi):
    """Log of the GIG kernel, real and nonnegative arguments."""
    if chi == 0.0:
        return -lam * math.log(psi / 2.0) + math.lgamma(lam)
    if psi == 0.0:
        return lam * math.log(chi / 2.0) + math.lgamma(-lam)
    z = math.sqrt(chi * psi)
    return LOG2 + 0.5 * lam * math.log(chi / psi) + _log_bessel_k(lam, z)


def _call_sign(is_call):
    return 1.0 if is_call else -1.0


def blsprice(spot, strike, rate, maturity, sigma, dividend, is_call):
    discount = math.exp(rate * maturity)
    forward = spot * math.exp((rate - dividend) * maturity)
    sigma_sqrt = sigma * math.sqrt(maturity)
    d1 = math.log(forward / strike) / sigma_sqrt + sigma_sqrt / 2.0
    sign = _call_sign(is_call)
    undiscounted = sign * (
        forward * _norm_cdf(sign * d1) - strike * _norm_cdf(sign * (d1 - sigma_sqrt))
    )
    return undiscounted / discount


def blsdelta(spot, strike, rate, maturity, sigma, dividend, is_call):
    carry = -dividend * maturity
    sigma_sqrt = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + rate * maturity + carry) / sigma_sqrt + sigma_sqrt / 2.0
    sign = _call_sign(is_call)
    return math.exp(carry) * _norm_cdf(sign * d1) * sign


def blsgamma(spot, strike, rate, maturity, sigma, dividend, is_call):
    del is_call
    carry = -dividend * maturity
    sigma_sqrt = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + rate * maturity + carry) / sigma_sqrt + sigma_sqrt / 2.0
    return math.exp(carry) * _norm_pdf(d1) / (spot * sigma_sqrt)


def blstheta(spot, strike, rate, maturity, sigma, dividend, is_call):
    sqrt_t = math.sqrt(maturity)
    sigma_sqrt = sigma * sqrt_t
    d1 = (math.log(spot / strike) + (rate - dividend) * maturity) / sigma_sqrt + sigma_sqrt / 2.0
    d2 = d1 - sigma_sqrt
    spot_adj = spot * math.exp(-dividend * maturity)
    shift = -spot_adj * _norm_pdf(d1) * sigma / (sqrt_t * 2.0)
    rate_term = rate * strike * math.exp(-rate * maturity)
    dividend_term = dividend * spot_adj
    sign = _call_sign(is_call)
    return shift - sign * (
        rate_term * _norm_cdf(sign * d2) - dividend_term * _norm_cdf(sign * d1)
    )


def _mm(left, right):
    # einsum avoids a spurious Accelerate divide-by-zero warning in matmul.
    if np.asarray(right).ndim == 1:
        return np.einsum("ij,j->i", left, right)
    return np.einsum("ij,jk->ik", left, right)


def _psd_sqrt(matrix):
    symmetric = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(symmetric)
    scaled = evecs * np.sqrt(np.maximum(evals, 0.0))
    return np.einsum("ij,kj->ik", scaled, evecs)


def portfolio():
    """Short strangle portfolio from es.jl, portfolio = 1.

    Ten options, five puts then five calls, unhedged and uncorrelated.
    The threshold grid is 3.5, 3.51, ..., 17.5.
    """
    m = 10
    lam = -0.5
    chi = 1.0
    psi = 1.0
    spot = 100.0
    strike = 100.0
    rate = 0.05
    sigma = 0.3
    horizon = 1.0 / 252.0
    maturity = 126.0 / 252.0
    is_call = [False] * 5 + [True] * 5
    number = -np.ones(m)
    greeks = [
        (
            blsdelta(spot, strike, rate, maturity, sigma, 0.0, flag),
            blsgamma(spot, strike, rate, maturity, sigma, 0.0, flag),
            blstheta(spot, strike, rate, maturity, sigma, 0.0, flag),
        )
        for flag in is_call
    ]
    delta = np.array([row[0] for row in greeks])
    gamma = np.array([row[1] for row in greeks])
    theta = np.array([row[2] for row in greeks])
    variance = math.exp(lklam(lam + 1.0, chi, psi) - lklam(lam, chi, psi))
    scale = sigma * spot * math.sqrt(horizon)
    dispersion = np.diag(np.full(m, scale * scale / variance))
    return {
        "a0": float(-horizon * np.sum(number * theta)),
        "a": -delta * number,
        "A": np.diag(-0.5 * gamma * number),
        "C": _psd_sqrt(dispersion),
        "mu": np.zeros(m),
        "gam": np.zeros(m),
        "lam": lam,
        "chi": chi,
        "psi": psi,
        "x": np.linspace(3.5, 17.5, 1401),
        "variance": variance,
        "scale": scale,
    }


def twosls():
    """Distribution of the just-identified 2SLS estimator, nu = 3 and 9.

    Z is replaced by its norm times the first unit vector, which is the
    only invariant the quadratic form depends on when the correlation
    matrix is the identity. Each case is evaluated at threshold 0; the
    estimator value is carried by a0, a, and A.
    """
    n = 25
    strength = 0.5
    instrument = np.zeros(n)
    instrument[0] = math.sqrt(strength)
    projection = np.zeros((n, n))
    projection[0, 0] = 1.0
    eye = np.eye(n)
    zeros = np.zeros((n, n))
    cases = []
    for nu in (3.0, 9.0):
        cov = ((nu - 2.0) / nu) * np.ones((2, 2))
        factor = _psd_sqrt(cov)
        scale = np.kron(factor, eye)
        for b in np.linspace(-3.0, 3.0, 61):
            slope = float(b)
            a_vec = np.concatenate([instrument, -2.0 * slope * instrument])
            quad = np.block(
                [
                    [zeros, 0.5 * projection],
                    [0.5 * projection, -slope * projection],
                ]
            )
            cases.append(
                {
                    "nu": nu,
                    "b": slope,
                    "a0": -slope * strength,
                    "a": a_vec,
                    "A": quad,
                    "C": scale,
                    "mu": np.zeros(2 * n),
                    "gam": np.zeros(2 * n),
                    "lam": -0.5 * nu,
                    "chi": nu,
                    "psi": 0.0,
                    "x": np.zeros(1),
                    "factor": factor,
                    "cov": cov,
                }
            )
    return cases
