"""Compare the two QuadraticFormsMGHyp examples with a 96-node quadrature."""

import numpy as np
import pytest

import QuadraticFormsMGHyp

from examples import blsdelta, blsgamma, blsprice, blstheta, portfolio, twosls
from quadrature import gil_pelaez


def test_black_scholes_matches_financial_toolbox():
    args = (10.0, 10.0, 0.01, 2.0, 0.2, 0.01, True)
    assert blsprice(*args) == pytest.approx(1.1023600107733191, abs=1e-12)
    assert blsdelta(*args) == pytest.approx(0.5452173371920436, abs=1e-12)
    assert blsgamma(*args) == pytest.approx(0.13687881535712826, abs=1e-12)
    assert blstheta(*args) == pytest.approx(-0.26273403060652334, abs=1e-12)


def test_portfolio_matches_quadrature():
    problem = portfolio()
    # chi = psi and |lambda| = 1/2, so the variance factor is 1 and C is diagonal.
    assert problem["variance"] == pytest.approx(1.0, abs=1e-12)
    assert problem["C"] == pytest.approx(np.diag(np.full(10, problem["scale"])), abs=1e-12)

    fit = QuadraticFormsMGHyp.QuadraticForm(
        problem["a0"], problem["a"], problem["A"], problem["C"],
        problem["mu"], problem["gam"], problem["lam"], problem["chi"], problem["psi"],
    )
    cdf, ccdf, pm, es = fit.eval(problem["x"], threads=1)
    assert cdf == pytest.approx(1.0 - ccdf, abs=0.0)
    assert pm == pytest.approx(es * ccdf, abs=0.0)
    ref_ccdf, ref_es = gil_pelaez(problem)
    ccdf_gap = float(np.max(np.abs(ccdf - ref_ccdf)))
    es_gap = float(np.max(np.abs(es - ref_es)))
    # 32 nodes match the probability to about 1e-8. Expected shortfall on
    # this grid reaches about 4e-4 at the last threshold, where that
    # probability is about 4e-4. Above 1% it stays under about 1e-6.
    body = ref_ccdf > 0.01
    body_gap = float(np.max(np.abs(es[body] - ref_es[body])))
    assert ccdf_gap < 1e-7, "max |ccdf error| = %g" % ccdf_gap
    assert body_gap < 5e-6, "max |es error| above 1%% = %g" % body_gap
    assert es_gap < 5e-4, "max |es error| = %g" % es_gap

    _, parallel_ccdf, _, parallel_es = fit.eval(problem["x"], threads=0)
    assert np.max(np.abs(parallel_ccdf - ccdf)) < 1e-12
    assert np.max(np.abs(parallel_es - es)) < 1e-9


def test_twosls_matches_quadrature():
    cases = twosls()
    assert len(cases) == 122
    first = cases[0]
    assert first["a"].shape == (50,)
    assert first["A"].shape == (50, 50)
    rebuilt = np.einsum("ij,jk->ik", first["factor"], first["factor"])
    assert rebuilt == pytest.approx(first["cov"], abs=1e-12)
    assert abs(first["a"][0] ** 2 - 0.5) < 1e-15

    ccdf_gap = 0.0
    es_gap = 0.0
    for problem in cases:
        fit = QuadraticFormsMGHyp.QuadraticForm(
            problem["a0"], problem["a"], problem["A"], problem["C"],
            problem["mu"], problem["gam"], problem["lam"], problem["chi"], problem["psi"],
        )
        _, ccdf, _, es = fit.eval(problem["x"], threads=1)
        ref_ccdf, ref_es = gil_pelaez(problem)
        ccdf_gap = max(ccdf_gap, float(np.max(np.abs(ccdf - ref_ccdf))))
        es_gap = max(es_gap, float(np.max(np.abs(es - ref_es))))
    assert ccdf_gap < 1e-8, "max |ccdf error| = %g" % ccdf_gap
    # Windows ARM clang-cl reached about 1.4e-7 against the 96-node rule.
    assert es_gap < 5e-7, "max |es error| = %g" % es_gap


def test_zero_skew_drops_infinite_second_moment():
    # lambda = -2, psi = 0, gamma = 0, so k = gamma' A gamma is exactly zero.
    # E[W^2] is infinite. E[L] = a0 + E[W] * trace(A) = 0.1 + 2 * 1.3 = 2.7.
    a0 = 0.1
    a = np.array([0.2, -0.1, 0.0, 0.3])
    A = np.diag([0.4, 0.0, 0.2, 0.7])
    z = np.zeros(4)
    fit = QuadraticFormsMGHyp.QuadraticForm(
        a0, a, A, np.eye(4), z, z, -2.0, 4.0, 0.0,
    )
    cdf, ccdf, pm, es = fit.eval(-1.0, threads=1)
    assert cdf == pytest.approx(1.0 - ccdf, abs=0.0)
    assert ccdf == pytest.approx(0.9999998269559728, abs=1e-12)
    assert es == pytest.approx(2.700000681379396, abs=1e-9)
    assert np.isfinite(es)
    assert pm == pytest.approx(es * ccdf, abs=0.0)
    _, grid_ccdf, _, grid_es = fit.eval(np.linspace(-1.0, 8.0, 50), threads=1)
    assert np.all(np.isfinite(grid_ccdf))
    assert np.all(np.isfinite(grid_es))


def test_concentrated_nig_uses_the_refined_rule():
    # chi = psi = 400. A fixed 32-node rule is about 0.016 high on the
    # survival and about 0.2 low on the shortfall.
    z = np.zeros(2)
    fit = QuadraticFormsMGHyp.QuadraticForm(
        0.0, z, np.diag([1.0, 0.5]), np.eye(2), z, z, -0.5, 400.0, 400.0,
    )
    _, ccdf, _, es = fit.eval(2.0, threads=1)
    assert ccdf == pytest.approx(0.2572066792989306, abs=1e-9)
    assert es == pytest.approx(3.6180484695108244, abs=1e-8)


def test_eval_reuses_the_object_and_close_rejects_another_call():
    problem = portfolio()
    args = (
        problem["a0"], problem["a"], problem["A"], problem["C"],
        problem["mu"], problem["gam"], problem["lam"], problem["chi"], problem["psi"],
    )
    grid = problem["x"][:5]
    with QuadraticFormsMGHyp.QuadraticForm(*args) as fit:
        assert repr(fit) == "QuadraticForm(dimension=10)"
        first = fit.eval(grid, threads=1)
        second = fit.eval(grid, threads=1)
    assert first[1] == pytest.approx(second[1], abs=0.0)
    assert first[3] == pytest.approx(second[3], abs=1e-12)
    with pytest.raises(RuntimeError):
        fit.eval(grid[:1])
