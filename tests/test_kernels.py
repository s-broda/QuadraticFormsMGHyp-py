"""The batched kernels, the node cache, and the worker split.

ES4_SLOW=1 forces the scalar integrand in a fresh process. The fast path
in this process is the one the extension actually ships: half-integer
batch, closed-form NIG and psi = 0, the fractional series, and Clenshaw
on grids longer than 24. threads=0 is the worker pool (GCD, OpenMP, or
the serial fallback). A dense spectrum of order 24 or more is the BLAS
eigensolver when the build found one, and Jacobi otherwise.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import QuadraticFormsMGHyp

from examples import portfolio
from quadrature import gil_pelaez


def _max_rel(got, ref):
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    diff = np.abs(got - ref)
    den = np.maximum(np.abs(got), np.abs(ref))
    rel = np.zeros(diff.shape, dtype=np.float64)
    for i in range(diff.size):
        if np.isnan(got[i]) and np.isnan(ref[i]):
            continue
        if den[i] == 0.0:
            rel[i] = 0.0 if diff[i] == 0.0 else np.inf
        else:
            rel[i] = diff[i] / den[i]
    return float(np.max(rel)) if rel.size else 0.0


def _args(lam, chi, psi, skew=False, d=2):
    a = np.zeros(d)
    mu = np.zeros(d)
    gam = np.zeros(d)
    eye = np.eye(d)
    diag = np.diag(np.linspace(0.4, 1.1, d))
    a0 = 0.0
    if skew:
        a0 = 0.1
        a = np.zeros(d)
        gam = np.zeros(d)
        a[:2] = (0.1, -0.2)
        gam[:2] = (0.2, -0.1)
        mu = np.full(d, 0.02)
    return a0, a, diag, eye, mu, gam, float(lam), float(chi), float(psi)


def _form(args):
    return QuadraticFormsMGHyp.QuadraticForm(*args)


def _eval(args, x, threads):
    _, ccdf, _, es = _form(args).eval(np.asarray(x, dtype=np.float64), threads=threads)
    return np.asarray(ccdf, dtype=np.float64), np.asarray(es, dtype=np.float64)


def _eval_slow(args, x):
    """Scalar integrand, so a batched kernel has something independent to match."""
    a0, a, A, C, mu, gam, lam, chi, psi = args
    x = np.asarray(x, dtype=np.float64)
    script = (
        "import sys\n"
        "import numpy as np\n"
        "import QuadraticFormsMGHyp\n"
        "d = np.load(sys.argv[1])\n"
        "fit = QuadraticFormsMGHyp.QuadraticForm(\n"
        "    float(d['a0'][0]), d['a'], d['A'], d['C'], d['mu'], d['gam'],\n"
        "    float(d['lam'][0]), float(d['chi'][0]), float(d['psi'][0]))\n"
        "_, ccdf, _, es = fit.eval(d['x'], threads=1)\n"
        "np.savez(sys.argv[2], ccdf=np.asarray(ccdf), es=np.asarray(es))\n"
    )
    env = os.environ.copy()
    env["ES4_SLOW"] = "1"
    env.pop("ES4_NNODE", None)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        inn = os.path.join(td, "in.npz")
        out = os.path.join(td, "out.npz")
        np.savez(
            inn, a=a, A=A, C=C, mu=mu, gam=gam, x=x,
            a0=np.array([a0]), lam=np.array([lam]),
            chi=np.array([chi]), psi=np.array([psi]),
        )
        ran = subprocess.run(
            [sys.executable, "-c", script, inn, out],
            env=env, capture_output=True, text=True,
        )
        if ran.returncode != 0:
            raise AssertionError(ran.stderr or ran.stdout)
        # Close the archive before the temporary directory is removed.
        # Windows keeps the file locked until NpzFile is closed.
        with np.load(out) as got:
            ccdf = got["ccdf"].copy()
            es = got["es"].copy()
    return ccdf, es


def _assert_pair(got_ccdf, got_es, ref_ccdf, ref_es, rel, label):
    cgap = _max_rel(got_ccdf, ref_ccdf)
    egap = _max_rel(got_es, ref_es)
    assert cgap < rel, "%s ccdf relative = %g" % (label, cgap)
    assert egap < rel, "%s es relative = %g" % (label, egap)


# Grids of length 16 stay on the direct integrand. 41 and 60 go through
# Clenshaw, including the scalar remainder of the 4-wide recurrence.
CASES = [
    ("half", -1.5, 2.0, 2.0, False),
    ("half-skew", -1.5, 1.5, 1.5, True),
    ("half-degree", -4.5, 2.0, 2.0, False),
    ("variance-gamma", -1.5, 0.0, 2.0, True),
    ("generic", -1.3, 2.0, 2.0, False),
    ("psi0", -5.0, 10.0, 0.0, False),
    ("integer-order", -2.0, 1.0, 1.0, False),
]


@pytest.mark.parametrize("name,lam,chi,psi,skew", CASES)
def test_fast_path_matches_the_scalar_integrand(name, lam, chi, psi, skew):
    if os.environ.get("ES4_SLOW") == "1":
        pytest.skip("ES4_SLOW is set, so this process has no fast path")
    args = _args(lam, chi, psi, skew=skew)
    x = np.linspace(-1.0, 8.0, 16)
    fast_c, fast_e = _eval(args, x, threads=1)
    slow_c, slow_e = _eval_slow(args, x)
    _assert_pair(fast_c, fast_e, slow_c, slow_e, 1e-9, name)
    assert np.all(np.isfinite(fast_c))
    assert np.all(np.isfinite(fast_e))


@pytest.mark.parametrize("name,lam,chi,psi,skew", CASES)
def test_worker_pool_matches_one_thread(name, lam, chi, psi, skew):
    if os.environ.get("ES4_SLOW") == "1":
        pytest.skip("ES4_SLOW is set, so the worker pool is the scalar integrand")
    args = _args(lam, chi, psi, skew=skew)
    x = np.linspace(-1.0, 8.0, 16)
    fit = _form(args)
    _, one_c, _, one_e = fit.eval(x, threads=1)
    _, pool_c, _, pool_e = fit.eval(x, threads=0)
    _assert_pair(pool_c, pool_e, one_c, one_e, 1e-12, name)


def test_nig_batch_matches_the_scalar_integrand():
    if os.environ.get("ES4_SLOW") == "1":
        pytest.skip("ES4_SLOW is set, so this process has no fast path")
    problem = portfolio()
    args = (
        problem["a0"], problem["a"], problem["A"], problem["C"],
        problem["mu"], problem["gam"], problem["lam"], problem["chi"], problem["psi"],
    )
    x = problem["x"][:16]
    fast_c, fast_e = _eval(args, x, threads=1)
    slow_c, slow_e = _eval_slow(args, x)
    _assert_pair(fast_c, fast_e, slow_c, slow_e, 1e-9, "nig")


def test_psi0_and_dense_spectrum_match_the_independent_rule():
    # psi = 0 is the power kernel. The dense order-24 matrix is Jacobi
    # or dsyev, depending on the platform, against a separate eigh.
    psi_args = _args(-5.0, 10.0, 0.0)
    x = np.linspace(1.0, 12.0, 8)
    ccdf, es = _eval(psi_args, x, threads=1)
    problem = {
        "a0": psi_args[0], "a": psi_args[1], "A": psi_args[2], "C": psi_args[3],
        "mu": psi_args[4], "gam": psi_args[5], "lam": psi_args[6],
        "chi": psi_args[7], "psi": psi_args[8], "x": x,
    }
    ref_c, ref_e = gil_pelaez(problem)
    assert float(np.max(np.abs(ccdf - ref_c))) < 1e-7
    assert float(np.max(np.abs(es - ref_e))) < 5e-6

    d = 24
    A = np.diag(np.linspace(0.2, 1.5, d))
    direction = np.linspace(-0.3, 0.3, d)
    A = A + np.outer(direction, direction)
    A = 0.5 * (A + A.T)
    assert np.max(np.abs(A - np.diag(np.diag(A)))) > 0.0
    z = np.zeros(d)
    dense = (0.0, z, A, np.eye(d), z, z, -0.5, 1.0, 1.0)
    xd = np.linspace(2.0, 30.0, 6)
    ccdf, es = _eval(dense, xd, threads=1)
    problem = {
        "a0": 0.0, "a": z, "A": A, "C": np.eye(d), "mu": z, "gam": z,
        "lam": -0.5, "chi": 1.0, "psi": 1.0, "x": xd,
    }
    ref_c, ref_e = gil_pelaez(problem)
    assert float(np.max(np.abs(ccdf - ref_c))) < 1e-7
    body = ref_c > 0.01
    if np.any(body):
        assert float(np.max(np.abs(es[body] - ref_e[body]))) < 5e-6


def test_long_grids_match_direct_evaluations():
    # Lengths past 24 use the Chebyshev grid. 41 leaves a remainder in
    # the 4-wide Clenshaw loop. Each direct call is one threshold.
    specs = [
        (_args(-0.5, 1.0, 1.0), np.linspace(1.0, 8.0, 41)),
        (_args(-1.5, 2.0, 2.0), np.linspace(-1.0, 8.0, 60)),
        (_args(-1.3, 2.0, 2.0), np.linspace(-1.0, 8.0, 41)),
    ]
    for args, x in specs:
        grid_c, grid_e = _eval(args, x, threads=1)
        point_c = np.empty(x.shape[0])
        point_e = np.empty(x.shape[0])
        for i, xi in enumerate(x):
            cc, ee = _eval(args, [xi], threads=1)
            point_c[i] = cc[0]
            point_e[i] = ee[0]
        _assert_pair(grid_c, grid_e, point_c, point_e, 1e-7, "grid")


def test_reused_object_swaps_back_to_the_same_values():
    args = _args(-1.5, 2.0, 2.0, skew=True)
    fit = _form(args)
    short = np.linspace(0.5, 4.0, 5)
    _, first_c, _, first_e = fit.eval(short, threads=1)
    fit.eval(np.linspace(-1.0, 8.0, 48), threads=1)
    _, again_c, _, again_e = fit.eval(short, threads=1)
    _assert_pair(again_c, again_e, first_c, first_e, 1e-12, "reuse")


def test_linux_build_links_openmp_and_openblas():
    # The numerical tests pass with Jacobi alone. This one checks that a
    # Linux machine which has the libraries, including CI, actually linked them.
    if not sys.platform.startswith("linux"):
        pytest.skip("OpenMP and OpenBLAS are the Linux build")
    import QuadraticFormsMGHyp._native as native

    ldd = subprocess.check_output(["ldd", native.__file__], text=True)
    on_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    blas = subprocess.call(
        ["pkg-config", "--exists", "openblas"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if on_ci or blas == 0:
        assert "openblas" in ldd.lower(), ldd
    if on_ci or "libgomp" in ldd or "libomp" in ldd:
        assert "libgomp" in ldd or "libomp" in ldd, ldd


def test_gaussian_panels_match_across_thread_counts():
    x = np.linspace(0.25, 12.0, 20)
    z = np.zeros(1)
    fit = QuadraticFormsMGHyp.QuadraticForm(
        0.0, z, np.ones((1, 1)), np.ones((1, 1)), z, z, 0.0, np.inf, np.inf,
    )
    _, one_c, _, one_e = fit.eval(x, threads=1)
    _, pool_c, _, pool_e = fit.eval(x, threads=0)
    _assert_pair(pool_c, pool_e, one_c, one_e, 1e-12, "gauss")
    # q = 0 is the log-substituted panel, not the oscillating one.
    _, c0, _, e0 = fit.eval(0.0, threads=1)
    assert c0 == pytest.approx(1.0, abs=1e-8)
    assert e0 == pytest.approx(1.0, abs=1e-8)
