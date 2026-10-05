"""Check the Python binding against the reference problems in this repo."""

import pathlib
import sys

import numpy as np

import es4mgh

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _take(tokens, i, n):
    chunk = tokens[i : i + n]
    if len(chunk) != n:
        raise RuntimeError("truncated problem file")
    return np.asarray(chunk, dtype=np.float64), i + n


def load_spectral(path):
    tokens = path.read_text().split()
    i = 1
    nprob = int(tokens[0])
    out = []
    for _ in range(nprob):
        name = tokens[i]
        n = int(tokens[i + 1])
        ne = int(tokens[i + 2])
        i += 3
        lam, chi, psi, c, k, kk = (float(tokens[i + j]) for j in range(6))
        i += 6
        omega, i = _take(tokens, i, ne)
        d, i = _take(tokens, i, ne)
        e, i = _take(tokens, i, ne)
        x, i = _take(tokens, i, n)
        cc, i = _take(tokens, i, n)
        es, i = _take(tokens, i, n)
        out.append(
            dict(
                name=name,
                lam=lam,
                chi=chi,
                psi=psi,
                c=c,
                k=k,
                kk=kk,
                omega=omega,
                d=d,
                e=e,
                x=x,
                cc=cc,
                es=es,
            )
        )
    return out


def load_matrix(path):
    tokens = path.read_text().split()
    i = 1
    nmat = int(tokens[0])
    out = []
    for _ in range(nmat):
        name = tokens[i]
        d = int(tokens[i + 1])
        n = int(tokens[i + 2])
        i += 3
        a0, lam, chi, psi = (float(tokens[i + j]) for j in range(4))
        i += 4
        a, i = _take(tokens, i, d)
        mu, i = _take(tokens, i, d)
        gam, i = _take(tokens, i, d)
        A, i = _take(tokens, i, d * d)
        C, i = _take(tokens, i, d * d)
        x, i = _take(tokens, i, n)
        cc, i = _take(tokens, i, n)
        es, i = _take(tokens, i, n)
        out.append(
            dict(
                name=name,
                d=d,
                a0=a0,
                lam=lam,
                chi=chi,
                psi=psi,
                a=a,
                mu=mu,
                gam=gam,
                A=A,
                C=C,
                x=x,
                cc=cc,
                es=es,
            )
        )
    return out


def max_abs(a, b):
    return float(np.max(np.abs(a - b)))


def check_pair(name, cc, es, cc_ref, es_ref, cc_lim, es_lim):
    dc = max_abs(cc, cc_ref)
    de = max_abs(es, es_ref)
    print("  %-16s |dccdf|=%.3e |des|=%.3e" % (name, dc, de))
    if dc > cc_lim or de > es_lim:
        raise AssertionError("%s |dccdf|=%g |des|=%g" % (name, dc, de))
    return dc, de


def test_spectral():
    problems = load_spectral(ROOT / "problems.txt")
    floors = {
        "portfolio_nig": (1e-8, 1e-6),
        "general_gh": (2e-8, 1e-4),
    }
    for p in problems:
        fit = es4mgh.QuadraticForm.from_spectral(
            p["omega"], p["d"], p["e"], p["c"], p["k"], p["kk"], p["lam"], p["chi"], p["psi"]
        )
        cc, es = fit.eval(p["x"], threads=0)
        assert cc.shape == p["x"].shape and cc.dtype == np.float64
        assert es.shape == p["x"].shape and es.dtype == np.float64
        cc_lim, es_lim = floors.get(p["name"], (5e-6, 5e-4))
        check_pair(p["name"], cc, es, p["cc"], p["es"], cc_lim, es_lim)
        fit.close()
        try:
            fit.eval(p["x"][:1])
        except RuntimeError:
            pass
        else:
            raise AssertionError("eval after close should fail")


def test_matrix():
    problems = load_matrix(ROOT / "matrices.txt")
    floors = {
        "portfolio_nig": (1e-8, 1e-6),
        "general_gh": (2e-8, 1e-4),
    }
    for p in problems:
        A = p["A"].reshape(p["d"], p["d"])
        C = p["C"].reshape(p["d"], p["d"])
        fit = es4mgh.QuadraticForm(
            p["a0"], p["a"], A, C, p["mu"], p["gam"], p["lam"], p["chi"], p["psi"]
        )
        assert fit.dimension == p["d"]
        cc, es = fit.eval(p["x"], threads=1)
        cc_lim, es_lim = floors.get(p["name"], (5e-5, 5e-3))
        check_pair("matrix " + p["name"], cc, es, p["cc"], p["es"], cc_lim, es_lim)
        flat = es4mgh.QuadraticForm(
            p["a0"], p["a"], p["A"], p["C"], p["mu"], p["gam"], p["lam"], p["chi"], p["psi"]
        )
        cc2, es2 = flat.eval(p["x"], threads=1)
        if max_abs(cc2, cc) > 0.0 or max_abs(es2, es) > 0.0:
            raise AssertionError("flat and shaped matrices disagree for %s" % p["name"])


def test_threads_and_shapes():
    problems = load_spectral(ROOT / "problems.txt")
    general = next(p for p in problems if p["name"] == "general_gh")
    fit = es4mgh.QuadraticForm.from_spectral(
        general["omega"],
        general["d"],
        general["e"],
        general["c"],
        general["k"],
        general["kk"],
        general["lam"],
        general["chi"],
        general["psi"],
    )
    a, b = fit.eval(general["x"], threads=1)
    c, d = fit.eval(general["x"], threads=0)
    if max_abs(a, c) > 1e-12 or max_abs(b, d) > 1e-12:
        raise AssertionError("serial and parallel eval disagree")
    short = general["x"][:8]
    cs, es = fit.eval(short, threads=1)
    if max_abs(cs, general["cc"][:8]) > 2e-8 or max_abs(es, general["es"][:8]) > 1e-4:
        raise AssertionError("short-vector eval left the reference floor")
    empty_cc, empty_es = fit.eval(np.zeros(0))
    assert empty_cc.shape == (0,) and empty_es.shape == (0,)
    try:
        fit.eval(general["x"].reshape(-1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("a column of thresholds should be rejected")
    try:
        es4mgh.QuadraticForm.from_spectral(
            general["omega"],
            general["d"][:-1],
            general["e"],
            general["c"],
            general["k"],
            general["kk"],
            general["lam"],
            general["chi"],
            general["psi"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched spectral vectors should be rejected")


def main():
    print("es4mgh", es4mgh.__version__)
    test_spectral()
    test_matrix()
    test_threads_and_shapes()
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
