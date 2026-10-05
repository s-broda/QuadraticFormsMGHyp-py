# es4mgh

Tail probability and expected shortfall of a quadratic form in a multivariate generalized hyperbolic vector.

The loss is `L = a0 + a'X + X'A X`, with `X = μ + W γ + √W C Z`, `Z ~ N(0, I)`, and `W ~ GIG(λ, χ, ψ)`. At a threshold `x` the routine returns `P(L > x)` and `E[L | L > x]`.

The numerical work is the C routine in this repository. The Python package calls that routine. The Fortran and Matlab code for the paper is in [s-broda/es4mgh](https://github.com/s-broda/es4mgh). This package is not on PyPI.

## Install

```bash
pip install "git+https://github.com/s-broda/es4mgh-c.git"
```

This builds the extension on the machine where it is installed, so a C compiler and Python headers are required. `numpy` is installed as a dependency. On macOS the build links Accelerate.

From a checkout, `pip install .` does the same thing.

## API

```python
import es4mgh

fit = es4mgh.QuadraticForm(a0, a, A, C, mu, gam, lam, chi, psi)
ccdf, es = fit.eval(x, threads=0)

fit = es4mgh.QuadraticForm.from_spectral(omega, d, e, c, k, kk, lam, chi, psi)
ccdf, es = fit.eval(x)
```

`gam` is the skewness vector γ. `a`, `mu`, and `gam` are length-`d` vectors. `A` and `C` are `d × d` row-major matrices, either shape `(d, d)` or a flat vector of length `d * d`. In `from_spectral`, `omega`, `d`, and `e` are the spectral weights and the two coefficient vectors. `threads <= 0` uses the performance-core count on Apple and the online CPU count elsewhere.

`close()` releases the C object. It is also released when the Python object is collected.

## Accuracy

Thresholds use a mapped Gauss–Legendre rule, 32 to 64 nodes depending on the law. More than 24 thresholds are evaluated on a short Chebyshev grid and interpolated. Against a 96-node reference, tail errors on the bundled problems are about `1e-9` or smaller. Expected-shortfall error stays under about `1e-4` on the general GH problem and is far smaller on the NIG, Student-t, and two-stage least squares problems.

No license is set.
