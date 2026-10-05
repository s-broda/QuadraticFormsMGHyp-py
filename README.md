# es4mgh

Tail probability and expected shortfall of a quadratic form in a multivariate generalized hyperbolic vector.

The loss is `L = a0 + a'X + X'A X`, with `X = μ + W γ + √W C Z`, `Z ~ N(0, I)`, and `W ~ GIG(λ, χ, ψ)`. At a threshold `x` the routine returns `P(L > x)` and `E[L | L > x]`.

The numerical work is the C routine in this repository. The Python package calls that routine. The Fortran and Matlab code for the paper is in [s-broda/es4mgh](https://github.com/s-broda/es4mgh). This package is not on PyPI.

## Install

```bash
pip install "git+https://github.com/s-broda/es4mgh-c.git"
```

This builds the extension on the machine where it is installed, so a C compiler and Python headers are required. `numpy` is installed as a dependency. On macOS the build links Accelerate. On Windows the build uses clang-cl from LLVM and the Microsoft linker, which is the same toolchain a wheel will use. Install [LLVM](https://github.com/llvm/llvm-project/releases) and the Microsoft C++ build tools.

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

Thresholds use a mapped Gauss–Legendre rule, 32 to 64 nodes depending on the law. More than 24 thresholds are evaluated on a short Chebyshev grid and interpolated.

## Tests

`pytest` is an optional extra.

```bash
pip install ".[test]"
pytest
```

The checks rebuild the option-portfolio and two-stage least squares examples from [QuadraticFormsMGHyp](https://github.com/s-broda/QuadraticFormsMGHyp.jl) and compare the package with a separate 96-node quadrature of the same integral. On the portfolio the tail probability stays within about `1e-8` of that rule. Expected shortfall stays within about `1e-6` where the tail probability is above 1%, and within about `5e-4` at the far end of the grid. GitHub Actions runs the tests on Linux, macOS, and Windows.

Released under the MIT License.
