# QuadraticFormsMGHyp

[![tests](https://github.com/s-broda/QuadraticFormsMGHyp-py/actions/workflows/ci.yml/badge.svg)](https://github.com/s-broda/QuadraticFormsMGHyp-py/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-pages-blue)](https://s-broda.github.io/QuadraticFormsMGHyp-py/)

Tail probability and expected shortfall of a quadratic form in a multivariate generalized hyperbolic vector.

The loss is `L = a0 + a'X + X'A X`, with `X = μ + W γ + √W C Z`, `Z ~ N(0, I)`, and `W ~ GIG(λ, χ, ψ)`. One evaluation returns `P(L <= x)`, `P(L > x)`, the upper partial moment `E[L 1_{L > x}]`, and the expected shortfall `E[L | L > x]`.

```python
import numpy as np
from QuadraticFormsMGHyp import QuadraticForm

qf = QuadraticForm(
    0.0,
    np.array([1.0]),
    np.zeros((1, 1)),
    np.ones((1, 1)),
    np.zeros(1),
    np.zeros(1),
    -0.5,  # lambda
    1.0,   # chi
    1.0,   # psi
)
cdf, ccdf, pm, es = qf.eval(np.linspace(-1.0, 3.0, 5))
```

Construction diagonalizes the form. Further grids on the same object reuse that reduction, and each `eval` integrates the whole vector once.

The numerical work is the C routine in this repository. The Julia package is [QuadraticFormsMGHyp.jl](https://github.com/s-broda/QuadraticFormsMGHyp.jl). The Fortran and Matlab code for the paper is in [s-broda/es4mgh](https://github.com/s-broda/es4mgh). The package is on PyPI. The [documentation](https://s-broda.github.io/QuadraticFormsMGHyp-py/) has the model, the API, and the quadrature.

## Install

```bash
pip install QuadraticFormsMGHyp
```

Python 3.9 or newer is required. `numpy` is installed as a dependency. Wheels cover Linux x86_64 and arm64 (manylinux and musllinux), macOS x86_64 and arm64, and Windows amd64. Windows arm64 wheels start at Python 3.11.

A source install builds the extension on the machine, so a C compiler and Python headers are required. On macOS the build links Accelerate. On Windows the build uses clang-cl from LLVM and the Microsoft linker. Install [LLVM](https://github.com/llvm/llvm-project/releases) and the Microsoft C++ build tools.

```bash
pip install "git+https://github.com/s-broda/QuadraticFormsMGHyp-py.git"
```

From a checkout, `pip install .` does the same thing. `pip install ".[test]"` adds pytest. `pip install ".[docs]"` adds Sphinx.

## Wheels

A push to `main` builds wheels with cibuildwheel and tests each one. The set is Linux x86_64 and arm64 (manylinux and musllinux), macOS x86_64 and arm64, and Windows x86_64 and arm64, for the CPython versions cibuildwheel still builds. Python 3.8 is not included. Windows arm64 wheels start at Python 3.11, which is the first release for which NumPy publishes that platform. A GitHub release uploads those wheels and the source distribution to PyPI. The Windows wheels use the same clang-cl and Microsoft linker path as a source build.

## API

```python
import QuadraticFormsMGHyp as qf

fit = qf.QuadraticForm(a0, a, A, C, mu, gam, lam, chi, psi)
cdf, ccdf, pm, es = fit.eval(x, threads=0)

fit = qf.QuadraticForm.from_spectral(omega, d, e, c, k, kk, lam, chi, psi)
cdf, ccdf, pm, es = fit.eval(x)
```

`gam` is the skewness vector γ. `a`, `mu`, and `gam` are length-`d` vectors. `A` and `C` are `d × d` row-major matrices, either shape `(d, d)` or a flat vector of length `d * d`. In `from_spectral`, `omega`, `d`, and `e` are the spectral weights and the two coefficient vectors.

`threads <= 0` asks for one worker per Apple performance core, or per online CPU elsewhere. That pool runs only on the scalar path. NIG, ψ = 0, and the general-GH series evaluate the whole vector on the calling thread. More than 24 thresholds are computed on a short Chebyshev grid and interpolated.

`close()` releases the C object. A `with` block does the same, as does collection.

## Accuracy

Thresholds use a mapped Gauss–Legendre rule: 32 nodes for NIG and the general series, 48 for a half-integer order, and 64 for a ψ = 0 law. More than 24 thresholds are evaluated on a Chebyshev grid of 6, 16, or 20 nodes and interpolated.

## Tests

```bash
pip install ".[test]"
pytest
```

The checks rebuild the option-portfolio and two-stage least squares examples from [QuadraticFormsMGHyp.jl](https://github.com/s-broda/QuadraticFormsMGHyp.jl) and compare the package with a separate 96-node quadrature of the same integral. On the portfolio the tail probability stays within about `1e-8` of that rule. Expected shortfall stays within about `1e-6` where the tail probability is above 1%, and within about `5e-4` at the far end of the grid. GitHub Actions runs the tests on Linux, macOS, and Windows.

## Citation

If you use this package in your research, please cite [Broda and Zambrano (2021)](https://doi.org/10.1093/biomet/asaa067). `CITATION.bib` and `CITATION.cff` are in the repository.

Released under the MIT License.
