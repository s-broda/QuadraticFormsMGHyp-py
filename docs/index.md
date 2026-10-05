# QuadraticFormsMGHyp

Tail probabilities and expected shortfalls of a quadratic form in a multivariate generalized hyperbolic vector. The import name matches the Julia package [QuadraticFormsMGHyp.jl](https://github.com/s-broda/QuadraticFormsMGHyp.jl). The numerical work is a C routine. The Fortran and Matlab code for the paper is [s-broda/es4mgh](https://github.com/s-broda/es4mgh).

```{toctree}
:hidden:

api
numerical
```

The [API](api.md) and the [quadrature](numerical.md) are documented separately.

## The random variable

$$
L = a_0 + a^\top X + X^\top A X,
$$

where $X$ has the stochastic representation

$$
X = \mu + W \gamma + \sqrt{W}\, C Z.
$$

$Z$ is standard normal, $\mu$ and $\gamma$ are constant vectors, $C$ is a square matrix, and $W$ is generalized inverse Gaussian with density proportional to

$$
w^{\lambda - 1} \exp\left\{-\frac12\left(\chi w^{-1} + \psi w\right)\right\}.
$$

Special cases include the variance-gamma law ($\lambda > 0$), Student's $t$ ($\lambda = -\nu/2$, $\chi = \nu$, $\psi = 0$), the normal-inverse Gaussian ($\lambda = -1/2$), and the hyperbolic law ($\lambda = 1$).

At a threshold $x$ the package returns $\mathrm{P}(L > x)$ and $\mathrm{E}[L \mid L > x]$. The algorithm is the Gil-Pelaez inversion from Broda and Zambrano, [*Biometrika* **108** (2021)](https://doi.org/10.1093/biomet/asaa067). It generalizes Imhof (1961) and Broda (2012).

## Installation

The package is not on PyPI. A C compiler and Python headers are required, because the extension is compiled at install time. NumPy is installed as a dependency.

```bash
pip install "git+https://github.com/s-broda/QuadraticFormsMGHyp-py.git"
```

On macOS the build links Accelerate. On Windows it uses clang-cl from LLVM and the Microsoft linker. Install [LLVM](https://github.com/llvm/llvm-project/releases) and the Microsoft C++ build tools. From a checkout, `pip install .` does the same thing.

```bash
pip install ".[test]"   # pytest
pip install ".[docs]"   # this site
sphinx-build -b html docs docs/_build/html
```

## Usage

`qfmgh` takes the same positional arguments as the Julia function. A scalar threshold returns two floats. A vector returns two arrays and reuses one spectral reduction for every entry.

```python
import numpy as np
import QuadraticFormsMGHyp as qf

ccdf, es = qf.qfmgh(
    np.linspace(-1.0, 3.0, 5),
    0.0,
    np.array([1.0]),
    np.zeros((1, 1)),
    np.ones((1, 1)),
    np.zeros(1),
    np.zeros(1),
    -0.5,
    1.0,
    1.0,
)
```

Hold a {class}`QuadraticFormsMGHyp.QuadraticForm` when several grids share the coefficients. {meth}`QuadraticFormsMGHyp.QuadraticForm.from_spectral` skips the reduction and takes the eigenvalues and the two coefficient vectors directly.

```python
with qf.QuadraticForm(a0, a, A, C, mu, gam, lam, chi, psi) as fit:
    ccdf, es = fit.eval(x)
```

This package always evaluates the integral. The Julia keywords `do_spa` and `order` are not accepted.

## Citation

Please cite [Broda and Zambrano (2021)](https://doi.org/10.1093/biomet/asaa067). The repository contains `CITATION.bib` and `CITATION.cff`.

The package is released under the MIT License.
