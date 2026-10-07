# Changelog

## 0.4.0

The quadrature is unchanged. On Apple the closed-form kernels evaluate the complex logarithm, exponential, and square root through Accelerate. A half-integer order is integrated in one batch on each worker slice, so the Apple pool and OpenMP both use that kernel. Gauss–Legendre nodes are cached for the process, and a reused object keeps the previous order. Grids longer than 24 thresholds use the same Clenshaw recurrence, four points at a time. On Linux the build links OpenMP, and OpenBLAS for the symmetric eigensolver and the matrix products when that library is installed. Windows stays on clang-cl and the Jacobi eigensolver.

## 0.3.0

The mapped rule doubles its node count until successive refinements agree to a relative tolerance of 1e-7, up to 4096 nodes. A long threshold list uses a Chebyshev series that grows the same way as the Julia package. `ES4_NNODE`, when set, is a fixed order in 4–8192. A half-integer order whose χ or ψ argument is exactly zero uses the gamma limit, so a skewed variance-gamma form at the origin stays finite. `chi = psi = inf` is the Gaussian limit, with the mixer fixed at 1. A low-rank Gaussian spectrum is integrated in panels because the characteristic function decays only as a power of the frequency.

## 0.2.0

No change to the numerical results. The README carries a Zenodo badge. Publishing this GitHub release archives the package on Zenodo.

## 0.1.2

When the coefficient of W squared is exactly zero, the partial-moment anchor omits that term before the exponential. For psi = 0 and lambda = -2 the second moment of W is infinite, and the product was a NaN shortfall. The survival probability was already finite.

## 0.1.1

The README and the documentation give `pip install QuadraticFormsMGHyp`. The 0.1.0 project page kept the earlier text that said the package was not on PyPI.

The default scalar-path worker count is the logical CPU count. On Apple that includes the efficiency cores.

## 0.1.0

First Python release. Requires Python 3.9 or newer. Construct a `QuadraticForm` and call `eval`. One call returns the cdf, the tail probability, the upper partial moment, and the expected shortfall. A GitHub release publishes wheels and a source distribution to PyPI.
