# Changelog

## 0.1.2

When the coefficient of W squared is exactly zero, the partial-moment anchor omits that term before the exponential. For psi = 0 and lambda = -2 the second moment of W is infinite, and the product was a NaN shortfall. The survival probability was already finite.

## 0.1.1

The README and the documentation give `pip install QuadraticFormsMGHyp`. The 0.1.0 project page kept the earlier text that said the package was not on PyPI.

The default scalar-path worker count is the logical CPU count. On Apple that includes the efficiency cores.

## 0.1.0

First Python release. Requires Python 3.9 or newer. Construct a `QuadraticForm` and call `eval`. One call returns the cdf, the tail probability, the upper partial moment, and the expected shortfall. A GitHub release publishes wheels and a source distribution to PyPI.
