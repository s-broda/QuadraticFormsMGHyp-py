# Quadrature

Each threshold is a mapped Gauss–Legendre rule on the Gil-Pelaez integral. The upper limit in the mapped coordinate is chosen from the same truncation used by the Fortran code, then clamped into $(10^{-6},\, 0.999999)$.

The node count starts from a default and doubles until the survival and the expected shortfall each change by less than a relative $10^{-7}$, or until 4096 nodes. The value returned is the finer of the two orders that agreed. The next call on the same object starts at the coarser of those two and checks the refinement again.

| Law | Starting nodes |
| --- | --- |
| Normal-inverse Gaussian, $\lambda = -1/2$ and $\chi, \psi > 0$ | 32 |
| General GH | 32 |
| Half-integer order | 48 |
| $\psi = 0$ with no skewness contribution that moves $\psi$ | 64 |

`ES4_NNODE` replaces that count and turns the refinement off. Values outside 4–8192 are clamped. `ES4_SLOW=1` skips the closed forms and the series and evaluates each threshold on its own.

A vector longer than 24 distinct thresholds is replaced by a Chebyshev grid, integrated there, and written back with Clenshaw's algorithm. The degree starts at 20 for the normal-inverse Gaussian and at 48 otherwise, and grows until the last coefficient is at most $10^{-9}$, up to 96. The node count is certified on those abscissae. Identical thresholds, up to $10^{-14}$ relative to the right endpoint, are evaluated once and copied.

## Threads

On the scalar path, macOS splits the threshold index into contiguous chunks and runs one Grand Central Dispatch worker per chunk, at most 64. OpenMP, when the extension was built with it, makes the same split. Windows evaluates that path on the calling thread.

`threads <= 0` selects the logical CPU count. The split is reached only when three things are true: the fast evaluation declined, `threads > 1`, and the remaining length is at least 4. NIG, a $\psi = 0$ law, and the general-GH series take the fast evaluation, so they do not split the vector. For a long half-integer problem the workers see the Chebyshev grid, and the original thresholds are interpolated on the calling thread.

## Accuracy

The test suite compares the portfolio and two-stage least squares examples from the Julia package with an independent 96-node quadrature of the same integral. On the portfolio the tail probability stays within about $10^{-8}$. Expected shortfall stays within about $10^{-6}$ where that probability is above 1%, and within about $5 \times 10^{-4}$ at the far end of the grid. The two-stage least squares cases stay within about $10^{-8}$ on the probability and about $5 \times 10^{-7}$ on the expected shortfall.

GitHub Actions runs those tests on Linux, macOS, and Windows, on Python 3.9 and 3.12.

The partial-moment anchor is $\mathrm{E}[L] - k_k = k\,\mathrm{E}[W^2] + \mathrm{E}[W](c + \mathrm{tr}(C'AC))$, with $k = \gamma'A\gamma$. When $k$ is exactly zero that first product is omitted before the exponential. For $\psi = 0$ and $\lambda = -2$, $\mathrm{E}[W^2]$ is infinite, and forming the product in floating point yields a NaN shortfall even though $k W^2$ is the zero random variable. The survival probability does not use the anchor.
