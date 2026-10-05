# Quadrature

Each threshold is a mapped Gauss–Legendre rule on the Gil-Pelaez integral. The upper limit in the mapped coordinate is chosen from the same truncation used by the Fortran code, then clamped into $(10^{-6},\, 0.999999)$.

The default node count depends on the law:

| Law | Nodes | Chebyshev nodes, when there are more than 24 thresholds |
| --- | --- | --- |
| Normal-inverse Gaussian, $\lambda = -1/2$ and $\chi, \psi > 0$ | 32 | 16 |
| General GH, fractional order with a convergent series | 32 | 6 |
| Half-integer order | 48 | 20 |
| $\psi = 0$ with no skewness contribution that moves $\psi$ | 64 | 16 |

`ES4_NNODE` replaces that count. Values outside 4–256 are clamped. `ES4_SLOW=1` skips the closed forms and the series and evaluates each threshold on its own.

A vector longer than 24 distinct thresholds is replaced by the Chebyshev grid, integrated there, and written back with Clenshaw's algorithm. Identical thresholds, up to $10^{-14}$ relative to the right endpoint, are evaluated once and copied.

## Threads

On the scalar path, macOS splits the threshold index into contiguous chunks and runs one Grand Central Dispatch worker per chunk, at most 64. OpenMP, when the extension was built with it, makes the same split. Windows evaluates that path on the calling thread.

`threads <= 0` selects the Apple performance-core count on that path, and the online CPU count elsewhere. The split is reached only when three things are true: the fast evaluation declined, `threads > 1`, and the remaining length is at least 4. NIG, a $\psi = 0$ law, and the general-GH series take the fast evaluation, so they do not split the vector. For a long half-integer problem the workers see the Chebyshev grid, and the original thresholds are interpolated on the calling thread.

## Accuracy

The test suite compares the portfolio and two-stage least squares examples from the Julia package with an independent 96-node quadrature of the same integral. On the portfolio the tail probability stays within about $10^{-8}$. Expected shortfall stays within about $10^{-6}$ where that probability is above 1%, and within about $5 \times 10^{-4}$ at the far end of the grid. The two-stage least squares cases stay within about $10^{-8}$ on the probability and about $5 \times 10^{-7}$ on the expected shortfall.

GitHub Actions runs those tests on Linux, macOS, and Windows, on Python 3.9 and 3.12.
