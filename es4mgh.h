#ifndef ES4MGH_H
#define ES4MGH_H

/* Tail probability and expected shortfall of a quadratic form in a
   multivariate generalized hyperbolic vector.

   L = a0 + a' X + X' A X,
   X = mu + W gamma + sqrt(W) C Z,
   Z ~ N(0, I), W ~ GIG(lam, chi, psi).

   es4mgh_eval writes
     ccdf[i] = P(L > x[i])
     es[i]   = E[L | L > x[i]]
   Either pointer may be NULL. Matrices are row-major.
   nthreads <= 0 uses the performance-core count on Apple and the
   online CPU count elsewhere. Vectors longer than 24 thresholds are
   evaluated on a short Chebyshev grid and interpolated. */

typedef struct es4mgh es4mgh;

es4mgh *es4mgh_create(
    int d, double a0,
    const double *a, const double *A, const double *C,
    const double *mu, const double *gam,
    double lam, double chi, double psi);

/* Integral-only setup. omega, dvec, evec are the spectral data
   produced by the same reduction as es4mgh_create. kk = a0 + a'mu + mu'A mu
   is added back when the conditional expectation is formed. */
es4mgh *es4mgh_create_spectral(
    int ne, const double *omega, const double *dvec, const double *evec,
    double c, double k, double kk,
    double lam, double chi, double psi);

void es4mgh_eval(const es4mgh *E, int n, const double *x,
                 double *ccdf, double *es, int nthreads);

void es4mgh_free(es4mgh *E);

#endif
