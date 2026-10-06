#include "es4mgh.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "es4_complex.h"
#else
#include <complex.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#elif defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const double PI = 3.14159265358979323846264338327950288;
static const double EU = 0.57721566490153286060651209008240243;
static const double LOG2 = 0.69314718055994530941723212145817657;
static const double INV_PI = 0.31830988618379067153776752674502872;

enum { KIND_GENERAL = 0, KIND_NIG = 1, KIND_HALF = 2, KIND_PSI0 = 3 };

enum { NNODE_CAP = 4096 };

static int kind_nnode(int kind) {
    /* Starting order. Successive refinements double it until the relative test. */
    if (kind == KIND_NIG || kind == KIND_GENERAL) return 32;
    if (kind == KIND_HALF) return 48;
    return 64;
}

/* A set ES4_NNODE is a fixed order, used as-is for benchmarks. */
static int env_nnode(void) {
    const char *e = getenv("ES4_NNODE");
    if (!e || !e[0]) return 0;
    int n = atoi(e);
    if (n < 4) n = 4;
    if (n > 8192) n = 8192;
    return n;
}

struct es4mgh {
    int kind;
    int nnode;
    int ne;          /* nonzero eigenvalues kept in the sums */
    int need_a1;     /* gamma / e contribution is present */
    double lam, chi, psi, kk, LK2;
    double *u;
    double *wo;      /* Gauss weight * du/dv / u */
    double _Complex *chi_base;
    double _Complex *psi_node;
    double _Complex *lrho;
    double _Complex *a2p;
    double _Complex *a1p;
    double _Complex *lrp;
    double _Complex *psi_inv;
    double _Complex *log_psi;
    double M20;
    double D2z, E2z, DEz, ccoef, kcoef, ub;
    double *spec; /* omega, d2, e2 packed with stride spec_n */
    double *de;
    int spec_n;
    int nlo;          /* coarse order that agreed with its double; 0 if unknown */
    int fixed_nnode;  /* ES4_NNODE, or 0 when the order adapts */
    int g_fast;
    double g_mu, g_sinmu, g_sin1, g_lgp, g_lgm, g_lgq, g_lgr;
    double g_elgp, g_elgm, g_elgq, g_elgr;
    double g_ip[29], g_im[29], g_iq[29], g_ir[29];
    double psi_c1, psi_c2; /* exp(c1-c0), exp(c2-c0) for the psi=0 moment */
};

/* ---------------- modified Bessel K, real order, complex argument ---------------- */

static double complex bessel_I(double nu, double complex z) {
    double complex y = 0.25 * z * z;
    double complex term = 1.0;
    double complex sum = 1.0;
    for (int k = 1; k < 250; ++k) {
        term *= y / (k * (nu + k));
        sum += term;
        if (cabs(term) <= 1e-17 * cabs(sum)) break;
    }
    return cexp(nu * clog(0.5 * z) - lgamma(nu + 1.0)) * sum;
}

static void bessel_K0_K1(double complex z, double complex *K0, double complex *K1) {
    double complex z2 = 0.5 * z;
    double complex y = z2 * z2;
    double complex t = 1.0;
    double complex I0 = 1.0;
    double complex S0 = -EU;
    double H = 0.0;
    for (int k = 1; k < 250; ++k) {
        t *= y / (double)(k * k);
        I0 += t;
        H += 1.0 / (double)k;
        S0 += t * (H - EU);
        if (cabs(t) <= 1e-17 * cabs(I0)) break;
    }
    double complex lz = clog(z2);
    *K0 = -I0 * lz + S0;

    t = z2;
    double complex I1 = t;
    double complex half = t * (1.0 - 2.0 * EU);
    H = 0.0;
    for (int k = 1; k < 250; ++k) {
        t *= y / (double)(k * (k + 1));
        I1 += t;
        H += 1.0 / (double)k;
        double psisum = -2.0 * EU + 2.0 * H + 1.0 / (double)(k + 1);
        half += t * psisum;
        if (cabs(t) <= 1e-17 * cabs(I1)) break;
    }
    *K1 = I1 * lz + 1.0 / z - 0.5 * half;
}

/* exp(z)*K_nu(z) by the divergent asymptotic series. Returns 0 if the
   smallest term is too large to trust. */
static int asymp_scaled(double nu, double complex z, double complex *scaled) {
    if (cabs(z) < 2.0) return 0;
    double complex inv = 1.0 / z;
    double complex term = 1.0;
    double complex sum = 1.0;
    double nu2 = 4.0 * nu * nu;
    double best = 1.0;
    int converged = 0;
    for (int k = 1; k <= 80; ++k) {
        double ak = nu2 - (double)((2 * k - 1) * (2 * k - 1));
        term *= ak * inv / (8.0 * (double)k);
        double at = cabs(term);
        if (k > 2 && at > best) break;
        if (at < best) best = at;
        sum += term;
        if (at < 1e-16 * cabs(sum)) { converged = 1; break; }
    }
    if (!converged && best > 1e-9) return 0;
    *scaled = csqrt(PI / (2.0 * z)) * sum;
    return 1;
}

/* log(exp(z) K_nu(z)). Any 2*pi*i branch is fine: only exp(log) is used. */
static double complex log_scaled_K(double nu, double complex z) {
    nu = fabs(nu);
    double nh = round(nu - 0.5);
    if (nh >= 0.0 && nh < 80.0 && fabs(nu - (nh + 0.5)) < 1e-10) {
        int n = (int)nh;
        double coef = 1.0;
        double complex sum = 1.0;
        double complex zk = 1.0;
        double complex twoz = 2.0 * z;
        for (int k = 1; k <= n; ++k) {
            coef *= (double)(n + k) * (double)(n - k + 1) / (double)k;
            zk *= twoz;
            sum += coef / zk;
        }
        return 0.5 * clog(PI / (2.0 * z)) + clog(sum);
    }

    double complex scaled;
    if (cabs(z) >= 6.0 && asymp_scaled(nu, z, &scaled)) return clog(scaled);

    int n = (int)floor(nu + 1e-14);
    double mu = nu - (double)n;
    if (mu < 1e-12) {
        double complex a, b;
        if (!(cabs(z) >= 6.0 && asymp_scaled(0.0, z, &a) && asymp_scaled(1.0, z, &b))) {
            double complex K0, K1;
            bessel_K0_K1(z, &K0, &K1);
            a = cexp(z) * K0;
            b = cexp(z) * K1;
        }
        if (n == 0) return clog(a);
        for (int i = 1; i < n; ++i) {
            double complex nxt = (2.0 * (double)i / z) * b + a;
            a = b;
            b = nxt;
        }
        return clog(b);
    }

    if (cabs(z) >= 4.0 && asymp_scaled(nu, z, &scaled)) return clog(scaled);

    double s = sin(PI * mu);
    double complex Kmu = (PI / (2.0 * s)) * (bessel_I(-mu, z) - bessel_I(mu, z));
    double mu1 = 1.0 - mu;
    double s2 = sin(PI * mu1);
    double complex K1mmu = (PI / (2.0 * s2)) * (bessel_I(-mu1, z) - bessel_I(mu1, z));
    double complex Kmu1 = (2.0 * mu / z) * Kmu + K1mmu;
    if (n == 0) return clog(cexp(z) * Kmu);
    double complex a = cexp(z) * Kmu;
    double complex b = cexp(z) * Kmu1;
    for (int i = 1; i < n; ++i) {
        double ord = mu + (double)i;
        double complex nxt = (2.0 * ord / z) * b + a;
        a = b;
        b = nxt;
    }
    return clog(b);
}

/* log k_lambda(chi, psi), the log of the GIG normaliser, analytically continued. */
static double complex lklam(double lam, double complex chi, double complex psi) {
    if (creal(chi) == 0.0 && cimag(chi) == 0.0) {
        if (creal(psi) < 0.0) return INFINITY;
        return -lam * clog(psi * 0.5) + lgamma(lam);
    }
    if (creal(psi) == 0.0 && cimag(psi) == 0.0) {
        if (creal(chi) < 0.0) return INFINITY;
        return lam * clog(chi * 0.5) + lgamma(-lam);
    }
    double complex prod = chi * psi;
    if (creal(prod) < 0.0 && fabs(cimag(prod)) <= 1e-14 * (1.0 + fabs(creal(prod))))
        return INFINITY;
    double complex z = csqrt(prod);
    double complex logK = log_scaled_K(lam, z) - z;
    return LOG2 + 0.5 * lam * clog(chi / psi) + logK;
}

static double lklam_real(double lam, double chi, double psi) {
    if (chi == 0.0) return -lam * log(psi * 0.5) + lgamma(lam);
    if (psi == 0.0) return lam * log(chi * 0.5) + lgamma(-lam);
    double z = sqrt(chi * psi);
    double complex logK = log_scaled_K(lam, z) - z;
    return LOG2 + 0.5 * lam * log(chi / psi) + creal(logK);
}

static inline double imag_exp(double complex phase) {
    double re = creal(phase);
    if (re < -700.0 || re > 700.0) return 0.0;
    return exp(re) * sin(cimag(phase));
}

static inline double imag_exp_mul(double complex phase, double complex c) {
    double re = creal(phase);
    if (re < -700.0 || re > 700.0) return 0.0;
    double e = exp(re);
    double s = sin(cimag(phase));
    double co = cos(cimag(phase));
    return e * (s * creal(c) + co * cimag(c));
}

/* ---------------- quadrature nodes ---------------- */

static void legendre(int n, double x, double *Pn, double *dPn) {
    double p0 = 1.0, p1 = x;
    if (n == 0) { *Pn = 1.0; *dPn = 0.0; return; }
    if (n == 1) { *Pn = x; *dPn = 1.0; return; }
    for (int k = 2; k <= n; ++k) {
        double p = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / (double)k;
        p0 = p1;
        p1 = p;
    }
    *Pn = p1;
    *dPn = n * (x * p1 - p0) / (x * x - 1.0);
}

static void gauss_legendre(int n, double *x, double *w) {
    for (int i = 0; i < n; ++i) {
        double theta = PI * (i + 0.75) / (n + 0.5);
        double xi = cos(theta);
        for (int it = 0; it < 20; ++it) {
            double Pn, dPn;
            legendre(n, xi, &Pn, &dPn);
            double dx = Pn / dPn;
            xi -= dx;
            if (fabs(dx) < 1e-15) break;
        }
        double Pn, dPn;
        legendre(n, xi, &Pn, &dPn);
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dPn * dPn);
    }
}

static double integration_ub(const double *omega, int ne) {
    double *o = (double *)malloc((size_t)ne * sizeof(double));
    if (!o) return 0.5;
    int m = 0;
    for (int i = 0; i < ne; ++i) {
        double a = fabs(omega[i]);
        if (a > 0.0) o[m++] = a;
    }
    for (int i = 1; i < m; ++i) {
        double v = o[i];
        int j = i;
        while (j > 0 && o[j - 1] < v) { o[j] = o[j - 1]; --j; }
        o[j] = v;
    }
    const double epsabs = 1e-10;
    double ub = 1.0;
    double logsum = 0.0;
    for (int i = 1; i <= m; ++i) {
        logsum += log(2.0 * o[i - 1]);
        double ubnew = (-1.0 / (double)i) * (2.0 * log(PI) + 2.0 * log((double)i)
            + 2.0 * log(epsabs) - log(4.0) + logsum);
        ubnew = exp(ubnew);
        ubnew = sqrt(ubnew) / (1.0 + sqrt(ubnew));
        if (ubnew < ub) ub = ubnew;
    }
    free(o);
    if (!(ub > 1e-6 && ub < 1.0)) ub = 0.99;
    if (ub > 0.999999) ub = 0.999999;
    return ub;
}

/* ---------------- symmetric eigen (cyclic Jacobi) ---------------- */

static void jacobi(int n, double *A, double *eval, double *V) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
    double scale = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            scale += fabs(A[i * n + j]);
    double tol = 1e-22 * (scale + 1.0);
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += A[p * n + q] * A[p * n + q];
        if (off <= tol) break;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = A[p * n + q];
                if (fabs(apq) <= tol) continue;
                double app = A[p * n + p];
                double aqq = A[q * n + q];
                double tau = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(tau) > 1e16) t = 0.5 / tau;
                else t = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau * tau)), tau);
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;
                A[p * n + p] = app - t * apq;
                A[q * n + q] = aqq + t * apq;
                A[p * n + q] = A[q * n + p] = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    double aik = A[k < p ? k * n + p : p * n + k];
                    double aqk = A[k < q ? k * n + q : q * n + k];
                    /* Read the stored triangle. Both triangles are maintained. */
                    aik = A[p * n + k];
                    aqk = A[q * n + k];
                    double np = c * aik - s * aqk;
                    double nq = s * aik + c * aqk;
                    A[p * n + k] = A[k * n + p] = np;
                    A[q * n + k] = A[k * n + q] = nq;
                }
                for (int k = 0; k < n; ++k) {
                    double vip = V[k * n + p];
                    double viq = V[k * n + q];
                    V[k * n + p] = c * vip - s * viq;
                    V[k * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) eval[i] = A[i * n + i];
}

/* Columns of V are eigenvectors, row-major. A is row-major symmetric. */
static void eigen_symmetric(int n, double *A, double *eval, double *V) {
#if defined(__APPLE__)
    /* Jacobi is faster for the small matrices. BLAS pays off from n=24. */
    if (n >= 24) {
    double *Ac = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (Ac) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                Ac[i + (size_t)j * (size_t)n] = A[i * n + j];
        char jobz = 'V', uplo = 'U';
        int info = 0, lda = n, lwork = -1;
        double query = 0.0;
        dsyev_(&jobz, &uplo, &n, Ac, &lda, eval, &query, &lwork, &info);
        lwork = (int)query;
        double *work = lwork > 0 ? (double *)malloc((size_t)lwork * sizeof(double)) : NULL;
        if (work && info == 0) {
            info = 0;
            dsyev_(&jobz, &uplo, &n, Ac, &lda, eval, work, &lwork, &info);
        }
        if (work && info == 0) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    V[i * n + j] = Ac[i + (size_t)j * (size_t)n];
            free(work);
            free(Ac);
            return;
        }
        free(work);
        free(Ac);
    }
    }
#endif
    jacobi(n, A, eval, V);
}

/* ---------------- build the node tables ---------------- */

static int half_n(double nu, int *n) {
    double a = fabs(nu);
    double h = round(a - 0.5);
    if (h >= -0.5 && h < 60.0 && fabs(a - (h + 0.5)) < 1e-10) {
        *n = (int)h;
        return 1;
    }
    return 0;
}

static void free_nodes(es4mgh *E) {
    free(E->u); E->u = NULL;
    free(E->wo); E->wo = NULL;
    free(E->chi_base); E->chi_base = NULL;
    free(E->psi_node); E->psi_node = NULL;
    free(E->lrho); E->lrho = NULL;
    free(E->a2p); E->a2p = NULL;
    free(E->a1p); E->a1p = NULL;
    free(E->lrp); E->lrp = NULL;
    free(E->psi_inv); E->psi_inv = NULL;
    free(E->log_psi); E->log_psi = NULL;
    E->nnode = 0;
}

/* Allocate the new tables first and swap only after they are filled. */
static int fill_nodes(es4mgh *E, int nn) {
    if (nn < 1) return 0;
    double *u = (double *)malloc((size_t)nn * sizeof(double));
    double *wo = (double *)malloc((size_t)nn * sizeof(double));
    double _Complex *chi_base = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *psi_node = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *lrho = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *a2p = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *a1p = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *lrp = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *psi_inv = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double _Complex *log_psi = (double _Complex *)malloc((size_t)nn * sizeof(double _Complex));
    double *gx = (double *)malloc((size_t)nn * sizeof(double));
    double *gw = (double *)malloc((size_t)nn * sizeof(double));
    if (!u || !wo || !chi_base || !psi_node || !lrho || !a2p || !a1p || !lrp ||
        !psi_inv || !log_psi || !gx || !gw) {
        free(u); free(wo); free(chi_base); free(psi_node); free(lrho);
        free(a2p); free(a1p); free(lrp); free(psi_inv); free(log_psi);
        free(gx); free(gw);
        return 0;
    }
    gauss_legendre(nn, gx, gw);
    const double *omega = E->spec;
    const double *d2 = omega ? omega + E->spec_n : NULL;
    const double *e2 = d2 ? d2 + E->spec_n : NULL;
    const double *de = E->de;
    const int ne = E->ne;
    const double ub = E->ub;
    const double chi = E->chi;
    const double psi = E->psi;
    const double D2z = E->D2z, E2z = E->E2z, DEz = E->DEz;
    const double c = E->ccoef, k = E->kcoef;
    for (int i = 0; i < nn; ++i) {
        double v = 0.5 * ub * (gx[i] + 1.0);
        double wv = 0.5 * ub * gw[i];
        double ss = v / (1.0 - v);
        double uu = ss * ss;
        double dudv = 2.0 * ss / ((1.0 - v) * (1.0 - v));
        u[i] = uu;
        wo[i] = wv * dudv / uu;

        double complex s = I * uu;
        double complex s2 = s * s;
        double complex t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
        double complex a2 = 0.0, a1 = 0.0, lr = 0.0;
        for (int j = 0; j < ne; ++j) {
            double complex nu = 1.0 / (1.0 - 2.0 * omega[j] * s);
            double complex nu2 = nu * nu;
            t1 += d2[j] * nu;
            t2 += e2[j] * nu;
            t3 += de[j] * nu;
            t4 += clog(nu);
            a2 += s * d2[j] * nu + s2 * d2[j] * omega[j] * nu2;
            a1 += s * e2[j] * nu + s2 * e2[j] * omega[j] * nu2;
            lr += 2.0 * s * de[j] * nu + 2.0 * s2 * de[j] * omega[j] * nu2 + omega[j] * nu;
        }
        t1 += D2z;
        t2 += E2z;
        t3 += DEz;
        a2 += s * D2z;
        a1 += s * E2z + k;
        lr += 2.0 * s * DEz + c;
        chi_base[i] = chi - s2 * t1;
        psi_node[i] = psi - 2.0 * (k * s + 0.5 * s2 * t2);
        lrho[i] = s * c + s2 * t3 + 0.5 * t4;
        a2p[i] = a2;
        a1p[i] = a1;
        lrp[i] = lr;
        {
            double complex p = psi_node[i];
            double pn = creal(p) * creal(p) + cimag(p) * cimag(p);
            if (pn == 0.0) {
                psi_inv[i] = 0.0;
                log_psi[i] = 0.0;
            } else {
                psi_inv[i] = 1.0 / p;
                log_psi[i] = clog(p);
            }
        }
    }
    free(gx);
    free(gw);
    free_nodes(E);
    E->u = u;
    E->wo = wo;
    E->chi_base = chi_base;
    E->psi_node = psi_node;
    E->lrho = lrho;
    E->a2p = a2p;
    E->a1p = a1p;
    E->lrp = lrp;
    E->psi_inv = psi_inv;
    E->log_psi = log_psi;
    E->nnode = nn;
    return 1;
}

static es4mgh *build_from_spectral(
    const double *omega_all, const double *d_all, const double *e_all, int ne_all,
    double c, double k, double kk, double lam, double chi, double psi) {

    double omax = 0.0;
    for (int i = 0; i < ne_all; ++i)
        if (fabs(omega_all[i]) > omax) omax = fabs(omega_all[i]);
    double otol = 1e-12 * (omax > 1.0 ? omax : 1.0);

    int ne = 0;
    double D2z = 0.0, E2z = 0.0, DEz = 0.0;
    double *omega = (double *)malloc((size_t)ne_all * sizeof(double) * 3);
    if (!omega && ne_all > 0) return NULL;
    double *d2 = omega + ne_all;
    double *e2 = d2 + ne_all;
    double *de = (double *)malloc((size_t)(ne_all > 0 ? ne_all : 1) * sizeof(double));
    if (!de) { free(omega); return NULL; }
    for (int i = 0; i < ne_all; ++i) {
        double di = d_all[i], ei = e_all[i];
        if (fabs(omega_all[i]) <= otol) {
            D2z += di * di;
            E2z += ei * ei;
            DEz += di * ei;
        } else {
            omega[ne] = omega_all[i];
            d2[ne] = di * di;
            e2[ne] = ei * ei;
            de[ne] = di * ei;
            ++ne;
        }
    }

    es4mgh *E = (es4mgh *)calloc(1, sizeof(*E));
    if (!E) { free(omega); free(de); return NULL; }
    E->ne = ne;
    E->kk = kk;
    E->lam = lam;
    E->chi = chi;
    E->psi = psi;
    E->need_a1 = (fabs(k) > 0.0 || E2z > 0.0);
    for (int i = 0; i < ne; ++i) if (e2[i] != 0.0 || de[i] != 0.0) E->need_a1 = 1;
    E->LK2 = lklam_real(lam, chi, psi);

    int hn = 0;
    if (fabs(lam + 0.5) < 1e-12 && chi > 0.0 && psi > 0.0) E->kind = KIND_NIG;
    else if (psi == 0.0 && fabs(k) == 0.0 && E2z == 0.0 && !E->need_a1) E->kind = KIND_PSI0;
    else if (psi == 0.0 && fabs(k) == 0.0 && E2z == 0.0) E->kind = KIND_PSI0;
    else if (half_n(lam, &hn)) E->kind = KIND_HALF;
    else E->kind = KIND_GENERAL;

    /* psi==0 and e==0,k==0 => psi argument stays 0 even if need_a1 is false.
       If e is not zero, psi argument moves and we cannot use the power branch. */
    if (E->kind == KIND_PSI0 && E->need_a1) E->kind = half_n(lam, &hn) ? KIND_HALF : KIND_GENERAL;

    if (E->kind == KIND_GENERAL) {
        double aord = fabs(lam);
        double mu = aord - floor(aord + 1e-14);
        if (mu > 1e-8 && mu < 1.0 - 1e-8) {
            E->g_fast = 1;
            E->g_mu = mu;
            E->g_sinmu = sin(PI * mu);
            E->g_sin1 = sin(PI * (1.0 - mu));
            E->g_lgp = lgamma(1.0 + mu);
            E->g_lgm = lgamma(1.0 - mu);
            E->g_lgq = lgamma(2.0 - mu);
            E->g_lgr = lgamma(mu);
            E->g_elgp = exp(-E->g_lgp);
            E->g_elgm = exp(-E->g_lgm);
            E->g_elgq = exp(-E->g_lgq);
            E->g_elgr = exp(-E->g_lgr);
            /* 1 / prod_{j=1}^{k} j*(order+j), the I-series coefficient of y^k. */
            double pp = 1.0, pm = 1.0, pq = 1.0, pr = 1.0;
            for (int k = 1; k <= 28; ++k) {
                double kd = (double)k;
                pp *= kd * (mu + kd);
                pm *= kd * (-mu + kd);
                pq *= kd * (1.0 - mu + kd);
                pr *= kd * (mu - 1.0 + kd);
                E->g_ip[k] = 1.0 / pp;
                E->g_im[k] = 1.0 / pm;
                E->g_iq[k] = 1.0 / pq;
                E->g_ir[k] = 1.0 / pr;
            }
        }
    }
    if (E->kind == KIND_PSI0) {
        double c0 = lgamma(-lam) - lam * LOG2 - E->LK2;
        double c1 = lgamma(-(lam + 1.0)) - (lam + 1.0) * LOG2 - E->LK2;
        double c2 = lgamma(-(lam + 2.0)) - (lam + 2.0) * LOG2 - E->LK2;
        E->psi_c1 = exp(c1 - c0);
        E->psi_c2 = exp(c2 - c0);
    }
    E->D2z = D2z;
    E->E2z = E2z;
    E->DEz = DEz;
    E->ccoef = c;
    E->kcoef = k;
    E->spec = omega;
    E->de = de;
    E->spec_n = ne_all;
    E->ub = integration_ub(omega_all, ne_all);
    E->nlo = 0;
    E->fixed_nnode = env_nnode();
    {
        int nn = E->fixed_nnode > 0 ? E->fixed_nnode : kind_nnode(E->kind);
        if (!fill_nodes(E, nn)) {
            es4mgh_free(E);
            return NULL;
        }
    }

    /* Partial-moment anchor M2(0). Independent of q. */
    {
        double complex lm1 = lklam(lam + 1.0, chi, psi) - E->LK2;
        double sum_om = 0.0;
        for (int i = 0; i < ne_all; ++i) sum_om += omega_all[i];
        /* k*E[W^2] is the zero random variable when k is 0. For psi = 0 and
           lambda = -2, E[W^2] is infinite and cexp(lm2)*0 is a NaN. */
        double complex skew = 0.0;
        if (k != 0.0) {
            double complex lm2 = lklam(lam + 2.0, chi, psi) - E->LK2;
            skew = cexp(lm2) * k;
        }
        E->M20 = creal(skew + cexp(lm1) * (c + sum_om));
    }
    return E;
}

es4mgh *es4mgh_create_spectral(
    int ne, const double *omega, const double *dvec, const double *evec,
    double c, double k, double kk, double lam, double chi, double psi) {
    if (ne < 0) return NULL;
    return build_from_spectral(omega, dvec, evec, ne, c, k, kk, lam, chi, psi);
}

es4mgh *es4mgh_create(
    int d, double a0, const double *a, const double *A, const double *C,
    const double *mu, const double *gam, double lam, double chi, double psi) {
    if (d <= 0) return NULL;
    double *As = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
    double *CAC = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
    double *CT = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
    double *eval = (double *)malloc((size_t)d * sizeof(double));
    double *V = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
    double *muA = (double *)malloc((size_t)d * sizeof(double));
    double *gA = (double *)malloc((size_t)d * sizeof(double));
    double *CP = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
    double *dvec = (double *)malloc((size_t)d * sizeof(double));
    double *evec = (double *)malloc((size_t)d * sizeof(double));
    if (!As || !CAC || !CT || !eval || !V || !muA || !gA || !CP || !dvec || !evec) {
        free(As); free(CAC); free(CT); free(eval); free(V);
        free(muA); free(gA); free(CP); free(dvec); free(evec);
        return NULL;
    }
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            As[i * d + j] = 0.5 * (A[i * d + j] + A[j * d + i]);

    /* CAC = C^T As C. CT holds C^T for the portable product. */
    double *tmp = CAC;
    double *TAs = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
    if (!TAs) {
        free(As); free(CAC); free(CT); free(eval); free(V);
        free(muA); free(gA); free(CP); free(dvec); free(evec);
        return NULL;
    }
#if defined(__APPLE__)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d, d, d, 1.0, C, d, As, d, 0.0, TAs, d);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                d, d, d, 1.0, TAs, d, C, d, 0.0, tmp, d);
    (void)CT;
#else
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            CT[i * d + j] = C[j * d + i];
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) s += CT[i * d + k] * As[k * d + j];
            TAs[i * d + j] = s;
        }
    }
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) s += TAs[i * d + k] * C[k * d + j];
            tmp[i * d + j] = s;
        }
    }
#endif
    for (int i = 0; i < d; ++i)
        for (int j = i + 1; j < d; ++j) {
            double v = 0.5 * (tmp[i * d + j] + tmp[j * d + i]);
            tmp[i * d + j] = tmp[j * d + i] = v;
        }
    eigen_symmetric(d, tmp, eval, V);

    for (int i = 0; i < d; ++i) {
        double s = 0.0, g = 0.0;
        for (int k = 0; k < d; ++k) {
            s += mu[k] * As[k * d + i];
            g += gam[k] * As[k * d + i];
        }
        muA[i] = s;
        gA[i] = g;
    }
    /* CP = C * V, V's columns are eigenvectors, V row-major V[r*d+c] */
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int k = 0; k < d; ++k) s += C[i * d + k] * V[k * d + j];
            CP[i * d + j] = s;
        }
    double ccoef = 0.0, kk = a0, kv = 0.0;
    for (int i = 0; i < d; ++i) {
        ccoef += a[i] * gam[i] + 2.0 * muA[i] * gam[i];
        kk += a[i] * mu[i] + muA[i] * mu[i];
        kv += gA[i] * gam[i];
    }
    for (int j = 0; j < d; ++j) {
        double dj = 0.0, ej = 0.0;
        for (int i = 0; i < d; ++i) {
            dj += (a[i] + 2.0 * muA[i]) * CP[i * d + j];
            ej += 2.0 * gA[i] * CP[i * d + j];
        }
        dvec[j] = dj;
        evec[j] = ej;
    }
    es4mgh *E = build_from_spectral(eval, dvec, evec, d, ccoef, kv, kk, lam, chi, psi);
    free(As); free(CAC); free(CT); free(eval); free(V);
    free(muA); free(gA); free(CP); free(dvec); free(evec); free(TAs);
    return E;
}

void es4mgh_free(es4mgh *E) {
    if (!E) return;
    free(E->u); free(E->wo);
    free(E->chi_base); free(E->psi_node); free(E->lrho);
    free(E->a2p); free(E->a1p); free(E->lrp);
    free(E->psi_inv); free(E->log_psi);
    free(E->spec); free(E->de);
    free(E);
}

/* ---------------- small-|z| K for three orders at once ---------------- */

typedef struct { double r, i; } Cpx;
static inline Cpx Cx(double r, double i) { Cpx c; c.r = r; c.i = i; return c; }
static inline Cpx cadd2(Cpx a, Cpx b) { return Cx(a.r + b.r, a.i + b.i); }
static inline Cpx cmul2(Cpx a, Cpx b) {
    return Cx(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}
static inline Cpx cscale2(Cpx a, double s) { return Cx(a.r * s, a.i * s); }
static inline Cpx cdivr2(Cpx a, double s) { return Cx(a.r / s, a.i / s); }
/* I-series at orders mu, -mu, 1-mu, mu-1. Fixed length: |z| < 4 needs <= 28 terms. */
static void series4(Cpx y, double mu, int nterm, Cpx *Sp, Cpx *Sm, Cpx *Sq, Cpx *Sr) {
    Cpx tp = Cx(1, 0), tm = Cx(1, 0), tq = Cx(1, 0), tr = Cx(1, 0);
    Cpx sp = Cx(1, 0), sm = Cx(1, 0), sq = Cx(1, 0), sr = Cx(1, 0);
    for (int k = 1; k <= nterm; ++k) {
        double kd = (double)k;
        tp = cdivr2(cmul2(tp, y), kd * (mu + kd));
        tm = cdivr2(cmul2(tm, y), kd * (-mu + kd));
        tq = cdivr2(cmul2(tq, y), kd * (1.0 - mu + kd));
        tr = cdivr2(cmul2(tr, y), kd * (mu - 1.0 + kd));
        sp = cadd2(sp, tp);
        sm = cadd2(sm, tm);
        sq = cadd2(sq, tq);
        sr = cadd2(sr, tr);
    }
    *Sp = sp; *Sm = sm; *Sq = sq; *Sr = sr;
}

/* Four nodes at once. The four I-series are independent, so interleaving them
   hides the latency of the complex multiply chain. */
static void series4x4_16(Cpx y0, Cpx y1, Cpx y2, Cpx y3, double mu,
                         Cpx Sp[4], Cpx Sm[4], Cpx Sq[4], Cpx Sr[4]) {
#define STEP4(tp, tm, tq, tr, sp, sm, sq, sr, y) \
    do { \
        tp = cdivr2(cmul2(tp, y), dp); sp = cadd2(sp, tp); \
        tm = cdivr2(cmul2(tm, y), dm); sm = cadd2(sm, tm); \
        tq = cdivr2(cmul2(tq, y), dq); sq = cadd2(sq, tq); \
        tr = cdivr2(cmul2(tr, y), dr); sr = cadd2(sr, tr); \
    } while (0)
    Cpx tp0 = Cx(1, 0), tm0 = Cx(1, 0), tq0 = Cx(1, 0), tr0 = Cx(1, 0);
    Cpx sp0 = Cx(1, 0), sm0 = Cx(1, 0), sq0 = Cx(1, 0), sr0 = Cx(1, 0);
    Cpx tp1 = Cx(1, 0), tm1 = Cx(1, 0), tq1 = Cx(1, 0), tr1 = Cx(1, 0);
    Cpx sp1 = Cx(1, 0), sm1 = Cx(1, 0), sq1 = Cx(1, 0), sr1 = Cx(1, 0);
    Cpx tp2 = Cx(1, 0), tm2 = Cx(1, 0), tq2 = Cx(1, 0), tr2 = Cx(1, 0);
    Cpx sp2 = Cx(1, 0), sm2 = Cx(1, 0), sq2 = Cx(1, 0), sr2 = Cx(1, 0);
    Cpx tp3 = Cx(1, 0), tm3 = Cx(1, 0), tq3 = Cx(1, 0), tr3 = Cx(1, 0);
    Cpx sp3 = Cx(1, 0), sm3 = Cx(1, 0), sq3 = Cx(1, 0), sr3 = Cx(1, 0);
#pragma clang loop unroll(full)
    for (int k = 1; k <= 16; ++k) {
        double kd = (double)k;
        double dp = kd * (mu + kd);
        double dm = kd * (-mu + kd);
        double dq = kd * (1.0 - mu + kd);
        double dr = kd * (mu - 1.0 + kd);
        STEP4(tp0, tm0, tq0, tr0, sp0, sm0, sq0, sr0, y0);
        STEP4(tp1, tm1, tq1, tr1, sp1, sm1, sq1, sr1, y1);
        STEP4(tp2, tm2, tq2, tr2, sp2, sm2, sq2, sr2, y2);
        STEP4(tp3, tm3, tq3, tr3, sp3, sm3, sq3, sr3, y3);
    }
#undef STEP4
    Sp[0] = sp0; Sm[0] = sm0; Sq[0] = sq0; Sr[0] = sr0;
    Sp[1] = sp1; Sm[1] = sm1; Sq[1] = sq1; Sr[1] = sr1;
    Sp[2] = sp2; Sm[2] = sm2; Sq[2] = sq2; Sr[2] = sr2;
    Sp[3] = sp3; Sm[3] = sm3; Sq[3] = sq3; Sr[3] = sr3;
}

static void series4x4(Cpx y0, Cpx y1, Cpx y2, Cpx y3, double mu, int nterm,
                      Cpx Sp[4], Cpx Sm[4], Cpx Sq[4], Cpx Sr[4]) {
#define STEP4(tp, tm, tq, tr, sp, sm, sq, sr, y) \
    do { \
        tp = cdivr2(cmul2(tp, y), dp); sp = cadd2(sp, tp); \
        tm = cdivr2(cmul2(tm, y), dm); sm = cadd2(sm, tm); \
        tq = cdivr2(cmul2(tq, y), dq); sq = cadd2(sq, tq); \
        tr = cdivr2(cmul2(tr, y), dr); sr = cadd2(sr, tr); \
    } while (0)
    Cpx tp0 = Cx(1, 0), tm0 = Cx(1, 0), tq0 = Cx(1, 0), tr0 = Cx(1, 0);
    Cpx sp0 = Cx(1, 0), sm0 = Cx(1, 0), sq0 = Cx(1, 0), sr0 = Cx(1, 0);
    Cpx tp1 = Cx(1, 0), tm1 = Cx(1, 0), tq1 = Cx(1, 0), tr1 = Cx(1, 0);
    Cpx sp1 = Cx(1, 0), sm1 = Cx(1, 0), sq1 = Cx(1, 0), sr1 = Cx(1, 0);
    Cpx tp2 = Cx(1, 0), tm2 = Cx(1, 0), tq2 = Cx(1, 0), tr2 = Cx(1, 0);
    Cpx sp2 = Cx(1, 0), sm2 = Cx(1, 0), sq2 = Cx(1, 0), sr2 = Cx(1, 0);
    Cpx tp3 = Cx(1, 0), tm3 = Cx(1, 0), tq3 = Cx(1, 0), tr3 = Cx(1, 0);
    Cpx sp3 = Cx(1, 0), sm3 = Cx(1, 0), sq3 = Cx(1, 0), sr3 = Cx(1, 0);
    for (int k = 1; k <= nterm; ++k) {
        double kd = (double)k;
        double dp = kd * (mu + kd);
        double dm = kd * (-mu + kd);
        double dq = kd * (1.0 - mu + kd);
        double dr = kd * (mu - 1.0 + kd);
        STEP4(tp0, tm0, tq0, tr0, sp0, sm0, sq0, sr0, y0);
        STEP4(tp1, tm1, tq1, tr1, sp1, sm1, sq1, sr1, y1);
        STEP4(tp2, tm2, tq2, tr2, sp2, sm2, sq2, sr2, y2);
        STEP4(tp3, tm3, tq3, tr3, sp3, sm3, sq3, sr3, y3);
    }
#undef STEP4
    Sp[0] = sp0; Sm[0] = sm0; Sq[0] = sq0; Sr[0] = sr0;
    Sp[1] = sp1; Sm[1] = sm1; Sq[1] = sq1; Sr[1] = sr1;
    Sp[2] = sp2; Sm[2] = sm2; Sq[2] = sq2; Sr[2] = sr2;
    Sp[3] = sp3; Sm[3] = sm3; Sq[3] = sq3; Sr[3] = sr3;
}

/* One shared power y^k for the four orders. The coefficients are 1/pochhammer. */
static void series_pow4(const double *ip, const double *im, const double *iq, const double *ir,
                        int nterm, Cpx y0, Cpx y1, Cpx y2, Cpx y3,
                        Cpx Sp[4], Cpx Sm[4], Cpx Sq[4], Cpx Sr[4]) {
    Cpx p0 = Cx(1, 0), p1 = Cx(1, 0), p2 = Cx(1, 0), p3 = Cx(1, 0);
    Cpx s0p = Cx(1, 0), s0m = Cx(1, 0), s0q = Cx(1, 0), s0r = Cx(1, 0);
    Cpx s1p = Cx(1, 0), s1m = Cx(1, 0), s1q = Cx(1, 0), s1r = Cx(1, 0);
    Cpx s2p = Cx(1, 0), s2m = Cx(1, 0), s2q = Cx(1, 0), s2r = Cx(1, 0);
    Cpx s3p = Cx(1, 0), s3m = Cx(1, 0), s3q = Cx(1, 0), s3r = Cx(1, 0);
    for (int k = 1; k <= nterm; ++k) {
        double ap = ip[k], am = im[k], aq = iq[k], ar = ir[k];
        p0 = cmul2(p0, y0);
        p1 = cmul2(p1, y1);
        p2 = cmul2(p2, y2);
        p3 = cmul2(p3, y3);
        s0p = cadd2(s0p, cscale2(p0, ap)); s0m = cadd2(s0m, cscale2(p0, am));
        s0q = cadd2(s0q, cscale2(p0, aq)); s0r = cadd2(s0r, cscale2(p0, ar));
        s1p = cadd2(s1p, cscale2(p1, ap)); s1m = cadd2(s1m, cscale2(p1, am));
        s1q = cadd2(s1q, cscale2(p1, aq)); s1r = cadd2(s1r, cscale2(p1, ar));
        s2p = cadd2(s2p, cscale2(p2, ap)); s2m = cadd2(s2m, cscale2(p2, am));
        s2q = cadd2(s2q, cscale2(p2, aq)); s2r = cadd2(s2r, cscale2(p2, ar));
        s3p = cadd2(s3p, cscale2(p3, ap)); s3m = cadd2(s3m, cscale2(p3, am));
        s3q = cadd2(s3q, cscale2(p3, aq)); s3r = cadd2(s3r, cscale2(p3, ar));
    }
    Sp[0] = s0p; Sm[0] = s0m; Sq[0] = s0q; Sr[0] = s0r;
    Sp[1] = s1p; Sm[1] = s1m; Sq[1] = s1q; Sr[1] = s1r;
    Sp[2] = s2p; Sm[2] = s2m; Sq[2] = s2q; Sr[2] = s2r;
    Sp[3] = s3p; Sm[3] = s3m; Sq[3] = s3q; Sr[3] = s3r;
}

/* exp(ph) * K_mu and exp(ph) * K_{1-mu}. The MGF phase is inside the
   prefactors, so the caller takes a plain imaginary part. */
static void U_from_sums(const es4mgh *E, double complex z, double complex ph,
                        Cpx Sp, Cpx Sm, Cpx Sq, Cpx Sr,
                        double complex *Umu, double complex *U1m) {
    double complex L = clog(0.5 * z);
    double mu = E->g_mu;
    double complex Ep = cexp(ph + mu * L - E->g_lgp);
    double complex Em = cexp(ph - mu * L - E->g_lgm);
    double complex Eq = cexp(ph + (1.0 - mu) * L - E->g_lgq);
    double complex Er = cexp(ph + (mu - 1.0) * L - E->g_lgr);
    double complex Ip = Ep * (Sp.r + I * Sp.i);
    double complex Im = Em * (Sm.r + I * Sm.i);
    double complex Iq = Eq * (Sq.r + I * Sq.i);
    double complex Ir = Er * (Sr.r + I * Sr.i);
    *Umu = (PI / (2.0 * E->g_sinmu)) * (Im - Ip);
    *U1m = (PI / (2.0 * E->g_sin1)) * (Ir - Iq);
}

/* K_mu and K_{1-mu} for mu in (0,1) and moderate |z|. */
static void K3_small(const es4mgh *E, double complex z, int nterm,
                     double complex *Kmu, double complex *K1m) {
    Cpx zz = Cx(creal(z), cimag(z));
    Cpx y = cscale2(cmul2(zz, zz), 0.25);
    Cpx Sp, Sm, Sq, Sr;
    series4(y, E->g_mu, nterm, &Sp, &Sm, &Sq, &Sr);
    /* K, not exp(ph)*K. ph = 0 recovers K because the prefactor has no phase. */
    U_from_sums(E, z, 0.0, Sp, Sm, Sq, Sr, Kmu, K1m);
}

/* K_order on the mu ladder or the (1-mu) ladder. */
static double complex K_order(double order, double mu, double complex invz,
                              double complex Kmu, double complex K1m) {
    order = fabs(order);
    double base, frac = order - floor(order);
    double complex prev, cur;
    if (fabs(frac - mu) <= 1e-6) {
        base = mu;
        prev = K1m; /* K_{mu-1} = K_{1-mu} */
        cur = Kmu;
    } else if (fabs(frac - (1.0 - mu)) <= 1e-6) {
        base = 1.0 - mu;
        prev = Kmu; /* K_{(1-mu)-1} = K_mu */
        cur = K1m;
    } else {
        return NAN;
    }
    int steps = (int)llround(order - base);
    if (steps < 0) return NAN;
    for (int s = 0; s < steps; ++s) {
        double nu = base + (double)s;
        double complex nxt = (2.0 * nu) * invz * cur + prev;
        prev = cur;
        cur = nxt;
    }
    return cur;
}

/* ---------------- integrands ---------------- */

static void eval_point(const es4mgh *E, double q, double *ccdf, double *es) {
    const int nn = E->nnode;
    const double lam = E->lam;
    const double LK2 = E->LK2;
    double Ic = 0.0, Ip = 0.0;
    const int do_es = es != NULL;

    if (E->kind == KIND_NIG) {
        for (int i = 0; i < nn; ++i) {
            double complex chi = E->chi_base[i] + I * (2.0 * E->u[i] * q);
            double complex psi = E->psi_node[i];
            double complex z = csqrt(chi * psi);
            double complex logK = -z + 0.5 * clog(PI / (2.0 * z));
            double complex lrat = clog(chi / psi);
            double complex lm0 = LOG2 - 0.25 * lrat + logK - LK2 + E->lrho[i];
            Ic += E->wo[i] * imag_exp(lm0);
            if (do_es) {
                double complex lm1 = LOG2 + 0.25 * lrat + logK - LK2 + E->lrho[i];
                double acc = imag_exp_mul(lm0, E->a2p[i]) + imag_exp_mul(lm1, E->lrp[i]);
                if (E->need_a1) {
                    double complex logK32 = logK + clog(1.0 + 1.0 / z);
                    double complex lm2 = LOG2 + 0.75 * lrat + logK32 - LK2 + E->lrho[i];
                    acc += imag_exp_mul(lm2, E->a1p[i]);
                }
                Ip += E->wo[i] * acc;
            }
        }
    } else if (E->kind == KIND_PSI0) {
        double c0 = lgamma(-lam) - lam * LOG2 - LK2;
        double c1 = lgamma(-(lam + 1.0)) - (lam + 1.0) * LOG2 - LK2;
        double c2 = lgamma(-(lam + 2.0)) - (lam + 2.0) * LOG2 - LK2;
        for (int i = 0; i < nn; ++i) {
            double complex chi = E->chi_base[i] + I * (2.0 * E->u[i] * q);
            double complex lc = clog(chi);
            double complex base = E->lrho[i];
            double complex lm0 = lam * lc + c0 + base;
            Ic += E->wo[i] * imag_exp(lm0);
            if (do_es) {
                double complex lm1 = (lam + 1.0) * lc + c1 + base;
                double complex acc = imag_exp_mul(lm0, E->a2p[i]) + imag_exp_mul(lm1, E->lrp[i]);
                if (E->need_a1) {
                    double complex lm2 = (lam + 2.0) * lc + c2 + base;
                    acc += imag_exp_mul(lm2, E->a1p[i]);
                }
                Ip += E->wo[i] * acc;
            }
        }
    } else if (E->kind == KIND_HALF) {
        for (int i = 0; i < nn; ++i) {
            double complex chi = E->chi_base[i] + I * (2.0 * E->u[i] * q);
            double complex psi = E->psi_node[i];
            /* chi = 0 or psi = 0 is the gamma limit. clog of that argument is NaN. */
            if ((creal(chi) == 0.0 && cimag(chi) == 0.0) ||
                (creal(psi) == 0.0 && cimag(psi) == 0.0)) {
                double complex base = -LK2 + E->lrho[i];
                double complex lm0 = lklam(lam, chi, psi) + base;
                Ic += E->wo[i] * imag_exp(lm0);
                if (do_es) {
                    double complex lm1 = lklam(lam + 1.0, chi, psi) + base;
                    double acc = imag_exp_mul(lm0, E->a2p[i]) + imag_exp_mul(lm1, E->lrp[i]);
                    if (E->need_a1) {
                        double complex lm2 = lklam(lam + 2.0, chi, psi) + base;
                        acc += imag_exp_mul(lm2, E->a1p[i]);
                    }
                    Ip += E->wo[i] * acc;
                }
                continue;
            }
            double complex prod = chi * psi;
            if (creal(prod) < 0.0 && fabs(cimag(prod)) <= 1e-14 * (1.0 + fabs(creal(prod))))
                continue;
            double complex z = csqrt(prod);
            double complex lrat = clog(chi / psi);
            double complex base = -LK2 + E->lrho[i];
            double complex lm0 = LOG2 + 0.5 * lam * lrat + (log_scaled_K(lam, z) - z) + base;
            Ic += E->wo[i] * imag_exp(lm0);
            if (do_es) {
                double complex lm1 = LOG2 + 0.5 * (lam + 1.0) * lrat + (log_scaled_K(lam + 1.0, z) - z) + base;
                double acc = imag_exp_mul(lm0, E->a2p[i]) + imag_exp_mul(lm1, E->lrp[i]);
                if (E->need_a1) {
                    double complex lm2 = LOG2 + 0.5 * (lam + 2.0) * lrat + (log_scaled_K(lam + 2.0, z) - z) + base;
                    acc += imag_exp_mul(lm2, E->a1p[i]);
                }
                Ip += E->wo[i] * acc;
            }
        }
    } else if (E->g_fast) {
        const double mu = E->g_mu;
        int i = 0;
        for (; i + 4 <= nn; i += 4) {
            double complex zc[4], chi[4], base[4];
            Cpx y[4];
            int nterm = 16;
            int ok = 1;
            for (int j = 0; j < 4; ++j) {
                int ix = i + j;
                chi[j] = E->chi_base[ix] + I * (2.0 * E->u[ix] * q);
                double complex psi = E->psi_node[ix];
                base[j] = -LK2 + E->lrho[ix];
                double complex prod = chi[j] * psi;
                int chi0 = creal(chi[j]) == 0.0 && cimag(chi[j]) == 0.0;
                int psi0 = creal(psi) == 0.0 && cimag(psi) == 0.0;
                if (chi0 || psi0 || (creal(prod) < 0.0 &&
                    fabs(cimag(prod)) <= 1e-14 * (1.0 + fabs(creal(prod))))) {
                    ok = 0;
                    break;
                }
                zc[j] = csqrt(prod);
                double r2 = creal(zc[j]) * creal(zc[j]) + cimag(zc[j]) * cimag(zc[j]);
                if (r2 < 1e-8 || r2 > 16.0) { ok = 0; break; }
                if (r2 > 5.0) nterm = 28;
                y[j] = cscale2(cmul2(Cx(creal(zc[j]), cimag(zc[j])),
                                     Cx(creal(zc[j]), cimag(zc[j]))), 0.25);
            }
            if (!ok) break;
            Cpx Sp[4], Sm[4], Sq[4], Sr[4];
            if (nterm == 16) series4x4_16(y[0], y[1], y[2], y[3], mu, Sp, Sm, Sq, Sr);
            else series4x4(y[0], y[1], y[2], y[3], mu, nterm, Sp, Sm, Sq, Sr);
            for (int j = 0; j < 4; ++j) {
                int ix = i + j;
                double complex lrat = clog(chi[j] / E->psi_node[ix]);
                double complex ph0 = LOG2 + 0.5 * lam * lrat + base[j];
                double complex Umu, U1m;
                U_from_sums(E, zc[j], ph0, Sp[j], Sm[j], Sq[j], Sr[j], &Umu, &U1m);
                double complex invz = 1.0 / zc[j];
                double complex U0 = K_order(lam, mu, invz, Umu, U1m);
                Ic += E->wo[ix] * cimag(U0);
                if (do_es) {
                    double complex eh = cexp(0.5 * lrat);
                    double complex U1 = K_order(lam + 1.0, mu, invz, Umu, U1m);
                    double acc = cimag(U0 * E->a2p[ix]) + cimag(eh * U1 * E->lrp[ix]);
                    if (E->need_a1) {
                        double complex U2 = K_order(lam + 2.0, mu, invz, Umu, U1m);
                        acc += cimag(eh * eh * U2 * E->a1p[ix]);
                    }
                    Ip += E->wo[ix] * acc;
                }
            }
        }
        for (; i < nn; ++i) {
            double complex chi = E->chi_base[i] + I * (2.0 * E->u[i] * q);
            double complex psi = E->psi_node[i];
            double complex base = -LK2 + E->lrho[i];
            double complex prod = chi * psi;
            int chi0 = creal(chi) == 0.0 && cimag(chi) == 0.0;
            int psi0 = creal(psi) == 0.0 && cimag(psi) == 0.0;
            int bad = chi0 || psi0 ||
                      (creal(prod) < 0.0 && fabs(cimag(prod)) <= 1e-14 * (1.0 + fabs(creal(prod))));
            double complex z = 0.0;
            double r2 = 0.0;
            if (!bad) {
                z = csqrt(prod);
                r2 = creal(z) * creal(z) + cimag(z) * cimag(z);
                bad = r2 < 1e-8 || r2 > 16.0;
            }
            if (bad) {
                double complex lm0 = lklam(lam, chi, psi) + base;
                Ic += E->wo[i] * imag_exp(lm0);
                if (do_es) {
                    double complex lm1 = lklam(lam + 1.0, chi, psi) + base;
                    double acc = imag_exp_mul(lm0, E->a2p[i]) + imag_exp_mul(lm1, E->lrp[i]);
                    if (E->need_a1) {
                        double complex lm2 = lklam(lam + 2.0, chi, psi) + base;
                        acc += imag_exp_mul(lm2, E->a1p[i]);
                    }
                    Ip += E->wo[i] * acc;
                }
                continue;
            }
            int nterm = r2 <= 5.0 ? 16 : 28;
            double complex Kmu, K1m;
            K3_small(E, z, nterm, &Kmu, &K1m);
            double complex invz = 1.0 / z;
            double complex K0 = K_order(lam, mu, invz, Kmu, K1m);
            double complex lrat = clog(chi / psi);
            double complex ph0 = LOG2 + 0.5 * lam * lrat + base;
            Ic += E->wo[i] * imag_exp_mul(ph0, K0);
            if (do_es) {
                double complex K1 = K_order(lam + 1.0, mu, invz, Kmu, K1m);
                double complex ph1 = LOG2 + 0.5 * (lam + 1.0) * lrat + base;
                double acc = imag_exp_mul(ph0, E->a2p[i] * K0) + imag_exp_mul(ph1, E->lrp[i] * K1);
                if (E->need_a1) {
                    double complex K2 = K_order(lam + 2.0, mu, invz, Kmu, K1m);
                    double complex ph2 = LOG2 + 0.5 * (lam + 2.0) * lrat + base;
                    acc += imag_exp_mul(ph2, E->a1p[i] * K2);
                }
                Ip += E->wo[i] * acc;
            }
        }
    } else {
        for (int i = 0; i < nn; ++i) {
            double complex chi = E->chi_base[i] + I * (2.0 * E->u[i] * q);
            double complex psi = E->psi_node[i];
            double complex base = -LK2 + E->lrho[i];
            double complex lm0 = lklam(lam, chi, psi) + base;
            Ic += E->wo[i] * imag_exp(lm0);
            if (do_es) {
                double complex lm1 = lklam(lam + 1.0, chi, psi) + base;
                double acc = imag_exp_mul(lm0, E->a2p[i]) + imag_exp_mul(lm1, E->lrp[i]);
                if (E->need_a1) {
                    double complex lm2 = lklam(lam + 2.0, chi, psi) + base;
                    acc += imag_exp_mul(lm2, E->a1p[i]);
                }
                Ip += E->wo[i] * acc;
            }
        }
    }

    double cval = 0.5 + INV_PI * Ic;
    if (ccdf) *ccdf = cval;
    if (es) {
        double den = ccdf ? cval : 0.5 + INV_PI * Ic;
        *es = (0.5 * E->M20 + INV_PI * Ip) / den + E->kk;
    }
}

typedef struct {
    const es4mgh *E;
    const double *x;
    double *ccdf;
    double *es;
    double kk;
    int i0, i1;
} Chunk;

static void eval_chunk(const Chunk *c) {
    for (int i = c->i0; i < c->i1; ++i) {
        double q = c->x[i] - c->kk;
        double *cc = c->ccdf ? c->ccdf + i : NULL;
        double *ee = c->es ? c->es + i : NULL;
        eval_point(c->E, q, cc, ee);
    }
}

enum { FAST_MAX = 1024 };

static _Thread_local unsigned char *arena_buf;
static _Thread_local size_t arena_cap;

static void *arena_acquire(size_t bytes) {
    if (bytes > arena_cap) {
        free(arena_buf);
        arena_buf = (unsigned char *)malloc(bytes);
        arena_cap = arena_buf ? bytes : 0;
    }
    return arena_buf;
}

static int host_threads(void) {
    static int cached = -1;
    if (cached > 0) return cached;
    int np = 0;
#if defined(__APPLE__)
    size_t sz = sizeof(np);
    if (sysctlbyname("hw.logicalcpu", &np, &sz, NULL, 0) != 0) np = 0;
#elif defined(_WIN32)
    DWORD nproc = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    np = nproc > 0 ? (int)nproc : 1;
#else
    long all = sysconf(_SC_NPROCESSORS_ONLN);
    np = all > 0 ? (int)all : 1;
#endif
    if (np < 1) np = 1;
    cached = np;
    return cached;
}

static int use_slow(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ES4_SLOW");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static void write_tail(const es4mgh *E, double Ic, double Ip, int do_es,
                       double *ccdf, double *es) {
    double cval = 0.5 + INV_PI * Ic;
    if (ccdf) *ccdf = cval;
    if (do_es) *es = (0.5 * E->M20 + INV_PI * Ip) / cval + E->kk;
}

/* Fractional-order nodes for many thresholds. Returns 0 if a node leaves
   the series region, in which case the caller uses the scalar kernel. */
static int gfast_many(const es4mgh *E, int nq, const double *x,
                      double *ccdf, double *es) {
    const int nn = E->nnode;
    if (nn <= 0 || nq <= 0) return 1;
    if ((long)nq * nn > FAST_MAX) {
        int tile = FAST_MAX / nn;
        if (tile < 1) return 0;
        for (int i = 0; i < nq; i += tile) {
            int m = nq - i;
            if (m > tile) m = tile;
            if (!gfast_many(E, m, x + i, ccdf ? ccdf + i : NULL, es ? es + i : NULL))
                return 0;
        }
        return 1;
    }
    const int N = nq * nn;
    const int do_es = es != NULL;
    const double mu = E->g_mu;
    const double lam = E->lam;
    size_t bytes = (size_t)N * (6 * sizeof(double complex) + 5 * sizeof(Cpx)) + 256;
    unsigned char *block = (unsigned char *)arena_acquire(bytes);
    if (!block) return 0;
    size_t off = 0;
#define TAKE(T, name) T *name = (T *)(block + off); off = (off + (size_t)N * sizeof(T) + 15u) & ~(size_t)15u
    TAKE(double complex, zc);
    TAKE(double complex, chi);
    TAKE(double complex, Lc);
    TAKE(double complex, ph);
    TAKE(double complex, Wc);
    TAKE(double complex, Pc);
    TAKE(Cpx, y);
    TAKE(Cpx, Sp);
    TAKE(Cpx, Sm);
    TAKE(Cpx, Sq);
    TAKE(Cpx, Sr);
#undef TAKE
    int nterm = 16;
    int ok = 1;
    for (int qi = 0; qi < nq && ok; ++qi) {
        double q = x[qi] - E->kk;
        for (int i = 0; i < nn; ++i) {
            int t = qi * nn + i;
            double complex c = E->chi_base[i] + I * (2.0 * E->u[i] * q);
            double complex psi = E->psi_node[i];
            double complex prod = c * psi;
            int chi0 = creal(c) == 0.0 && cimag(c) == 0.0;
            int psi0 = creal(psi) == 0.0 && cimag(psi) == 0.0;
            if (chi0 || psi0 || (creal(prod) < 0.0 &&
                fabs(cimag(prod)) <= 1e-14 * (1.0 + fabs(creal(prod))))) {
                ok = 0;
                break;
            }
            double complex z = csqrt(prod);
            double r2 = creal(z) * creal(z) + cimag(z) * cimag(z);
            if (r2 < 1e-8 || r2 > 16.0) { ok = 0; break; }
            if (r2 > 5.0) nterm = 28;
            chi[t] = c;
            zc[t] = z;
            y[t] = cscale2(cmul2(Cx(creal(z), cimag(z)), Cx(creal(z), cimag(z))), 0.25);
        }
    }
    if (ok) {
        int t = 0;
        for (; t + 4 <= N; t += 4) {
            Cpx Spp[4], Smm[4], Sqq[4], Srr[4];
            series_pow4(E->g_ip, E->g_im, E->g_iq, E->g_ir, nterm,
                        y[t], y[t + 1], y[t + 2], y[t + 3], Spp, Smm, Sqq, Srr);
            for (int j = 0; j < 4; ++j) {
                Sp[t + j] = Spp[j];
                Sm[t + j] = Smm[j];
                Sq[t + j] = Sqq[j];
                Sr[t + j] = Srr[j];
            }
        }
        for (; t < N; ++t) {
            Cpx Spp[4], Smm[4], Sqq[4], Srr[4];
            Cpx z0 = Cx(0, 0);
            series_pow4(E->g_ip, E->g_im, E->g_iq, E->g_ir, nterm,
                        y[t], z0, z0, z0, Spp, Smm, Sqq, Srr);
            Sp[t] = Spp[0]; Sm[t] = Smm[0]; Sq[t] = Sqq[0]; Sr[t] = Srr[0];
        }
        for (int t = 0; t < N; ++t) Lc[t] = clog(0.5 * zc[t]);
        for (int t = 0; t < N; ++t) {
            int i = t % nn;
            double complex lrat = clog(chi[t]) - E->log_psi[i];
            ph[t] = LOG2 + 0.5 * lam * lrat + (-E->LK2 + E->lrho[i]);
        }
        for (int t = 0; t < N; ++t) Wc[t] = cexp(mu * Lc[t]);
        for (int t = 0; t < N; ++t) Pc[t] = cexp(ph[t]);
        for (int qi = 0; qi < nq; ++qi) {
            double Ic = 0.0, Ip = 0.0;
            for (int i = 0; i < nn; ++i) {
                int t = qi * nn + i;
                double complex W = Wc[t];
                double wr = creal(W), wi = cimag(W);
                if (wr * wr + wi * wi < 1e-300) { ok = 0; break; }
                double complex P = Pc[t];
                double complex invW = 1.0 / W;
                double complex invz = 1.0 / zc[t];
                double complex Ep = (E->g_elgp * P) * W;
                double complex Em = (E->g_elgm * P) * invW;
                double complex Eq = (E->g_elgq * P) * ((0.5 * zc[t]) * invW);
                double complex Er = (E->g_elgr * P) * (W * (2.0 * invz));
                double complex SPc = Sp[t].r + I * Sp[t].i;
                double complex SMc = Sm[t].r + I * Sm[t].i;
                double complex SQc = Sq[t].r + I * Sq[t].i;
                double complex SRc = Sr[t].r + I * Sr[t].i;
                double complex Umu = (PI / (2.0 * E->g_sinmu)) * (Em * SMc - Ep * SPc);
                double complex U1m = (PI / (2.0 * E->g_sin1)) * (Er * SRc - Eq * SQc);
                double complex U0 = K_order(lam, mu, invz, Umu, U1m);
                Ic += E->wo[i] * cimag(U0);
                if (do_es) {
                    double complex eh = zc[t] * E->psi_inv[i];
                    double complex U1 = K_order(lam + 1.0, mu, invz, Umu, U1m);
                    double acc = cimag(U0 * E->a2p[i]) + cimag(eh * U1 * E->lrp[i]);
                    if (E->need_a1) {
                        double complex U2 = K_order(lam + 2.0, mu, invz, Umu, U1m);
                        acc += cimag((eh * eh) * U2 * E->a1p[i]);
                    }
                    Ip += E->wo[i] * acc;
                }
            }
            if (!ok) break;
            write_tail(E, Ic, Ip, do_es, ccdf ? ccdf + qi : NULL, es ? es + qi : NULL);
        }
    }
    return ok;
}

static int nig_many(const es4mgh *E, int nq, const double *x,
                    double *ccdf, double *es) {
    const int nn = E->nnode;
    if (nn <= 0 || nq <= 0) return 1;
    if ((long)nq * nn > FAST_MAX) {
        int tile = FAST_MAX / nn;
        if (tile < 1) return 0;
        for (int i = 0; i < nq; i += tile) {
            int m = nq - i;
            if (m > tile) m = tile;
            if (!nig_many(E, m, x + i, ccdf ? ccdf + i : NULL, es ? es + i : NULL))
                return 0;
        }
        return 1;
    }
    const int N = nq * nn;
    const int do_es = es != NULL;
    size_t bytes = (size_t)N * 5 * sizeof(double complex) + 128;
    unsigned char *block = (unsigned char *)arena_acquire(bytes);
    if (!block) return 0;
    size_t off = 0;
#define TAKE(T, name) T *name = (T *)(block + off); off = (off + (size_t)N * sizeof(T) + 15u) & ~(size_t)15u
    TAKE(double complex, chi);
    TAKE(double complex, zc);
    TAKE(double complex, lz);
    TAKE(double complex, lc);
    TAKE(double complex, Ec);
#undef TAKE
    for (int qi = 0; qi < nq; ++qi) {
        double q = x[qi] - E->kk;
        double twoq = 2.0 * q;
        for (int i = 0; i < nn; ++i)
            chi[qi * nn + i] = E->chi_base[i] + I * (E->u[i] * twoq);
    }
    for (int t = 0; t < N; ++t) {
        int i = t % nn;
        zc[t] = csqrt(chi[t] * E->psi_node[i]);
    }
    for (int t = 0; t < N; ++t) lz[t] = clog(zc[t]);
    for (int t = 0; t < N; ++t) lc[t] = clog(chi[t]);
    const double log_half_pi = log(0.5 * PI);
    for (int t = 0; t < N; ++t) {
        int i = t % nn;
        double complex lrat = lc[t] - E->log_psi[i];
        double complex logK = -zc[t] + 0.5 * (log_half_pi - lz[t]);
        lz[t] = LOG2 - 0.25 * lrat + logK - E->LK2 + E->lrho[i];
    }
    for (int t = 0; t < N; ++t) Ec[t] = cexp(lz[t]);
    for (int qi = 0; qi < nq; ++qi) {
        double Ic = 0.0, Ip = 0.0;
        for (int i = 0; i < nn; ++i) {
            int t = qi * nn + i;
            double complex e0 = Ec[t];
            Ic += E->wo[i] * cimag(e0);
            if (do_es) {
                double complex eh = zc[t] * E->psi_inv[i];
                double acc = cimag(e0 * E->a2p[i]) + cimag((e0 * eh) * E->lrp[i]);
                if (E->need_a1) {
                    double complex e2 = (e0 * (chi[t] * E->psi_inv[i])) * (1.0 + 1.0 / zc[t]);
                    acc += cimag(e2 * E->a1p[i]);
                }
                Ip += E->wo[i] * acc;
            }
        }
        write_tail(E, Ic, Ip, do_es, ccdf ? ccdf + qi : NULL, es ? es + qi : NULL);
    }
    return 1;
}

static int psi_many(const es4mgh *E, int nq, const double *x,
                    double *ccdf, double *es) {
    const int nn = E->nnode;
    if (nn <= 0 || nq <= 0) return 1;
    if (!isfinite(E->psi_c1) || (E->need_a1 && !isfinite(E->psi_c2))) return 0;
    if ((long)nq * nn > FAST_MAX) {
        int tile = FAST_MAX / nn;
        if (tile < 1) return 0;
        for (int i = 0; i < nq; i += tile) {
            int m = nq - i;
            if (m > tile) m = tile;
            if (!psi_many(E, m, x + i, ccdf ? ccdf + i : NULL, es ? es + i : NULL))
                return 0;
        }
        return 1;
    }
    const int N = nq * nn;
    const int do_es = es != NULL;
    const double lam = E->lam;
    const double c0 = lgamma(-lam) - lam * LOG2 - E->LK2;
    size_t bytes = (size_t)N * 3 * sizeof(double complex) + 64;
    unsigned char *block = (unsigned char *)arena_acquire(bytes);
    if (!block) return 0;
    size_t off = 0;
#define TAKE(T, name) T *name = (T *)(block + off); off = (off + (size_t)N * sizeof(T) + 15u) & ~(size_t)15u
    TAKE(double complex, chi);
    TAKE(double complex, lm0);
    TAKE(double complex, Ec);
#undef TAKE
    for (int qi = 0; qi < nq; ++qi) {
        double q = x[qi] - E->kk;
        for (int i = 0; i < nn; ++i)
            chi[qi * nn + i] = E->chi_base[i] + I * (2.0 * E->u[i] * q);
    }
    for (int t = 0; t < N; ++t) {
        int i = t % nn;
        lm0[t] = lam * clog(chi[t]) + c0 + E->lrho[i];
    }
    for (int t = 0; t < N; ++t) Ec[t] = cexp(lm0[t]);
    for (int qi = 0; qi < nq; ++qi) {
        double Ic = 0.0, Ip = 0.0;
        for (int i = 0; i < nn; ++i) {
            int t = qi * nn + i;
            double complex e0 = Ec[t];
            Ic += E->wo[i] * cimag(e0);
            if (do_es) {
                double complex e1 = (e0 * chi[t]) * E->psi_c1;
                double acc = cimag(e0 * E->a2p[i]) + cimag(e1 * E->lrp[i]);
                if (E->need_a1) {
                    double complex e2 = (e0 * chi[t] * chi[t]) * E->psi_c2;
                    acc += cimag(e2 * E->a1p[i]);
                }
                Ip += E->wo[i] * acc;
            }
        }
        write_tail(E, Ic, Ip, do_es, ccdf ? ccdf + qi : NULL, es ? es + qi : NULL);
    }
    return 1;
}

static int fast_batch(const es4mgh *E, int n, const double *x,
                      double *ccdf, double *es) {
    if (use_slow()) return 0;
    if (E->kind == KIND_NIG) return nig_many(E, n, x, ccdf, es);
    if (E->kind == KIND_PSI0) return psi_many(E, n, x, ccdf, es);
    if (E->g_fast) return gfast_many(E, n, x, ccdf, es);
    return 0;
}

static int cheb_start(int kind) {
    /* First degree tried. The normal-inverse-Gaussian tail has settled by 20. */
    if (kind == KIND_NIG) return 20;
    return 48;
}

static int pair_settled(double a, double b) {
    int fa = isfinite(a);
    int fb = isfinite(b);
    if (fa && fb) {
        double m = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
        return fabs(a - b) <= 1e-7 * m;
    }
    if (!fa && !fb) {
        if (isnan(a) && isnan(b)) return 1;
        if (isinf(a) && isinf(b) && signbit(a) == signbit(b)) return 1;
    }
    return 0;
}

static int vecs_settled(int n, const double *a, const double *b) {
    if (!a && !b) return 1;
    if (!a || !b) return 0;
    for (int i = 0; i < n; ++i)
        if (!pair_settled(a[i], b[i])) return 0;
    return 1;
}

static void copy_out(int n, double *dst, const double *src) {
    if (dst && src) memcpy(dst, src, (size_t)n * sizeof(double));
}

static void cheb_coeffs(int m, const double *yq, double *a) {
    for (int k = 0; k < m; ++k) {
        double s = 0.0;
        for (int j = 0; j < m; ++j)
            s += yq[j] * cos(PI * k * (j + 0.5) / (double)m);
        a[k] = (k == 0) ? s / (double)m : 2.0 * s / (double)m;
    }
}

static double clenshaw(int m, const double *a, double t) {
    double u2 = 0.0, u1 = 0.0;
    for (int k = m - 1; k >= 1; --k) {
        double u = 2.0 * t * u1 - u2 + a[k];
        u2 = u1;
        u1 = u;
    }
    return t * u1 - u2 + a[0];
}

static void interp_on(int m, double mid, double half, const double *yq,
                      int n, const double *x, double *y) {
    double a[96];
    if (m < 1 || m > 96) return;
    cheb_coeffs(m, yq, a);
    double invh = half == 0.0 ? 0.0 : 1.0 / half;
    for (int i = 0; i < n; ++i)
        y[i] = clenshaw(m, a, (x[i] - mid) * invh);
}

static int series_settled(int m, const double *ac, const double *ae, const double *esq) {
    if (ac && fabs(ac[m - 1]) > 1e-9) return 0;
    if (ae && esq) {
        double scale = 1.0;
        for (int i = 0; i < m; ++i) {
            double av = fabs(esq[i]);
            if (av > scale) scale = av;
        }
        if (fabs(ae[m - 1]) > 1e-9 * scale) return 0;
    }
    return 1;
}

static void eval_prepared(const es4mgh *E, int n, const double *x,
                          double *ccdf, double *es, int nthreads) {
    if (fast_batch(E, n, x, ccdf, es)) return;
    if (nthreads <= 0) nthreads = host_threads();
    if (nthreads > n) nthreads = n;
    if (nthreads <= 1 || n < 4) {
        Chunk c = { E, x, ccdf, es, E->kk, 0, n };
        eval_chunk(&c);
        return;
    }
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double q = x[i] - E->kk;
        double *cc = ccdf ? ccdf + i : NULL;
        double *ee = es ? es + i : NULL;
        eval_point(E, q, cc, ee);
    }
#elif defined(__APPLE__)
    if (nthreads > 64) nthreads = 64;
    Chunk stack_chunks[64];
    Chunk *chunks = stack_chunks;
    int base = n / nthreads, rem = n % nthreads, cursor = 0;
    for (int t = 0; t < nthreads; ++t) {
        int len = base + (t < rem ? 1 : 0);
        chunks[t] = (Chunk){ E, x, ccdf, es, E->kk, cursor, cursor + len };
        cursor += len;
    }
    dispatch_apply((size_t)nthreads,
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t t) { eval_chunk(&chunks[t]); });
#else
    Chunk c = { E, x, ccdf, es, E->kk, 0, n };
    eval_chunk(&c);
    (void)nthreads;
#endif
}

/* Compare nn nodes with the next refinement. Remember the coarse order that
   agreed, and write the finer values. At 4096, write the finest table. */
static void eval_adaptive(es4mgh *E, int n, const double *x,
                          double *ccdf, double *es, int nthreads) {
    if (E->fixed_nnode > 0) {
        if (E->nnode != E->fixed_nnode) fill_nodes(E, E->fixed_nnode);
        eval_prepared(E, n, x, ccdf, es, nthreads);
        return;
    }
    int nn = E->nlo > 0 ? E->nlo : kind_nnode(E->kind);
    if (nn > NNODE_CAP) nn = NNODE_CAP;
    if (nn < 1) nn = 1;
    double *cc_c = ccdf ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *es_c = es ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *cc_f = ccdf ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *es_f = es ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    if ((ccdf && (!cc_c || !cc_f)) || (es && (!es_c || !es_f))) {
        free(cc_c); free(es_c); free(cc_f); free(es_f);
        eval_prepared(E, n, x, ccdf, es, nthreads);
        return;
    }
    if (!fill_nodes(E, nn)) {
        free(cc_c); free(es_c); free(cc_f); free(es_f);
        eval_prepared(E, n, x, ccdf, es, nthreads);
        return;
    }
    eval_prepared(E, n, x, cc_c, es_c, nthreads);
    if (nn >= NNODE_CAP) {
        copy_out(n, ccdf, cc_c);
        copy_out(n, es, es_c);
    }
    while (nn < NNODE_CAP) {
        int n2 = nn * 2;
        if (n2 > NNODE_CAP) n2 = NNODE_CAP;
        if (!fill_nodes(E, n2)) {
            copy_out(n, ccdf, cc_c);
            copy_out(n, es, es_c);
            break;
        }
        eval_prepared(E, n, x, cc_f, es_f, nthreads);
        if (vecs_settled(n, cc_c, cc_f) && vecs_settled(n, es_c, es_f)) {
            E->nlo = nn;
            copy_out(n, ccdf, cc_f);
            copy_out(n, es, es_f);
            break;
        }
        if (n2 == NNODE_CAP) {
            copy_out(n, ccdf, cc_f);
            copy_out(n, es, es_f);
            break;
        }
        {
            double *tc = cc_c; cc_c = cc_f; cc_f = tc;
            double *te = es_c; es_c = es_f; es_f = te;
        }
        nn = n2;
    }
    free(cc_c); free(es_c); free(cc_f); free(es_f);
}

void es4mgh_eval(es4mgh *E, int n, const double *x,
                 double *ccdf, double *es, int nthreads) {
    if (!E || n <= 0) return;
    if (n <= 24) {
        eval_adaptive(E, n, x, ccdf, es, nthreads);
        return;
    }
    /* A smooth tail is analytic in the threshold. Certify the node count on
       the Chebyshev abscissae, then interpolate. A sharp bend is integrated
       directly. The direct call is eval_adaptive, not es4mgh_eval. */
    double xmin = x[0], xmax = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }
    if (xmax - xmin <= 1e-14 * (1.0 + fabs(xmax))) {
        double cc = 0.0, ee = 0.0;
        eval_adaptive(E, 1, x, ccdf ? &cc : NULL, es ? &ee : NULL, 1);
        for (int i = 0; i < n; ++i) {
            if (ccdf) ccdf[i] = cc;
            if (es) es[i] = ee;
        }
        return;
    }
    double mid = 0.5 * (xmin + xmax), half = 0.5 * (xmax - xmin);
    int m = cheb_start(E->kind);
    if (m > n) m = n;
    while (1) {
        double xq[96], ccq[96], esq[96], ac[96], ae[96];
        for (int j = 0; j < m; ++j)
            xq[j] = mid + half * cos(PI * (j + 0.5) / (double)m);
        eval_adaptive(E, m, xq, ccdf ? ccq : NULL, es ? esq : NULL, nthreads);
        if (ccdf) cheb_coeffs(m, ccq, ac);
        if (es) cheb_coeffs(m, esq, ae);
        if (series_settled(m, ccdf ? ac : NULL, es ? ae : NULL, es ? esq : NULL)) {
            if (ccdf) interp_on(m, mid, half, ccq, n, x, ccdf);
            if (es) interp_on(m, mid, half, esq, n, x, es);
            return;
        }
        if (m >= n || m >= 96) {
            eval_adaptive(E, n, x, ccdf, es, nthreads);
            return;
        }
        int step = m / 2;
        if (step < 8) step = 8;
        int nxt = m + step;
        if (nxt > n) nxt = n;
        if (nxt > 96) nxt = 96;
        if (nxt <= m) {
            eval_adaptive(E, n, x, ccdf, es, nthreads);
            return;
        }
        m = nxt;
    }
}
