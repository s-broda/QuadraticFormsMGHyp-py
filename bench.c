#include "es4mgh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int read_vec(FILE *f, int n, double *v) {
    for (int i = 0; i < n; ++i)
        if (fscanf(f, "%lf", &v[i]) != 1) return 0;
    return 1;
}

static double max_abs(int n, const double *a, const double *b) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *prob_path = argc > 1 ? argv[1] : "problems.txt";
    const char *mat_path = argc > 2 ? argv[2] : "matrices.txt";
    FILE *f = fopen(prob_path, "r");
    if (!f) { perror(prob_path); return 1; }
    int nprob = 0;
    if (fscanf(f, "%d", &nprob) != 1) return 1;
    printf("spectral problems: %d\n", nprob);
    int worst_fail = 0;
    double twosls_ms = 0.0;
    int twosls_n = 0;
    for (int p = 0; p < nprob; ++p) {
        char name[128];
        if (fscanf(f, "%127s", name) != 1) return 1;
        int n, ne;
        if (fscanf(f, "%d %d", &n, &ne) != 2) return 1;
        double lam, chi, psi, c, k, kk;
        if (fscanf(f, "%lf %lf %lf %lf %lf %lf", &lam, &chi, &psi, &c, &k, &kk) != 6) return 1;
        double *omega = (double *)malloc((size_t)ne * sizeof(double));
        double *d = (double *)malloc((size_t)ne * sizeof(double));
        double *e = (double *)malloc((size_t)ne * sizeof(double));
        double *x = (double *)malloc((size_t)n * sizeof(double));
        double *cc_ref = (double *)malloc((size_t)n * sizeof(double));
        double *es_ref = (double *)malloc((size_t)n * sizeof(double));
        double *cc = (double *)malloc((size_t)n * sizeof(double));
        double *es = (double *)malloc((size_t)n * sizeof(double));
        if (!read_vec(f, ne, omega) || !read_vec(f, ne, d) || !read_vec(f, ne, e) ||
            !read_vec(f, n, x) || !read_vec(f, n, cc_ref) || !read_vec(f, n, es_ref)) {
            fprintf(stderr, "read failed at %s\n", name);
            return 1;
        }
        es4mgh *E = es4mgh_create_spectral(ne, omega, d, e, c, k, kk, lam, chi, psi);
        if (!E) { fprintf(stderr, "create failed %s\n", name); return 1; }
        int cdf_only = getenv("ES4_CDF_ONLY") != NULL;
        es4mgh_eval(E, n, x, cc, cdf_only ? NULL : es, 1);
        double dc = max_abs(n, cc, cc_ref);
        double de = cdf_only ? 0.0 : max_abs(n, es, es_ref);
        double rel = 0.0;
        if (!cdf_only) {
            for (int i = 0; i < n; ++i) {
                double s = fabs(es_ref[i]);
                if (s < 1.0) continue;
                double r = fabs(es[i] - es_ref[i]) / s;
                if (r > rel) rel = r;
            }
        }
        int reps = n >= 200 ? 5 : 20;
        /* serial */
        es4mgh_eval(E, n, x, cc, cdf_only ? NULL : es, 1);
        double t1 = now_ms();
        for (int r = 0; r < reps; ++r) es4mgh_eval(E, n, x, cc, cdf_only ? NULL : es, 1);
        t1 = (now_ms() - t1) / reps;
        /* parallel, all cpus */
        es4mgh_eval(E, n, x, cc, cdf_only ? NULL : es, 0);
        double tp = now_ms();
        for (int r = 0; r < reps; ++r) es4mgh_eval(E, n, x, cc, cdf_only ? NULL : es, 0);
        tp = (now_ms() - tp) / reps;
        printf("  %-16s n=%5d ne=%3d  |dccdf|=%.3e |des|=%.3e relES %.3e  serial %.3f ms  parallel %.3f ms\n",
               name, n, ne, dc, de, rel, t1, tp);
        if (dc > 5e-6 || de > 5e-4) {
            printf("    FAIL first cc %.8g vs %.8g   es %.8g vs %.8g\n",
                   cc[0], cc_ref[0], es[0], es_ref[0]);
            worst_fail = 1;
        }
        if (strncmp(name, "twosls_", 7) == 0) {
            twosls_ms += tp;
            twosls_n++;
        }
        es4mgh_free(E);
        free(omega); free(d); free(e); free(x); free(cc_ref); free(es_ref); free(cc); free(es);
    }
    fclose(f);
    if (twosls_n)
        printf("  twosls spectral parallel total %.3f ms  (%.3f ms/call)\n",
               twosls_ms, twosls_ms / twosls_n);

    FILE *mf = fopen(mat_path, "r");
    if (!mf) { perror(mat_path); return 1; }
    int nmat = 0;
    if (fscanf(mf, "%d", &nmat) != 1) return 1;
    printf("matrix problems: %d\n", nmat);
    for (int p = 0; p < nmat; ++p) {
        char name[128];
        if (fscanf(mf, "%127s", name) != 1) return 1;
        int d, n;
        if (fscanf(mf, "%d %d", &d, &n) != 2) return 1;
        double a0, lam, chi, psi;
        if (fscanf(mf, "%lf %lf %lf %lf", &a0, &lam, &chi, &psi) != 4) return 1;
        double *a = (double *)malloc((size_t)d * sizeof(double));
        double *mu = (double *)malloc((size_t)d * sizeof(double));
        double *gam = (double *)malloc((size_t)d * sizeof(double));
        double *A = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
        double *C = (double *)malloc((size_t)d * (size_t)d * sizeof(double));
        double *x = (double *)malloc((size_t)n * sizeof(double));
        double *cc_ref = (double *)malloc((size_t)n * sizeof(double));
        double *es_ref = (double *)malloc((size_t)n * sizeof(double));
        double *cc = (double *)malloc((size_t)n * sizeof(double));
        double *es = (double *)malloc((size_t)n * sizeof(double));
        if (!read_vec(mf, d, a) || !read_vec(mf, d, mu) || !read_vec(mf, d, gam) ||
            !read_vec(mf, d * d, A) || !read_vec(mf, d * d, C) ||
            !read_vec(mf, n, x) || !read_vec(mf, n, cc_ref) || !read_vec(mf, n, es_ref)) {
            fprintf(stderr, "matrix read failed %s\n", name);
            return 1;
        }
        double t0 = now_ms();
        es4mgh *E = es4mgh_create(d, a0, a, A, C, mu, gam, lam, chi, psi);
        double t_create = now_ms() - t0;
        if (!E) { fprintf(stderr, "matrix create failed %s\n", name); return 1; }
        es4mgh_eval(E, n, x, cc, es, 0);
        double dc = max_abs(n, cc, cc_ref);
        double de = max_abs(n, es, es_ref);
        int reps = n >= 200 ? 5 : 10;
        double t1 = now_ms();
        for (int r = 0; r < reps; ++r) {
            es4mgh_free(E);
            E = es4mgh_create(d, a0, a, A, C, mu, gam, lam, chi, psi);
            es4mgh_eval(E, n, x, cc, es, 0);
        }
        double tp = (now_ms() - t1) / reps;
        printf("  matrix %-16s d=%3d n=%5d  |dccdf|=%.3e |des|=%.3e  create %.3f ms  create+eval %.3f ms\n",
               name, d, n, dc, de, t_create, tp);
        if (dc > 5e-5 || de > 5e-3) {
            printf("    FAIL cc0 %.8g vs %.8g  es0 %.8g vs %.8g\n", cc[0], cc_ref[0], es[0], es_ref[0]);
            worst_fail = 1;
        }
        es4mgh_free(E);
        free(a); free(mu); free(gam); free(A); free(C); free(x); free(cc_ref); free(es_ref); free(cc); free(es);
    }
    fclose(mf);
    return worst_fail;
}
