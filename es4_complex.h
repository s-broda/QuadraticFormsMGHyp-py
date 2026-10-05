/* C11 complex arithmetic for clang-cl.

The Universal CRT does not provide csqrt, clog, cexp, or cabs. These are
the C99 principal values, built from the real libm. The imaginary unit
is a compiler builtin, so this header does not include complex.h.
*/

#ifndef ES4_COMPLEX_H
#define ES4_COMPLEX_H

#include <math.h>

#define complex _Complex

static inline double es4_creal(double complex z) { return __builtin_creal(z); }
static inline double es4_cimag(double complex z) { return __builtin_cimag(z); }

static inline double es4_cabs(double complex z) {
    return hypot(es4_creal(z), es4_cimag(z));
}

static inline double complex es4_cexp(double complex z) {
    double scale = exp(es4_creal(z));
    double angle = es4_cimag(z);
    return __builtin_complex(scale * cos(angle), scale * sin(angle));
}

static inline double complex es4_clog(double complex z) {
    return __builtin_complex(
        log(es4_cabs(z)), atan2(es4_cimag(z), es4_creal(z)));
}

/* Principal square root. The real part is nonnegative. */
static inline double complex es4_csqrt(double complex z) {
    double a = es4_creal(z);
    double b = es4_cimag(z);
    if (b == 0.0) {
        if (a >= 0.0)
            return __builtin_complex(sqrt(a), b);
        return __builtin_complex(0.0, copysign(sqrt(-a), b));
    }
    {
        double radius = hypot(a, b);
        double root = sqrt(0.5 * (radius + fabs(a)));
        if (a >= 0.0)
            return __builtin_complex(root, 0.5 * b / root);
        return __builtin_complex(0.5 * fabs(b) / root, copysign(root, b));
    }
}

#define creal(z) es4_creal(z)
#define cimag(z) es4_cimag(z)
#define cabs(z) es4_cabs(z)
#define cexp(z) es4_cexp(z)
#define clog(z) es4_clog(z)
#define csqrt(z) es4_csqrt(z)
#define I __builtin_complex(0.0, 1.0)

#endif
