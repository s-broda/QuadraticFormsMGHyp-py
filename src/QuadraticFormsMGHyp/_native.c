#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <string.h>

#include "es4mgh.h"

#define CAPSULE_NAME "es4mgh"

typedef struct {
    Py_buffer view;
    int held;
} Buf;

static void capsule_free(PyObject *capsule) {
    es4mgh_free((es4mgh *)PyCapsule_GetPointer(capsule, CAPSULE_NAME));
}

/* Accept a native float64 buffer. NumPy reports the endian prefix. */
static int native_f64(const Py_buffer *v) {
    const char *f;
    if (!v->format || v->itemsize != (Py_ssize_t)sizeof(double)) return 0;
    f = v->format;
    if (f[0] == '@' || f[0] == '=')
        ++f;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    else if (f[0] == '>')
        ++f;
    else if (f[0] == '<' || f[0] == '!')
        return 0;
#else
    else if (f[0] == '<')
        ++f;
    else if (f[0] == '>' || f[0] == '!')
        return 0;
#endif
    return f[0] == 'd' && f[1] == '\0';
}

static void buf_release(Buf *b) {
    if (b->held) {
        PyBuffer_Release(&b->view);
        b->held = 0;
    }
}

static int buf_acquire(Buf *b, PyObject *obj, const char *name, int writable) {
    int flags = PyBUF_ND | PyBUF_FORMAT;
    b->held = 0;
    memset(&b->view, 0, sizeof(b->view));
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj, &b->view, flags) < 0) return -1;
    b->held = 1;
    if (!native_f64(&b->view) || !PyBuffer_IsContiguous(&b->view, 'C')) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous float64 array", name);
        return -1;
    }
    return 0;
}

static int n_items(const Py_buffer *v, Py_ssize_t *n) {
    if (v->itemsize <= 0 || v->len % v->itemsize != 0) return -1;
    *n = v->len / v->itemsize;
    return 0;
}

static int require_vec(const Py_buffer *v, Py_ssize_t expect, const char *name,
                       Py_ssize_t *n) {
    Py_ssize_t got;
    if (n_items(v, &got) < 0 || v->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be a 1-d float64 vector", name);
        return -1;
    }
    if (got > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "%s is too long", name);
        return -1;
    }
    if (expect >= 0 && got != expect) {
        PyErr_Format(PyExc_ValueError, "%s must have length %lld, got %lld",
                     name, (long long)expect, (long long)got);
        return -1;
    }
    *n = got;
    return 0;
}

static int require_mat(const Py_buffer *v, int d, const char *name) {
    Py_ssize_t got, need;
    if (d > 0 && (Py_ssize_t)d > PY_SSIZE_T_MAX / (Py_ssize_t)d) {
        PyErr_Format(PyExc_ValueError, "%s is too large", name);
        return -1;
    }
    need = (Py_ssize_t)d * (Py_ssize_t)d;
    if (n_items(v, &got) < 0 || got != need) {
        PyErr_Format(PyExc_ValueError,
                     "%s must contain %d by %d float64 entries", name, d, d);
        return -1;
    }
    if (v->ndim == 1) return 0;
    if (v->ndim == 2 && v->shape && v->shape[0] == (Py_ssize_t)d &&
        v->shape[1] == (Py_ssize_t)d)
        return 0;
    PyErr_Format(PyExc_ValueError,
                 "%s must be a flat vector of length %lld or a (%d, %d) matrix",
                 name, (long long)need, d, d);
    return -1;
}

static PyObject *empty_f64(Py_ssize_t n) {
    PyObject *np = NULL, *empty = NULL, *args = NULL, *kwargs = NULL;
    PyObject *nobj = NULL, *dtype = NULL, *arr = NULL;

    np = PyImport_ImportModule("numpy");
    if (!np) goto done;
    empty = PyObject_GetAttrString(np, "empty");
    if (!empty) goto done;
    nobj = PyLong_FromSsize_t(n);
    dtype = PyUnicode_FromString("float64");
    if (!nobj || !dtype) goto done;
    args = PyTuple_Pack(1, nobj);
    kwargs = PyDict_New();
    if (!args || !kwargs || PyDict_SetItemString(kwargs, "dtype", dtype) < 0)
        goto done;
    arr = PyObject_Call(empty, args, kwargs);
done:
    Py_XDECREF(np);
    Py_XDECREF(empty);
    Py_XDECREF(nobj);
    Py_XDECREF(dtype);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    return arr;
}

static PyObject *py_create(PyObject *self, PyObject *args) {
    double a0, lam, chi, psi;
    PyObject *oa, *oA, *oC, *omu, *ogam, *cap;
    Buf a, A, C, mu, gam;
    Py_ssize_t dlen, mulen, glen;
    int d;
    es4mgh *E;

    (void)self;
    memset(&a, 0, sizeof a);
    memset(&A, 0, sizeof A);
    memset(&C, 0, sizeof C);
    memset(&mu, 0, sizeof mu);
    memset(&gam, 0, sizeof gam);
    if (!PyArg_ParseTuple(args, "dOOOOOddd", &a0, &oa, &oA, &oC, &omu, &ogam,
                          &lam, &chi, &psi))
        return NULL;
    if (buf_acquire(&a, oa, "a", 0) < 0) goto fail;
    if (require_vec(&a.view, -1, "a", &dlen) < 0) goto fail;
    if (dlen <= 0) {
        PyErr_SetString(PyExc_ValueError, "a must be a non-empty vector");
        goto fail;
    }
    d = (int)dlen;
    if (buf_acquire(&A, oA, "A", 0) < 0) goto fail;
    if (buf_acquire(&C, oC, "C", 0) < 0) goto fail;
    if (buf_acquire(&mu, omu, "mu", 0) < 0) goto fail;
    if (buf_acquire(&gam, ogam, "gam", 0) < 0) goto fail;
    if (require_mat(&A.view, d, "A") < 0) goto fail;
    if (require_mat(&C.view, d, "C") < 0) goto fail;
    if (require_vec(&mu.view, d, "mu", &mulen) < 0) goto fail;
    if (require_vec(&gam.view, d, "gam", &glen) < 0) goto fail;

    Py_BEGIN_ALLOW_THREADS
    E = es4mgh_create(d, a0, (const double *)a.view.buf, (const double *)A.view.buf,
                      (const double *)C.view.buf, (const double *)mu.view.buf,
                      (const double *)gam.view.buf, lam, chi, psi);
    Py_END_ALLOW_THREADS
    buf_release(&a);
    buf_release(&A);
    buf_release(&C);
    buf_release(&mu);
    buf_release(&gam);
    if (!E) {
        PyErr_SetString(PyExc_RuntimeError, "es4mgh_create failed");
        return NULL;
    }
    cap = PyCapsule_New(E, CAPSULE_NAME, capsule_free);
    if (!cap) es4mgh_free(E);
    return cap;

fail:
    buf_release(&a);
    buf_release(&A);
    buf_release(&C);
    buf_release(&mu);
    buf_release(&gam);
    return NULL;
}

static PyObject *py_create_spectral(PyObject *self, PyObject *args) {
    double c, k, kk, lam, chi, psi;
    PyObject *oo, *od, *oe, *cap;
    Buf omega, dvec, evec;
    Py_ssize_t ne, nd, nvec;
    es4mgh *E;

    (void)self;
    memset(&omega, 0, sizeof omega);
    memset(&dvec, 0, sizeof dvec);
    memset(&evec, 0, sizeof evec);
    if (!PyArg_ParseTuple(args, "OOOdddddd", &oo, &od, &oe, &c, &k, &kk, &lam,
                          &chi, &psi))
        return NULL;
    if (buf_acquire(&omega, oo, "omega", 0) < 0) goto fail;
    if (require_vec(&omega.view, -1, "omega", &ne) < 0) goto fail;
    if (buf_acquire(&dvec, od, "d", 0) < 0) goto fail;
    if (buf_acquire(&evec, oe, "e", 0) < 0) goto fail;
    if (require_vec(&dvec.view, ne, "d", &nd) < 0) goto fail;
    if (require_vec(&evec.view, ne, "e", &nvec) < 0) goto fail;

    Py_BEGIN_ALLOW_THREADS
    E = es4mgh_create_spectral((int)ne, (const double *)omega.view.buf,
                               (const double *)dvec.view.buf,
                               (const double *)evec.view.buf, c, k, kk, lam, chi,
                               psi);
    Py_END_ALLOW_THREADS
    buf_release(&omega);
    buf_release(&dvec);
    buf_release(&evec);
    if (!E) {
        PyErr_SetString(PyExc_RuntimeError, "es4mgh_create_spectral failed");
        return NULL;
    }
    cap = PyCapsule_New(E, CAPSULE_NAME, capsule_free);
    if (!cap) es4mgh_free(E);
    return cap;

fail:
    buf_release(&omega);
    buf_release(&dvec);
    buf_release(&evec);
    return NULL;
}

static PyObject *py_eval(PyObject *self, PyObject *args) {
    PyObject *capsule, *ox, *cc = NULL, *es = NULL, *out;
    int threads;
    es4mgh *E;
    Buf x, cb, eb;
    Py_ssize_t n;

    (void)self;
    memset(&x, 0, sizeof x);
    memset(&cb, 0, sizeof cb);
    memset(&eb, 0, sizeof eb);
    if (!PyArg_ParseTuple(args, "OOi", &capsule, &ox, &threads)) return NULL;
    E = (es4mgh *)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (!E) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, "expected an es4mgh object");
        return NULL;
    }
    if (buf_acquire(&x, ox, "x", 0) < 0) goto fail;
    if (require_vec(&x.view, -1, "x", &n) < 0) goto fail;
    cc = empty_f64(n);
    es = empty_f64(n);
    if (!cc || !es) goto fail;
    if (buf_acquire(&cb, cc, "ccdf", 1) < 0) goto fail;
    if (buf_acquire(&eb, es, "es", 1) < 0) goto fail;

    Py_BEGIN_ALLOW_THREADS
    es4mgh_eval(E, (int)n, (const double *)x.view.buf, (double *)cb.view.buf,
                (double *)eb.view.buf, threads);
    Py_END_ALLOW_THREADS
    buf_release(&x);
    buf_release(&cb);
    buf_release(&eb);
    out = PyTuple_Pack(2, cc, es);
    Py_DECREF(cc);
    Py_DECREF(es);
    return out;

fail:
    buf_release(&x);
    buf_release(&cb);
    buf_release(&eb);
    Py_XDECREF(cc);
    Py_XDECREF(es);
    return NULL;
}

static PyMethodDef methods[] = {
    {"create", py_create, METH_VARARGS,
     "create(a0, a, A, C, mu, gam, lam, chi, psi) -> capsule"},
    {"create_spectral", py_create_spectral, METH_VARARGS,
     "create_spectral(omega, d, e, c, k, kk, lam, chi, psi) -> capsule"},
    {"eval", py_eval, METH_VARARGS,
     "eval(capsule, x, threads) -> (ccdf, es)"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "C binding for the QuadraticFormsMGHyp tail probability and expected shortfall.",
    -1,
    methods};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&moduledef); }
