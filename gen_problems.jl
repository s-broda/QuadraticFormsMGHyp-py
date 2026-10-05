# Build spectral problems, correct-formula references, and Julia timings.
using QuadraticFormsMGHyp
using LinearAlgebra
using Printf
using Random

const RP = 0.318309886183790671537767526745

function spectral_of(a0, a, A, C, mu, gam)
    As = (A + A') / 2
    CAC = Symmetric(C' * As * C)
    F = eigen(CAC)
    omega = F.values
    P = F.vectors
    muA = vec(As' * mu)
    gA = vec(As' * gam)
    CP = C * P
    c = dot(a, gam) + 2 * dot(muA, gam)
    d = vec(CP' * a + 2 * CP' * muA)
    e = vec(2 * CP' * gA)
    k = dot(gA, gam)
    kk = a0 + dot(a, mu) + dot(muA, mu)
    return omega, d, e, c, k, kk
end

function fortran_ub(omega; epsabs=1e-10)
    o = sort(abs.(filter(!=(0.0), omega)), rev=true)
    ub = 1.0
    logsum = 0.0
    for i in eachindex(o)
        logsum += log(2 * o[i])
        ubnew = (-1 / i) * (2log(π) + 2log(i) + 2log(epsabs) - log(4) + logsum)
        ubnew = exp(ubnew)
        ubnew = sqrt(ubnew) / (1 + sqrt(ubnew))
        ub = min(ub, ubnew)
    end
    return clamp(ub, 1e-3, 0.999999)
end

function gl_nodes(n, a, b)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    T = SymTridiagonal(zeros(n), β)
    F = eigen(T)
    x = F.values
    w = 2 * F.vectors[1, :] .^ 2
    mid = (a + b) / 2
    half = (b - a) / 2
    return mid .+ half .* x, half .* w
end

function map_nodes(n, ub)
    v, wv = gl_nodes(n, 0.0, ub)
    us = similar(v); ws = similar(wv)
    for i in eachindex(v)
        ss = v[i] / (1 - v[i])
        us[i] = ss * ss
        ws[i] = wv[i] * (2 * ss / (1 - v[i])^2)
    end
    return us, ws
end

function refs(omega, d, e, c, k, kk, lam, chi, psi, x; nnode=96)
    de = d .* e; d2 = d .* d; e2 = e .* e
    LK2 = QuadraticFormsMGHyp.lklam(lam, chi, psi)
    ub = fortran_ub(omega)
    us, ws = map_nodes(nnode, ub)
    # precompute q-independent pieces
    pre = Vector{NTuple{7,ComplexF64}}(undef, length(us))
    for (ii, u) in pairs(us)
        s = 1im * u
        t1=0im; t2=0im; t3=0im; t4=0im; a2p=0im; a1p=0im; lrp=0im
        @inbounds for j in eachindex(omega)
            om = omega[j]
            nu = 1 / (1 - 2 * om * s)
            nu2 = nu * nu
            t1 += d2[j] * nu
            t2 += e2[j] * nu
            t3 += de[j] * nu
            t4 += log(nu)
            a2p += s * d2[j] * nu + s^2 * d2[j] * om * nu2
            a1p += s * e2[j] * nu + s^2 * e2[j] * om * nu2
            lrp += 2s * de[j] * nu + 2s^2 * de[j] * om * nu2 + om * nu
        end
        a1p += k
        lrp += c
        chi_b = chi - s^2 * t1
        psi_n = psi - 2 * (k * s + 0.5 * s^2 * t2)
        lrho = s * c + s^2 * t3 + 0.5 * t4
        pre[ii] = (chi_b, psi_n, lrho, a2p, a1p, lrp, u)
    end
    function lk(order, chi_a, psi_a)
        try
            QuadraticFormsMGHyp.lklam(order, chi_a, psi_a)
        catch
            -Inf + 0im
        end
    end
    lm1_0 = lk(lam + 1, chi, psi) - LK2
    lm2_0 = lk(lam + 2, chi, psi) - LK2
    M20 = real(exp(lm2_0) * k + exp(lm1_0) * (c + sum(omega)))
    ccdf = similar(x); es = similar(x)
    for (iq, xv) in pairs(x)
        q = xv - kk
        Ic = 0.0; Ip = 0.0
        for (ii, w) in pairs(ws)
            chi_b, psi_n, lrho, a2p, a1p, lrp, u = pre[ii]
            chi_a = chi_b + 2im * u * q
            base = -LK2 + lrho
            lm0 = lk(lam, chi_a, psi_n) + base
            lm1 = lk(lam + 1, chi_a, psi_n) + base
            lm2 = lk(lam + 2, chi_a, psi_n) + base
            function ix(z)
                r = real(z)
                r < -700 && return 0.0
                return exp(r) * sin(imag(z))
            end
            function ixm(z, coef)
                r = real(z)
                r < -700 && return 0.0
                ee = exp(r); s = sin(imag(z)); co = cos(imag(z))
                return ee * (s * real(coef) + co * imag(coef))
            end
            Ic += (w / u) * ix(lm0)
            Ip += (w / u) * (ixm(lm0, a2p) + ixm(lm2, a1p) + ixm(lm1, lrp))
        end
        cc = 0.5 + RP * Ic
        ccdf[iq] = cc
        es[iq] = (0.5 * M20 + RP * Ip) / cc + kk
    end
    return ccdf, es
end

function write_vec(io, v)
    for x in v
        @printf(io, "%.17g\n", x)
    end
end

problems = []

function add_problem!(name, x, a0, a, A, C, mu, gam, lam, chi, psi; with_matrix=false)
    omega, d, e, c, k, kk = spectral_of(a0, a, A, C, mu, gam)
    # shrink reference work: for long vectors, references on all points
    ccdf, es = refs(omega, d, e, c, k, kk, lam, chi, psi, x)
    push!(problems, (; name, x, a0, a, A, C, mu, gam, lam, chi, psi, omega, d, e, c, k, kk, ccdf, es, with_matrix))
    println("built ", name, "  n=", length(x), " ne=", length(omega))
end

# Portfolio NIG, same shape as the paper's expected-shortfall example.
Random.seed!(1)
m = 10
delta = randn(m)
gamma = abs.(randn(m)) .+ 0.1
number = -ones(m)
a = -delta .* number
A = diagm(0 => -0.5 .* gamma .* number)
sigma = fill(0.3, m); S0 = fill(100.0, m); h = 1/252
R = 0.5 * ones(m, m) + 0.5 * I
sigs = diagm(0 => sigma .* S0 .* sqrt(h))
lam, chi, psi = -0.5, 1.0, 1.0
V = exp(QuadraticFormsMGHyp.lklam(lam+1, chi, psi) - QuadraticFormsMGHyp.lklam(lam, chi, psi))
Sigma = sigs * R * sigs / V
C = Matrix(sqrt(Hermitian((Sigma + Sigma') / 2)))
x = collect(3.5:0.01:17.5)
add_problem!("portfolio_nig", x, 0.01, a, A, C, zeros(m), zeros(m), lam, chi, psi; with_matrix=true)

# General GH, nonzero gamma, non half-integer order.
Random.seed!(2)
d = 16
A2 = randn(d, d); A2 = (A2 + A2') / 2
M = randn(d, d); M = M * M' + I(d)
C2 = Matrix(cholesky(Hermitian((M + M') / 2)).L)
a2 = randn(d); mu2 = 0.1 .* randn(d); gam2 = 0.2 .* randn(d)
x2 = collect(range(-1.0, 6.0, length=64))
add_problem!("general_gh", x2, 0.1, a2, A2, C2, mu2, gam2, -1.3, 1.5, 0.8; with_matrix=true)

# Student's t, rank-4 quadratic form.
Random.seed!(3)
d3 = 40
B = randn(d3, 4)
A3 = Matrix(Hermitian(B * B'))
C3 = Matrix(I(d3) * 1.0)
a3 = randn(d3)
nu = 5.0
x3 = collect(range(-1.0, 12.0, length=64))
add_problem!("student_t", x3, 0.0, a3, A3, C3, zeros(d3), zeros(d3), -nu/2, nu, 0.0; with_matrix=true)

# 2SLS-like: one threshold, matrix changes with b. Dimension 50, t errors.
Random.seed!(4)
n = 25
Z = randn(n, 1)
Pz = Z * ((Z' * Z) \ Z')
S = randn(2n, 2n); S = S * S' + I(2n)
S = Matrix(cholesky(Hermitian((S + S') / 2)).L)
a4 = randn(2n)
for i in 1:20
    bb = 0.05 * i
    AA = 0.5 * [zeros(n, n) Pz; Pz -2bb * Pz]
    add_problem!("twosls_$i", [0.0], -bb, a4, AA, S, zeros(2n), zeros(2n), -2.5, 5.0, 0.0;
                 with_matrix=(i == 1))
end

open("/Users/broda/es4mgh-c/problems.txt", "w") do io
    println(io, length(problems))
    for p in problems
        println(io, p.name)
        @printf(io, "%d %d\n", length(p.x), length(p.omega))
        @printf(io, "%.17g %.17g %.17g %.17g %.17g %.17g\n", p.lam, p.chi, p.psi, p.c, p.k, p.kk)
        write_vec(io, p.omega)
        write_vec(io, p.d)
        write_vec(io, p.e)
        write_vec(io, p.x)
        write_vec(io, p.ccdf)
        write_vec(io, p.es)
    end
end

open("/Users/broda/es4mgh-c/matrices.txt", "w") do io
    mats = filter(p -> p.with_matrix, problems)
    println(io, length(mats))
    for p in mats
        d = length(p.a)
        println(io, p.name)
        @printf(io, "%d %d\n", d, length(p.x))
        @printf(io, "%.17g %.17g %.17g %.17g\n", p.a0, p.lam, p.chi, p.psi)
        write_vec(io, p.a)
        write_vec(io, p.mu)
        write_vec(io, p.gam)
        for col in 1:d, row in 1:d
            # store row-major: index (row, col)
        end
        for row in 1:d, col in 1:d
            @printf(io, "%.17g\n", p.A[row, col])
        end
        for row in 1:d, col in 1:d
            @printf(io, "%.17g\n", p.C[row, col])
        end
        write_vec(io, p.x)
        write_vec(io, p.ccdf)
        write_vec(io, p.es)
    end
end

println("files written")

# Timings of the Julia package on the same problems.
function time_qfmgh(p, reps)
    args = (p.x, p.a0, p.a, p.A, p.C, p.mu, p.gam, p.lam, p.chi, p.psi)
    qfmgh(args...)
    GC.gc()
    t = @elapsed for _ in 1:reps
        qfmgh(args...)
    end
    return 1e3 * t / reps
end

println("\nJulia qfmgh wall time (threads=", Threads.nthreads(), ")")
for p in problems
    if startswith(p.name, "twosls_") && p.name != "twosls_1"
        continue
    end
    reps = length(p.x) >= 200 ? 3 : 5
    ms = time_qfmgh(p, reps)
    # cross-check package vs correct reference
    cc, pm = qfmgh(p.x, p.a0, p.a, p.A, p.C, p.mu, p.gam, p.lam, p.chi, p.psi)
    dc = maximum(abs.(cc .- p.ccdf))
    de = maximum(abs.(pm .- p.es))
    @printf("  %-16s n=%5d  %8.3f ms   |d ccdf|=%.3e  |d es vs correct|=%.3e\n",
            p.name, length(p.x), ms, dc, de)
end

# 2SLS loop timing
sls = filter(p -> startswith(p.name, "twosls_"), problems)
qfmgh(sls[1].x, sls[1].a0, sls[1].a, sls[1].A, sls[1].C, sls[1].mu, sls[1].gam, sls[1].lam, sls[1].chi, sls[1].psi)
t = @elapsed for _ in 1:3
    for p in sls
        qfmgh(p.x, p.a0, p.a, p.A, p.C, p.mu, p.gam, p.lam, p.chi, p.psi)
    end
end
@printf("  %-16s calls=%d  %8.3f ms/call\n", "twosls_loop", length(sls), 1e3 * t / 3 / length(sls))
