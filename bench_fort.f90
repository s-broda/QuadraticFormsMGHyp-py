program bench_fort
    use iso_fortran_env, only: int64, real64
    use omp_lib
    implicit none
    integer(int64) :: n, ne, docdf_i
    integer :: nprob, p, i, reps, ios
    real(real64) :: lam, chi, psi, c, k, kk, docdf, t0, t1, rate
    real(real64), allocatable :: omega(:), d(:), e(:), x(:), q(:), out(:), ref(:)
    character(len=128) :: name
    integer(int64) :: count_i, count_f, count_rate

    interface
        subroutine es4mgh(i, qvec, omega, k, c, d, e, lambda, chi, psi, n, ne, docdf)
            import :: int64, real64
            integer(int64), intent(in) :: n, ne
            real(real64), intent(in) :: qvec(:), omega(:), d(:), e(:)
            real(real64), intent(in) :: k, c, lambda, chi, psi, docdf
            real(real64), intent(out) :: i(1:n)
        end subroutine
    end interface

    open(10, file="problems.txt", status="old")
    read(10, *) nprob
    print *, "fortran problems", nprob, " omp threads", omp_get_max_threads()
    do p = 1, nprob
        read(10, *) name
        read(10, *) n, ne
        read(10, *) lam, chi, psi, c, k, kk
        allocate(omega(ne), d(ne), e(ne), x(n), q(n), out(n), ref(n))
        read(10, *) omega
        read(10, *) d
        read(10, *) e
        read(10, *) x
        read(10, *) ref
        ! skip es reference
        read(10, *) (q(i), i=1,n)
        q = x - kk
        docdf = 1.0_real64
        call es4mgh(out, q, omega, k, c, d, e, lam, chi, psi, n, ne, docdf)
        reps = 5
        if (n < 32) reps = 10
        call system_clock(count_i, count_rate)
        do i = 1, reps
            call es4mgh(out, q, omega, k, c, d, e, lam, chi, psi, n, ne, docdf)
        end do
        call system_clock(count_f)
        t1 = real(count_f - count_i, real64) / real(count_rate, real64) * 1.0e3_real64 / real(reps, real64)
        write(*, "(A,A,A,I6,A,F10.3,A,ES10.3)") "  ", trim(name), " n=", n, &
            "  fortran ccdf ", t1, " ms   max|d|=", maxval(abs(out - ref))
        deallocate(omega, d, e, x, q, out, ref)
    end do
    close(10)
end program
