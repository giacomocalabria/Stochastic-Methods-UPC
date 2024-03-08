!     Fortran code implementing a Monte Carlo program to compute the expectation values of two given functions:

!     The random variable x with uniform distribution probability is generated in the 0<= x <= 1 domain.

!     the expectation value of two functions is calculated:

!     a)  f(x)  =  Exp(-x)

!     b)  f(x) = x   if  x^2 > 1/4        ,     f(x) = 1/2 otherwise



program monte_carlo
    implicit none
    integer, parameter :: n_samples = 1000000
    real :: sum_exp, sum_condition, x, expectation_exp, expectation_condition
    integer :: i
    character(len=100) :: filename

    ! Initialize sums
    sum_exp = 0.0
    sum_condition = 0.0

    ! Open file for writing results
    open(unit=10, file='monte_carlo_results.txt', status='unknown')

    ! Generate random numbers and compute sum for each function
    do i = 1, n_samples
        call random_number(x)
        
        ! Function f(x) = Exp(-x)
        sum_exp = sum_exp + exp(-x)
        
        ! Function f(x) = x if x^2 > 1/4, f(x) = 1/2 otherwise
        if (x**2 > 0.25) then
            sum_condition = sum_condition + x
        else
            sum_condition = sum_condition + 0.5
        endif
    end do

    ! Compute expectation values
    expectation_exp = sum_exp / real(n_samples)
    expectation_condition = sum_condition / real(n_samples)

    ! Write results to file
    write(10, *) 'Expectation value of Exp(-x): ', expectation_exp
    write(10, *) 'Expectation value of x if x^2 > 1/4, 1/2 otherwise: ', expectation_condition

    ! Close file
    close(10)

end program monte_carlo

