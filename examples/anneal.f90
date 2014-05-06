! written by Antoine Dymond, Jan 2012, based upon the Scipy.optimize.anneal code, the scipy license agreement is as follows:
!!$
!!$Copyright (c) 2001, 2002 Enthought, Inc.
!!$All rights reserved.
!!$
!!$Copyright (c) 2003-2009 SciPy Developers.
!!$All rights reserved.
!!$
!!$Redistribution and use in source and binary forms, with or without
!!$modification, are permitted provided that the following conditions are met:
!!$
!!$  a. Redistributions of source code must retain the above copyright notice,
!!$     this list of conditions and the following disclaimer.
!!$  b. Redistributions in binary form must reproduce the above copyright
!!$     notice, this list of conditions and the following disclaimer in the
!!$     documentation and/or other materials provided with the distribution.
!!$  c. Neither the name of the Enthought nor the names of its contributors
!!$     may be used to endorse or promote products derived from this software
!!$     without specific prior written permission.
!!$
!!$
!!$THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!!$AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!!$IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!!$ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
!!$ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
!!$DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
!!$SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
!!$CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
!!$LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
!!$OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
!!$DAMAGE.
!
! compile using
! $ f2py -c -m anneal_fortran anneal.f90


module anneal_module
  ! to compile using f2py, leave real type as default, needs to be compadiable between numpy and fortran.
  implicit None  ! fortran is case insensitive.
  !f2py forces all variable to lower case in linking module
  integer :: problem_id, objFun_evals
  real(8), allocatable, dimension(:) :: fval_hist, x_min
  integer, allocatable, dimension(:) :: eval_hist

contains

  function objFun(x, D)
    integer :: D, i
    real(8) :: x(D), objFun
    objFun = 0
    objFun_evals = objFun_evals + 1
    select case(problem_id)
    case (1) !general Rossenbrock
       do i = 1,D-1
          objFun = objFun + 100.0*(x(i+1) - x(i)**2.0)**2.0 + (1.0 - x(i))**2.0
       end do
    case (2) !sphere
       objFun = sum(x**2)
    end select
  end function objFun

  SUBROUTINE set_random_seed(seed_offset)
    !taken from the gfortran site. 
    !http://gcc.gnu.org/onlinedocs/gfortran/RANDOM_005fSEED.html#RANDOM_005fSEED
    INTEGER :: i, n, seed_offset
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed   
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = seed_offset + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)
  END SUBROUTINE set_random_seed


  function rand_uniform()
    real(8) :: rand_uniform
    call random_number(rand_uniform)
  end function rand_uniform

  subroutine fast_sa_run(prob_id, x0, D, T0, dwell, m, n, quench, boltzmann, maxEvals, lower, upper, random_seed)
    integer :: prob_id, D, dwell, maxEvals, random_seed
    real(8) :: T0, m, n, quench, boltzmann
    real(8), dimension(D) :: x0, u, lower, upper 
    real(8) :: T, c, dE, y_j
    integer :: k, kMax, acceptTest, i, j
    real(8) :: current_state_fv, last_state_fv, best_state_fv
    real(8), dimension(D) ::current_state_xv, last_state_xv, best_state_xv


    call set_random_seed( random_seed )
    problem_id  = prob_id
    kMax = maxEvals/ dwell
    if (allocated(fval_hist)) deallocate(fval_hist, eval_hist )
    allocate(fval_hist(kMax), eval_hist(kMax))

    fval_hist = 0.0
    objFun_evals = 0
    T = T0
    c = m * exp(n * quench)
    last_state_xv = x0
    last_state_fv = objFun(x0, D)
    best_state_xv = x0
    best_state_fv = last_state_fv
    do k = 1,kMax
       do i = 1, dwell
          ! schedule.update_guess
          call random_number(u)
          do j = 1,D
             y_j = T * ( ( 1+1.0/T) ** abs(2*u(i)-1) - 1.0 )
             if ( (u(j) - 0.5) < 0 ) y_j = - y_j
             current_state_xv(j) = last_state_xv(j) + y_j*(upper(j) - lower(j))
             if (current_state_xv(j) < lower(j)) current_state_xv(j) = lower(j)
             if (current_state_xv(j) > upper(j)) current_state_xv(j) = upper(j)
          end do
          current_state_fv =  objFun(current_state_xv, D)
          dE = current_state_fv - last_state_fv
          !schedule.accept_test(dE)
          acceptTest = 0
          if (dE < 0) then
             acceptTest = 1
          else
             if  ( exp(-dE* 1.0 / boltzmann /T ) > rand_uniform() ) acceptTest = 1
          end if
          if ( acceptTest == 1) then
             last_state_xv = current_state_xv
             last_state_fv = current_state_fv
             if (last_state_fv < best_state_fv) then
                best_state_xv = last_state_xv
                best_state_fv = last_state_fv
             end if
          end if
       end do
       !tempreture update
       T = T0 * exp(-c*k*quench)
       !book keeping
       fval_hist(k) = best_state_fv
       eval_hist(k) = objFun_evals
    end do
    if (allocated(x_min)) deallocate(x_min)
    allocate(x_min(D))
    x_min = best_state_xv
  end subroutine fast_sa_run

end module anneal_module
