module pso_module
  implicit None
  integer :: termination_f_no_improve = -1 !if termination_f_no_improve > 0, DE is stopped if there is no improvement over the last termination_f_no_improve generation/itterations.
  real(8), allocatable, dimension(:) :: PSOrun_xMin, PSOrun_fvalHist, temp_buffer

contains
  subroutine PSOrun(tp_id, d, N, w_i, w_f, c1, c2, evals, random_seed)
    ! PSO routine only for the test functions
    ! parameters :
    !   tp_id - specify test problem to optimize, see test_problems_SO for further details.
    !   d - number of test problem dimensions
    !   N - number of particles
    !   w_i, w_f - initial inertia, final inertia
    !   c1,c2 - personal and global beliefs
    !   lb,ub - upper and lowwer boundary for initial population
    !   evals - number of function evalutions.
    !   random_seed - the pseudo random number generators seed
    use cec2005problems
    use random
    integer :: tp_id, d, N, evals, random_seed
    real(8) :: w_i, w_f, w, c1, c2, lb, ub
    integer :: k, i, j, It
    real(8) :: f_val
    real(8), dimension(N) :: f_pb
    real(8), dimension(d) :: x_gb, x_prev
    real(8), dimension(d,N) :: x, v, x_pb
    call set_random_seed( random_seed )
    cec_tp = tp_id !setting test problem.
    lb = cec2005fun_lowerbound()
    ub = cec2005fun_upperbound()
    do i = 1,N ! generating intial population
       x(:,i) = lb + rand_vec(d)*(ub - lb)
       f_pb(i) = cec2005fun(x(:,i),d)
    end do
    x_pb = x
    x_gb = x(:,minval(minloc(f_pb)))
    v = 0.0
    it = evals/N !number of itterations (including populating phase), this rounds down by default
    !print *, 'itterations :',it
    if (allocated(PSOrun_fvalHist)) deallocate(PSOrun_fvalHist)
    allocate(PSOrun_fvalHist(It))
    PSOrun_fvalHist(1) = minval(f_pb) 
    do k = 2,it
       w = w_i + (k-2.0)/(it-2.0)*(w_f - w_i)
       do i = 1,N
          v(:,i) = w*v(:,i) + c1*rand_vec(d)*(x_pb(:,i)-x(:,i)) + c2*rand_vec(d)*(x_gb-x(:,i))
          x_prev = x(:,i)
          x(:,i) = x(:,i) + v(:,i)
          if (cec2005fun_bound_constrained()) then
             do j = 1,d
                if (x(j,i) < lb) x(j,i) = (lb + x_prev(j))/2
                if (x(j,i) > ub) x(j,i) = (ub + x_prev(j))/2
             end do
          end if

          f_val = cec2005fun(x(:,i),d)
          !if (i == 1) print *,'x_gb= ', x_gb , ' x(:,1) = ' , x(:,i)
          if (f_val < f_pb(i)) then
             f_pb(i) = f_val
             x_pb(:,i) = x(:,i)
          end if
       end do
       x_gb = x_pb(:,minval(minloc(f_pb)))
       PSOrun_fvalHist(k) = minval(f_pb)
       !print *,'fmin(it=',k,') :',PSOrun_fvalHist(k)
       !termination criterea checks
       if (( termination_f_no_improve > 0 ) .and. ( k >= termination_f_no_improve ) ) then
          if ( sum(PSOrun_fvalHist(k-termination_f_no_improve:k)) == PSOrun_fvalHist(k)*(termination_f_no_improve+1) ) then !terminate
             if (allocated(temp_buffer)) deallocate(temp_buffer)
             allocate(temp_buffer(k))
             temp_buffer = PSOrun_fvalHist(1:k)
             deallocate(PSOrun_fvalHist)
             allocate(PSOrun_fvalHist(k))
             PSOrun_fvalHist = temp_buffer
             exit !do loop
          end if
       end if
    end do
    if (allocated(PSOrun_xMin)) deallocate(PSOrun_xMin)
    allocate(PSOrun_xMin(d))
    PSOrun_xMin = x_gb
  end subroutine PSOrun
end module pso_module
