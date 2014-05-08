module de_module
  ! to compile using f2py, leave real type as default, needs to be compadiable between numpy and fortran.
  implicit None  ! fortran is case insensitive.
  !f2py forces all variable to lower case in linking module
  ! DE additional settings (x = (best,rand) , y =  (1,2,3,...) , z = (bin))
  integer :: DE_x = 0 !1=best, 0 = rand
  integer :: DE_y = 1 !number of difference vectors, must not exceed 10
  integer :: DE_z = 0 !has no effect because bin will always be used.
  integer :: termination_f_no_improve = -1 !if termination_f_no_improve > 0, DE is stopped if there is no improvement over the last termination_f_no_improve generation/itterations.
  real(8), allocatable, dimension(:) :: DErun_xMin, DErun_fvalHist, temp_buffer

contains

  subroutine DErun(tp_id, D, Np, F, CR, evals, random_seed)
    ! Differntial evolution, limited impentation,no constraints, only test funs
    ! set current_tp to change test problem
    ! inputs
    !    tp_id - specify test problem to optimize, see test_problems_SO for further details.
    !    D - number of test problem dimensions
    !    Np - size of population, guide between 5 and 10 times the number of dimensions 
    !    F -  amlification of differnetial variation, normally 0.5 or between [0,2]
    !    Cr - cross-over constant, between [0,1]. 0 minimal crossover, 1 max crossover
    !    evals - number of function evaluation that will be done.
    !   random_seed - the pseudo random number generators seed
    use cec2005problems
    use random
    integer :: tp_id, Np, evals, gMax, D, random_seed
    real(8) :: F,CR,lb,ub
    real(8), dimension(D,Np) :: x, u, v, x_next
    real(8) :: obf_x(Np),fv , vec_mutation(D)
    integer :: i, j, js, g, rnbr, i_best
    integer :: r(10*2+1)
    
    call set_random_seed( random_seed )
    !call init_random_seed() !randomize, call set_random_seed instead ...
    cec_tp = tp_id !setting test problem.
    lb = cec2005fun_lowerbound()
    ub = cec2005fun_upperbound()

    gMax = evals/Np !number of itterations (including populating phase), rounds down by default
    if (allocated(DErun_fvalHist)) deallocate(DErun_fvalHist)
    allocate(DErun_fvalHist(gMax))
    ! generating intail population
    do i = 1,Np 
       x(:,i) = (ub-lb)*rand_vec(D) + lb 
       obf_x(i) = cec2005fun(x(:,i),D) !used to store function evaluations
    end do
    DErun_fvalHist = 0.0
    DErun_fvalHist(1) = minval(obf_x)
    ! Main loop
    do g = 2,gMax
       i_best = minval(minloc(obf_x))
       do i = 1,Np
          ! Mutation(v)
          js = 2
          if (DE_x == 0) js = 1 !basis for mutation random
          if (DE_x == 1) r(1) = i_best !for mutation best value so far 
          do j = js,DE_y*2+1 ! generate unique population indexes
             do
                r(j) = rand_int(1,Np)
                if (count( [i,r(1:(j-1))]==r(j) )==0) exit
             end do
          end do
          vec_mutation = 0.0
          do j = 1,DE_y
             vec_mutation = vec_mutation + x(:,r(2*j)) - x(:,r(2*j+1))
          end do
          v(:,i) = x(:,r(1)) + F*vec_mutation
          ! Cross over (u), generating candiate vector
          u(:,i) = merge(v(:,i),x(:,i),rand_vec(D)<CR)  
          rnbr = rand_int(1,D) !one component that is forced to swap 
          u(rnbr,i) = v(rnbr,i)
          if (cec2005fun_bound_constrained()) then
             do j = 1,D
                if (u(j,i) < lb) u(j,i) = (lb + x(j,i))/2
                if (u(j,i) > ub) u(j,i) = (ub + x(j,i))/2
             end do
          end if
          ! Natural selection, using Greedy criteria
          fv = cec2005fun(u(:,i), D)
          if (fv < obf_x(i)) then !population updated
             obf_x(i) = fv
             x_next(:,i) = u(:,i)
          else
             x_next(:,i) = x(:,i)
          end if
       end do       
       DErun_fvalHist(g) = minval(obf_x)
       x = x_next 
       !termination criterea checks
       if (( termination_f_no_improve > 0 ) .and. ( g >= termination_f_no_improve ) ) then
          if ( sum(DErun_fvalHist(g-termination_f_no_improve:g)) == DErun_fvalHist(g)*(termination_f_no_improve+1) ) then !terminate
             if (allocated(temp_buffer)) deallocate(temp_buffer)
             allocate(temp_buffer(g))
             temp_buffer = DErun_fvalHist(1:g)
             deallocate(DErun_fvalHist)
             allocate(DErun_fvalHist(g))
             DErun_fvalHist = temp_buffer
             exit !do loop
          end if
       end if
    end do
    if (allocated(DErun_xMin)) deallocate(DErun_xMin)
    allocate(DErun_xMin(D))
    DErun_xMin = x(:,minval(minloc(obf_x))) !minval nessary as minloc returns vector
  end subroutine DErun

end module de_module
