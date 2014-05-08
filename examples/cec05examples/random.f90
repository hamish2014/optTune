module random
  implicit None
  real(8) :: pi = asin(1.0_8)*2
  
contains

  SUBROUTINE init_random_seed() 
    ! randomize based on system time taken from from the gfortran site. 
    ! http://gcc.gnu.org/onlinedocs/gfortran/RANDOM_005fSEED.html#RANDOM_005fSEED
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed   
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    CALL SYSTEM_CLOCK(COUNT=clock)
    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)
  END SUBROUTINE init_random_seed

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

  function rand_int(lo,up)
    integer :: rand_int,lo,up
    real(8) :: rn
    call random_number(rn)
    rand_int = floor(rn*(up-lo)+0.5)+lo
  end function rand_int

  function rand_vec(n)
    ! returns a random vector of len n, with each element ranging from [0,1).
    integer :: n
    real(8) :: rand_vec(n)
    call random_number(rand_vec)
  end function rand_vec

  function randc(x0, gamma)
    ! inverse transform sampling used for cauchy distribution
    real(8) :: randc, x0, gamma
    randc = tan((rand_uniform()-0.5)*pi)*gamma + x0
  end function randc

  function randn(mean, std)
    !Box-muller transfor used for normal distrubion
    real(8) :: randn, mean, std
    randn = (-2 * log(rand_uniform())) ** 0.5 * cos(2*pi * rand_uniform()) * std + mean
  end function randn

  function randn_vec(n)
    ! returns a random vector of len n, with each element ranging from [0,1).
    integer :: n,i
    real(8) :: randn_vec(n)
    do i = 1,n 
       randn_vec(i) = randn(0.0_8, 1.0_8)
    end do
  end function randn_vec

end module random
