module cec2005problems
  use random
  use cec2005problems_load_data_files
  implicit None
  integer :: cec_tp = 1

  real(8), allocatable, dimension(:) :: o_1, o_2, o_3, o_5, B_5, o_6, o_7, o_8, o_9, o_10, o_11, alpha_12, a_12_i, o_13, o_14
  real(8), allocatable, dimension(:,:) :: M_3, A_5, A_5_all, M_7, M_8, M_10, M_11, data_12,  a_12, b_12, M_14
  real(8) :: k_sum_f11 = 0.0
  !Composition functions
  real(8), allocatable :: f15_o(:,:), f15_M(:,:,:), f15_data(:,:)
  real(8) :: f15_max(10), f15_lambda(10),  f15_sigma(10), f15_bias(10)
  integer :: f15_func_ids(10)
  real(8), allocatable :: f16_o(:,:), f16_M(:,:,:),  f16_data(:,:), f16_M_data(:,:)
  real(8) :: f16_max(10), f16_lambda(10),  f16_sigma(10), f16_bias(10)
  integer :: f16_func_ids(10)
  real(8), allocatable :: f18_o(:,:), f18_M(:,:,:),  f18_data(:,:), f18_M_data(:,:)
  real(8) :: f18_max(10), f18_lambda(10),  f18_sigma(10), f18_bias(10)
  integer :: f18_func_ids(10)
  real(8), allocatable :: f19_o(:,:), f19_M(:,:,:),  f19_data(:,:), f19_M_data(:,:)
  real(8) :: f19_max(10), f19_lambda(10),  f19_sigma(10), f19_bias(10)
  integer :: f19_func_ids(10)
  real(8), allocatable :: f20_o(:,:), f20_M(:,:,:),  f20_data(:,:), f20_M_data(:,:)
  real(8) :: f20_max(10), f20_lambda(10),  f20_sigma(10), f20_bias(10)
  integer :: f20_func_ids(10)
  real(8), allocatable :: f21_o(:,:), f21_M(:,:,:),  f21_data(:,:), f21_M_data(:,:)
  real(8) :: f21_max(10), f21_lambda(10),  f21_sigma(10), f21_bias(10)
  integer :: f21_func_ids(10)
  real(8), allocatable :: f22_o(:,:), f22_M(:,:,:),  f22_data(:,:), f22_M_data(:,:)
  real(8) :: f22_max(10), f22_lambda(10),  f22_sigma(10), f22_bias(10)
  integer :: f22_func_ids(10)
  real(8), allocatable :: f23_o(:,:), f23_M(:,:,:),  f23_data(:,:), f23_M_data(:,:)
  real(8) :: f23_max(10), f23_lambda(10),  f23_sigma(10), f23_bias(10)
  integer :: f23_func_ids(10)
  real(8), allocatable :: f24_o(:,:), f24_M(:,:,:),  f24_data(:,:), f24_M_data(:,:)
  real(8) :: f24_max(10), f24_lambda(10),  f24_sigma(10), f24_bias(10)
  integer :: f24_func_ids(10)

contains

  function cec2005fun(x,D)
    integer :: D
    real(8) :: x(D), cec2005fun
    cec2005fun = 0.0_8
    select case ( cec_tp)
    case (1)
       cec2005fun = f1(x,D)
    case (2) 
       cec2005fun  = f2(x,D)
    case (3) 
       cec2005fun  = f3(x,D) 
    case (4)
       cec2005fun  = (f2(x,D) + 450)*(1 + 0.4*abs(randn(0.0_8,1.0_8))) - 450
    case (5)
       cec2005fun  = f5(x,D)   
    case (6)
       cec2005fun  = f6(x,D)   
    case (7)
       cec2005fun  = f7(x,D)   
    case (8)
       cec2005fun  = f8(x,D)   
    case (9)
       cec2005fun  = f9(x,D)   
    case (10)
       cec2005fun  = f10(x,D)   
    case (11)
       cec2005fun  = f11(x,D)   
    case (12)
       cec2005fun  = f12(x,D)   
    case (13)
       cec2005fun  = f13(x,D)   
    case (14)
       cec2005fun  = f14(x,D)   
    case (15)
       cec2005fun  = f15(x,D)   
    case (16)
       cec2005fun  = f16(x,D)   
    case (17)
       cec2005fun  = 120 + (f16(x,D) - 120)*(1 + 0.2*abs(randn(0.0_8,1.0_8)))
    case (18)
       cec2005fun  = f18(x,D)   
    case (19)
       cec2005fun  = f19(x,D)   
    case (20)
       cec2005fun  = f20(x,D)   
    case (21)
       cec2005fun  = f21(x,D)   
    case (22)
       cec2005fun  = f22(x,D)   
    case (23)
       cec2005fun  = f23(x,D)   
    case (24)
       cec2005fun  = f24(x,D)   
    case (25)
       cec2005fun  = f24(x,D)   
    end select
  end function cec2005fun

  function int2str(i)
    integer :: i
    character(len=20) :: int2str
    write (int2str,*) i
    int2str = adjustl(int2str)
  end function int2str
  

  function round(x)
    real(8) :: x, round
    if (int(x) == x) then
       round = x
    else
       if (x < 0) then
          round = ceiling(x - 0.5_8)
       else
          round = floor(x + 0.5_8)
       end if
    end if
  end function round

  function f1(x,D)
    !Shifted Sphere Function
    integer :: D
    real(8) :: x(D), z(D), f1
    if (.not. allocated(o_1)) call load_data_file(o_1, D, 'sphere_func_data.txt')
    z = x - o_1
    f1 = sum(z*z) - 450.0_8
  end function f1

  function f2(x,D)
    !Shifted Schwefel’s Problem 1.2
    integer :: D, i
    real(8) :: x(D), z(D), f2
    if (.not. allocated(o_2)) call load_data_file(o_2, D, 'schwefel_102_data.txt')
    z = x - o_2
    f2 = -450.0_8
    do i = 1,D
       f2 = f2 + sum(z(1:i))**2
    end do
  end function f2

  function f3(x,D)
    !Shifted Rotated High Conditioned Elliptic Function
    integer :: D, i
    real(8) :: x(D), z(D), f3
    if (.not. allocated(o_3)) then
       call load_data_file(o_3, D, 'high_cond_elliptic_rot_data.txt')
       call load_data_file_2D(M_3, D, D, 'elliptic_M_D' // trim(int2str(D)) // '.txt')
    end if
    !z = dot_Trans(M_3, x - o_3, D)
    z = matmul(M_3, x - o_3)
    f3 = -450.0_8
    do i = 1,D
       f3 = f3 + (10.0_8**6) ** ((i-1.0_8)/(D-1)) * z(i)**2
    end do
  end function f3

  function f5(x,D)
    !Schwefel’s Problem 2.6 with Global Optimum on Bounds
    integer :: D
    real(8) :: x(D), f5
    if (.not. allocated(o_5)) then
       allocate(o_5(D), A_5(D,D), B_5(D))
       call load_data_file_2D(A_5_all, 100, 101, 'schwefel_206_data.txt')
       o_5 = A_5_all(1:D,1) !fortan array order i think
       o_5(1:ceiling(D/4.0)) = -100
       o_5(floor(D*0.75):D) = 100
       A_5 = A_5_all(1:D,2:D+1)
       B_5 = matmul(transpose(A_5),o_5)
    end if
    f5 = maxval(abs(matmul(transpose(A_5),x)-B_5)) -310.0_8
  end function f5

  function f6(x,D)
    !Shifted Rosenbrock’s Function
    integer :: D, i
    real(8) :: x(D), z(D), f6
    if (.not. allocated(o_6)) call load_data_file(o_6, D, 'rosenbrock_func_data.txt')
    z = x - o_6 + 1.0
    f6 = 390.0_8
    do i = 1,D-1
       f6 = f6 + 100.0_8*(z(i)**2.0-z(i+1))**2.0 + (z(i)-1.0_8)**2.0
    end do
  end function f6
  
  function f7(x,D)
    !Shifted Rotated Griewank’s Function without Bounds
    integer :: D, i
    real(8) :: x(D), z(D), f7, sum_i
    if (.not. allocated(o_7)) then
       !print *,'cec2005problems.f90, f7(loading data)'
       call load_data_file(o_7, D, 'griewank_func_data.txt')
       call load_data_file_2D(M_7, D, D, 'griewank_M_D' // trim(int2str(D)) // '.txt')
    end if
    !z = dot_Trans(M_7, x - o_7, D)
    z = matmul(M_7, x - o_7)
    sum_i = 1.0_8
    do i = 1,D
       sum_i = sum_i*cos(z(i)/(i)**0.5_8)
    end do
    f7 = sum(z**2 )/4000 - sum_i + 1 - 180.0_8 
  end function f7

  function f8(x,D)
    !Shifted Rotated Ackley’s Function with Global Optimum on Bounds
    integer :: D, i
    real(8) :: x(D), z(D), f8, sum_i
    if (.not. allocated(o_8)) then
       call load_data_file(o_8, D, 'ackley_func_data.txt')
       do i = 1,D/2
          o_8(2*i-1) = -32.0_8
       end do
       call load_data_file_2D(M_8, D, D, 'ackley_M_D' // trim(int2str(D)) // '.txt')
    end if
    !z = dot_Trans(M_8, x - o_8, D)
    z = matmul(M_8, x - o_8)
    sum_i = 0
    do i = 1,D
       sum_i = sum_i + cos(2*pi*z(i))
    end do
    f8 = -20.0_8*exp(-0.2_8*(1.0_8/D*sum(z**2 ))**0.5_8) - exp(1.0_8/D*sum_i) + 20 + exp(1.0_8) - 140 
  end function f8

  function f9(x,D)
    ! Shifted Rastrigin’s Function
    integer :: D, i
    real(8) :: x(D), z(D), f9
    if (.not. allocated(o_9)) call load_data_file(o_9, D, 'rastrigin_func_data.txt')
    z = x - o_9
    f9 = -330.0_8
    do i = 1,D
       f9 = f9 + z(i)**2.0_8 - 10.0_8*cos(2.0_8*pi*z(i)) + 10.0_8
    end do
  end function f9
  
  function f10(x,D)
    ! Shifted Rotated Rastrigin’s Function
    integer :: D, i
    real(8) :: x(D), z(D), f10
    if (.not. allocated(o_10)) then
       call load_data_file(o_10, D, 'rastrigin_func_data.txt')
       call load_data_file_2D(M_10, D, D, 'rastrigin_M_D' // trim(int2str(D)) // '.txt')
    end if
    !z = dot_Trans(M_10, x - o_10, D)
    z = matmul(M_10, x - o_10)
    f10 = -330.0_8
    do i = 1,D
       f10 = f10 + z(i)**2 - 10*cos(2*pi*z(i)) + 10
    end do
  end function f10

  function f11(x,D)
    ! Shifted Rotated Weierstrass Function
    integer :: D, i, k
    real(8) :: x(D), z(D), f11, sum_k
    integer :: k_max = 20
    real(8) :: a = 0.5_8
    real(8) :: b = 3.0_8
    if (.not. allocated(o_11)) then
       call load_data_file(o_11, D, 'weierstrass_data.txt')
       call load_data_file_2D(M_11, D, D, 'weierstrass_M_D' // trim(int2str(D)) // '.txt')
       k_sum_f11 = 0.0
       do k = 0,k_max
          k_sum_f11 = k_sum_f11 + a**k * cos(2*pi*b**k*0.5)
       end do
    end if
    !z = dot_Trans(M_11, x - o_11, D)
    z = matmul(M_11, x - o_11)
    f11 = 90
    do i = 1,D
       sum_k = 0.0
       do k = 0,k_max
          sum_k = sum_k + a**k * cos(2*pi*b**k*(z(i)+0.5))
       end do
       f11 = f11 + sum_k - k_sum_f11 
    end do
  end function f11

   function f12(x,D)
    !Schwefel’s Problem 2.13
    integer :: D, j, i
    real(8) :: x(D), f12, b_i
    if (.not. allocated(alpha_12)) then
       allocate(a_12(100,100), b_12(100,100), alpha_12(100), a_12_i(D))
       call load_data_file_2D(data_12, 100, 201, 'schwefel_213_data.txt')
       a_12 = data_12(1:100,1:100)
       b_12 = data_12(1:100,101:200)
       alpha_12 = data_12(1:100,201)
       a_12_i = 0.0
       do i = 1,D
          do j = 1,D
             a_12_i(i) = a_12_i(i) + a_12(j,i)*sin(alpha_12(j)) +  b_12(j,i)*cos(alpha_12(j))
          end do
       end do
    end if
    f12 = -460
    do i = 1,D
       b_i = 0
       do j = 1,D
          b_i = b_i + a_12(j,i)*sin(x(j)) +  b_12(j,i)*cos(x(j))
       end do
       f12 = f12 + (a_12_i(i) - b_i)**2
    end do
  end function f12

  function f13(x,D)
    integer :: D, i
    real(8) :: x(D), z(D), f13 , y
    if (.not. allocated(o_13)) call load_data_file(o_13, D, 'EF8F2_func_data.txt')
    z = x - o_13 + 1
    f13 = -130.0_8
    do i = 1,D
       !print *,'i = ',mod(i,D)+1
       y = 100 * (z(i)**2 - z(mod(i,D)+1)) ** 2 + (z(i) - 1)**2
       f13 = f13 + y**2 / 4000 - cos(y) + 1
    end do
  end function f13

  function f14(x,D)
    integer :: D, i
    real(8) :: x(D), z(D), f14, ss
    if (.not. allocated(o_14)) then
       call load_data_file(o_14, D, 'E_ScafferF6_func_data.txt')
       call load_data_file_2D(M_14, D, D, 'E_ScafferF6_M_D' // trim(int2str(D)) // '.txt')
    end if
    !z = dot_Trans(M_14, x - o_14, D)
    z = matmul(M_14, x - o_14)
    f14 = -300
    do i = 1,D
       ss = z(i)**2 + z(mod(i,D)+1)**2
       f14 = f14 + 0.5 + (sin(ss**0.5)**2 - 0.5) / (1 + 0.001_8*ss)**2 
    end do
  end function f14

  function composition_subfunctions(x , D, id)
    !if id < 0, then the `non-continous' form is used.
    integer :: D, id, i, k
    real(8) :: x(D), composition_subfunctions, f, p, x1, x2
    f = 0
    if (id < 0) then ! Non-continous x used
       do i = 1,D
          if (abs(x(i)) >= 0.5_8) x(i) = round(2*x(i))/2
       end do
       id = -id
    end if
    select case (id)
    case (1) ! Rastrigins Function
       f = sum(x **2 - 10*cos(2*pi*x) + 10)
       !print *,'f,x',f,',',x
    case (2) ! Weierstrass Function
       if (k_sum_f11 == 0) then
          do k = 0,20
             k_sum_f11 = k_sum_f11 + 0.5_8**k * cos(2.0_8*pi*3.0_8**k*0.5_8)
          end do
          !print *, 'k_sum_f11 =',k_sum_f11
       end if
       do i = 1,D
          do k = 0,20
             f = f + 0.5_8**k * cos(2.0_8*pi*3.0_8**k*(x(i)+0.5_8))
          end do
       end do
       f = f - D * k_sum_f11
    case (3) ! Griewank function
       p = 1.0_8
       do i = 1,D
          p = p * cos(x(i) / (i*1.0)**0.5)
       end do
       f = sum(x**2)/4000.0_8 - p + 1.0_8
    case (4) ! Ackleys function
       f = -20*exp(-0.2_8*(1.0_8/D*sum(x**2 ))**0.5_8) - exp(1.0_8/D* sum(cos(2*pi*x)) ) + 20 + exp(1.0_8) 
    case (5) !sphere
       f = sum(x**2)
    case (6) !rotated expanded Scaffers F6 function
       do i = 1,D
          x1 = x(i)
          x2 = x(mod(i,D)+1)
          p = x1**2 + x2**2
          f = f + 0.5_8 + ( sin(p**0.5)**2 - 0.5)/ (1 + 0.001_8 * p)**2
       end do
    case (7) !F8F2 function
       do i = 1,D
          x1 = 100 * (x(i)**2 - x(mod(i,D)+1)) ** 2 + (x(i) - 1)**2
          f = f + x1**2 / 4000 - cos(x1) + 1
       end do
    case (8) !High conditioned elliptic function
       do i = 1,D
          f = f + (10.0_8**6) ** ((i-1.0_8)/(D-1)) * x(i)**2
       end do
    case (9) !Sphere function with noise
       f = sum(x**2) * (1 + 0.1*abs(randn(0.0_8,1.0_8)))
    end select
    composition_subfunctions = f
  end function composition_subfunctions

  function composition_function(x, D, func_ids, sigma, lambda, bias, C, o_i, M_i, f_max)
    integer :: D, func_ids(10), i
    real(8) :: composition_function, SW, MaxW, C
    real(8) :: sigma(10), lambda(10), bias(10), w(10), x(D), z(D), y(D), fit(10), f_max(10), o_i(10,D), M_i(10,D,D)
    y = 5
    do i = 1,10
       w(i) = exp(-sum( (x - o_i(i,:))**2.0_8 ) / (2.0_8 * D * sigma(i)**2.0_8))
       !print *,i,'sum ,divisor,w ', sum( (x - o_i(i,:))**2.0_8 ), (2.0_8 * D * sigma(i)**2.0_8),w(i)
       z = matmul( M_i(i,1:D,1:D), (x - o_i(i,1:D))/ lambda(i))
       !z = dot_Trans( M_i(i,1:D,1:D), (x - o_i(i,1:D))/ lambda(i), D)
       fit(i) = C * composition_subfunctions(z, D, func_ids(i) ) / f_max(i)
    end do
    SW = sum(w)
    MaxW = maxval(w)
    do i = 1,10
       if (w(i) /= MaxW) then
          w(i) = w(i) * (1.0_8 - MaxW ** 10.0_8)
       end if
    end do
    w = w / sum(w) !this is different from  problem description, but consistent with code
    !do i = 1,10
    !   print *, 'i, 2000.*f./f1, fit1, w', fit(i), f_max(i), w(i)
    !end do
    composition_function = sum(w*(fit + bias)) 
  end function composition_function

  function f15(x, D)
    integer :: D , i, j
    real(8) :: x(D), z(D), y(D), f15 
    if (.not. allocated(f15_o)) then
       allocate(f15_o(10,D), f15_M(10,D,D))
       call load_data_file_2D(f15_data, 100 , 10, 'hybrid_func1_data.txt')
       y = 5.0
       f15_func_ids = (/ 1,1,2,2,3,3,4,4,5,5 /)
       f15_sigma = 1.0
       f15_lambda = (/1.0_8, 1.0_8, 10.0_8, 10.0_8, 5.0_8/60, 5.0_8/60, 5.0_8/32, 5.0_8/32, 0.05_8, 0.05_8 /)
       do i = 1,10
          f15_bias(i) = (i-1)*100.0_8
          f15_o(i,:) = f15_data(1:D,i)
          f15_M(i,:,:) = 0
          do j = 1,D
             f15_M(i,j,j) = 1.0
          end do
          !z = dot_Trans( f15_M(i,1:D,1:D), y / f15_lambda(i), D)
          z = matmul( f15_M(i,1:D,1:D), y / f15_lambda(i))
          f15_max(i) = composition_subfunctions(z , D, f15_func_ids(i) )
       end do
    end if
    f15 = 120 + composition_function(x, D , f15_func_ids, f15_sigma, f15_lambda, f15_bias,  2000.0_8, f15_o, f15_M, f15_max)
  end function f15

  function f16(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f16
    if (.not. allocated(f16_o)) then
       allocate(f16_o(10,D), f16_M(10,D,D))
       call load_data_file_2D(f16_data, 100 , 10, 'hybrid_func1_data.txt')
       call load_data_file_2D(f16_M_data, D, D*10, 'hybrid_func1_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0
       f16_func_ids = (/ 1,1,2,2,3,3,4,4,5,5 /)
       f16_sigma = 1.0
       f16_lambda = (/1.0_8, 1.0_8, 10.0_8, 10.0_8, 5.0_8/60, 5.0_8/60, 5.0_8/32, 5.0_8/32, 0.05_8, 0.05_8 /)
       do i = 1,10
          f16_bias(i) = (i-1)*100.0_8
          f16_o(i,:) = f16_data(1:D,i)
          !f16_M(i,:,:) = f16_M_data(1+(i-1)*D:i*D, 1:D)
          f16_M(i,:,:) = f16_M_data(1:D, 1+(i-1)*D:i*D)
          !z = dot_Trans( f16_M(i,1:D,1:D), y / f16_lambda(i), D)
          z = matmul( f16_M(i,1:D,1:D), y / f16_lambda(i))
          f16_max(i) = composition_subfunctions(z , D, f16_func_ids(i) )
       end do
    end if
    f16 = 120 + composition_function(x, D , f16_func_ids, f16_sigma, f16_lambda, f16_bias,  2000.0_8, f16_o, f16_M, f16_max)
  end function f16

  function f18(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f18
    if (.not. allocated(f18_o)) then
       allocate(f18_o(10,D), f18_M(10,D,D))
       call load_data_file_2D(f18_data, 100 , 10, 'hybrid_func2_data.txt')
       call load_data_file_2D(f18_M_data, D, D*10, 'hybrid_func2_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f18_func_ids = (/ 4,4,1,1,5,5,2,2,3,3 /)
       f18_sigma =  (/1.0_8, 2.0_8, 1.5_8, 1.5_8, 1.0_8, 1.0_8, 1.5_8, 1.5_8, 2.0_8, 2.0_8 /)
       f18_lambda = (/10.0_8/32, 5.0_8/32, 2.0_8, 1.0_8, 0.10_8, 0.05_8, 20.0_8, 10.0_8, 10.0_8/60, 5.0_8/60 /)
       do i = 1,10
          f18_bias(i) = (i-1)*100.0_8
          if (i /= 10) then
             f18_o(i,:) = f18_data(1:D,i)
          else
             f18_o(i,:) = 0
          end if
          f18_M(i,:,:) = f18_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f18_M(i,1:D,1:D), y / f18_lambda(i))
          f18_max(i) = composition_subfunctions(z , D, f18_func_ids(i) )
       end do
    end if
    f18 = 10 + composition_function(x, D , f18_func_ids, f18_sigma, f18_lambda, f18_bias,  2000.0_8, f18_o, f18_M, f18_max)
  end function f18

  function f19(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f19
    if (.not. allocated(f19_o)) then
       allocate(f19_o(10,D), f19_M(10,D,D))
       call load_data_file_2D(f19_data, 100 , 10, 'hybrid_func2_data.txt')
       call load_data_file_2D(f19_M_data, D, D*10, 'hybrid_func2_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f19_func_ids = (/ 4,4,1,1,5,5,2,2,3,3 /)
       f19_sigma =  (/0.1_8, 2.0_8, 1.5_8, 1.5_8, 1.0_8, 1.0_8, 1.5_8, 1.5_8, 2.0_8, 2.0_8 /)
       f19_lambda = (/0.5_8/32, 5.0_8/32, 2.0_8, 1.0_8, 0.10_8, 0.05_8, 20.0_8, 10.0_8, 10.0_8/60, 5.0_8/60 /)
       do i = 1,10
          f19_bias(i) = (i-1)*100.0_8
          if (i /= 10) then
             f19_o(i,:) = f19_data(1:D,i)
          else
             f19_o(i,:) = 0
          end if
          f19_M(i,:,:) = f19_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f19_M(i,1:D,1:D), y / f19_lambda(i))
          f19_max(i) = composition_subfunctions(z , D, f19_func_ids(i) )
       end do
    end if
    f19 = 10 + composition_function(x, D , f19_func_ids, f19_sigma, f19_lambda, f19_bias,  2000.0_8, f19_o, f19_M, f19_max)
  end function f19

  function f20(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f20
    if (.not. allocated(f20_o)) then
       allocate(f20_o(10,D), f20_M(10,D,D))
       call load_data_file_2D(f20_data, 100 , 10, 'hybrid_func2_data.txt')
       call load_data_file_2D(f20_M_data, D, D*10, 'hybrid_func2_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f20_func_ids = (/ 4,4,1,1,5,5,2,2,3,3 /)
       f20_sigma = (/1.0_8, 2.0_8, 1.5_8, 1.5_8, 1.0_8, 1.0_8, 1.5_8, 1.5_8, 2.0_8, 2.0_8 /)
       f20_lambda = (/10.0_8/32, 5.0_8/32, 2.0_8, 1.0_8, 0.10_8, 0.05_8, 20.0_8, 10.0_8, 10.0_8/60, 5.0_8/60 /)
       do i = 1,D/2
          f20_data(i*2,1) = 5.0
       end do
       f20_data(1:D,10) = 0
       do i = 1,10
          f20_bias(i) = (i-1)*100.0_8
          f20_o(i,:) = f20_data(1:D,i)
          f20_M(i,:,:) = f20_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f20_M(i,1:D,1:D), y / f20_lambda(i))
          f20_max(i) = composition_subfunctions(z , D, f20_func_ids(i) )
       end do
    end if
    f20 = 10 + composition_function(x, D , f20_func_ids, f20_sigma, f20_lambda, f20_bias,  2000.0_8, f20_o, f20_M, f20_max)
  end function f20

  function f21(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f21
    if (.not. allocated(f21_o)) then
       allocate(f21_o(10,D), f21_M(10,D,D))
       call load_data_file_2D(f21_data, 100 , 10, 'hybrid_func3_data.txt')
       call load_data_file_2D(f21_M_data, D, D*10, 'hybrid_func3_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f21_func_ids = (/ 6,6,1,1,7,7,2,2,3,3 /)
       f21_sigma = (/1.0_8, 1.0_8, 1.0_8, 1.0_8, 1.0_8, &
            2.0_8, 2.0_8, 2.0_8, 2.0_8, 2.0_8 /)
       f21_lambda = (/ 0.25_8, 0.05_8, 5.0_8, 1.0_8, 5.0_8, &
            1.0_8, 50.0_8, 10.0_8, 0.125_8, 0.025_8 /)
       !print *, f21_lambda
       do i = 1,10
          f21_bias(i) = (i-1)*100.0_8
          f21_o(i,:) = f21_data(1:D,i)
          f21_M(i,:,:) = f21_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f21_M(i,1:D,1:D), y / f21_lambda(i))
          f21_max(i) = composition_subfunctions(z , D, f21_func_ids(i) )
       end do
       !print *, 'M1(1,1),M_data(1,1)', f21_M(1,1,1), f21_M_data(1,1)
    end if
    f21 = 360 + composition_function(x, D , f21_func_ids, f21_sigma, f21_lambda, f21_bias,  2000.0_8, f21_o, f21_M, f21_max)
  end function f21

  function f22(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f22
    if (.not. allocated(f22_o)) then
       allocate(f22_o(10,D), f22_M(10,D,D))
       call load_data_file_2D(f22_data, 100 , 10, 'hybrid_func3_data.txt')
       call load_data_file_2D(f22_M_data, D, D*10, 'hybrid_func3_HM_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f22_func_ids = (/ 6,6,1,1,7,7,2,2,3,3 /)
       f22_sigma = (/1.0_8, 1.0_8, 1.0_8, 1.0_8, 1.0_8, &
            2.0_8, 2.0_8, 2.0_8, 2.0_8, 2.0_8 /)
       f22_lambda = (/ 0.25_8, 0.05_8, 5.0_8, 1.0_8, 5.0_8, &
            1.0_8, 50.0_8, 10.0_8, 0.125_8, 0.025_8 /)
       do i = 1,10
          f22_bias(i) = (i-1)*100.0_8
          f22_o(i,:) = f22_data(1:D,i)
          f22_M(i,:,:) = f22_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f22_M(i,1:D,1:D), y / f22_lambda(i))
          f22_max(i) = composition_subfunctions(z , D, f22_func_ids(i) )
       end do
       !print *, 'M1(1,1),M_data(1,1)', f22_M(1,1,1), f22_M_data(1,1)
    end if
    f22 = 360 + composition_function(x, D , f22_func_ids, f22_sigma, f22_lambda, f22_bias,  2000.0_8, f22_o, f22_M, f22_max)
  end function f22

  function f23(x, D)
    integer :: D , i, j
    real(8) :: x(D), z(D), y(D), f23
    if (.not. allocated(f23_o)) then
       allocate(f23_o(10,D), f23_M(10,D,D))
       call load_data_file_2D(f23_data, 100 , 10, 'hybrid_func3_data.txt')
       call load_data_file_2D(f23_M_data, D, D*10, 'hybrid_func3_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f23_func_ids = (/ 6,6, 1,1, 7,7, 2,2, 3,3 /)
       f23_sigma = (/1.0_8, 1.0_8, 1.0_8, 1.0_8, 1.0_8, &
            2.0_8, 2.0_8, 2.0_8, 2.0_8, 2.0_8 /)
       f23_lambda = (/ 0.25_8, 0.05_8, 5.0_8, 1.0_8, 5.0_8, &
            1.0_8, 50.0_8, 10.0_8, 0.125_8, 0.025_8 /)
       do i = 1,10
          f23_bias(i) = (i-1)*100.0_8
          f23_o(i,:) = f23_data(1:D,i)
          f23_M(i,:,:) = f23_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f23_M(i,1:D,1:D), y / f23_lambda(i))
          f23_max(i) = composition_subfunctions(z , D, f23_func_ids(i) )
       end do
    end if
    !making x `non-continous'
    do j=1,D
       if (abs(x(j) - f23_o(1,j)) > 0.5) x(j) = round(2*x(j))/2
    end do

    f23 = 360 + composition_function(x, D , f23_func_ids, f23_sigma, f23_lambda, f23_bias,  2000.0_8, f23_o, f23_M, f23_max)
  end function f23

  function f24(x, D)
    integer :: D , i
    real(8) :: x(D), z(D), y(D), f24
    if (.not. allocated(f24_o)) then
       allocate(f24_o(10,D), f24_M(10,D,D))
       call load_data_file_2D(f24_data, 100 , 10, 'hybrid_func4_data.txt')
       call load_data_file_2D(f24_M_data, D, D*10, 'hybrid_func4_M_D' // trim(int2str(D)) // '.txt')
       y = 5.0_8
       f24_func_ids = (/ 2, 6, 7, 4, 1, 3, -6, -1, 8, 9 /)
       f24_sigma = 2
       f24_lambda = (/ 10.0_8, 5.0_8/20, 1.0_8, 5.0_8/32, 1.0_8, &
            0.05_8, 0.1_8, 1.0_8, 0.05_8, 0.05_8 /)
       do i = 1,10
          f24_bias(i) = (i-1)*100.0_8
          f24_o(i,:) = f24_data(1:D,i)
          f24_M(i,:,:) = f24_M_data(1:D, 1+(i-1)*D:i*D)
          z = matmul( f24_M(i,1:D,1:D), y / f24_lambda(i))
          f24_max(i) = composition_subfunctions(z , D, f24_func_ids(i) )
       end do
    end if
    f24 = 260 + composition_function(x, D , f24_func_ids, f24_sigma, f24_lambda, f24_bias,  2000.0_8, f24_o, f24_M, f24_max)
  end function f24

  !CEC function bounds

  function cec2005fun_lowerbound()
    real(8) :: cec2005fun_lowerbound
    cec2005fun_lowerbound = 0
    select case ( cec_tp)
      case (1);  cec2005fun_lowerbound = -100
      case (2);  cec2005fun_lowerbound = -100
      case (3);  cec2005fun_lowerbound = -100
      case (4);  cec2005fun_lowerbound = -100
      case (5);  cec2005fun_lowerbound = -100
      case (6);  cec2005fun_lowerbound = -100
      case (7);  cec2005fun_lowerbound = 0
      case (8);  cec2005fun_lowerbound = -32
      case (9);  cec2005fun_lowerbound = -5
      case (10); cec2005fun_lowerbound = -5
      case (11); cec2005fun_lowerbound = -0.5_8
      case (12); cec2005fun_lowerbound = -pi
      case (13); cec2005fun_lowerbound = -3
      case (14); cec2005fun_lowerbound = -100
      case (15); cec2005fun_lowerbound = -5
      case (16); cec2005fun_lowerbound = -5
      case (17); cec2005fun_lowerbound = -5
      case (18); cec2005fun_lowerbound = -5
      case (19); cec2005fun_lowerbound = -5
      case (20); cec2005fun_lowerbound = -5
      case (21); cec2005fun_lowerbound = -5
      case (22); cec2005fun_lowerbound = -5
      case (23); cec2005fun_lowerbound = -5
      case (24); cec2005fun_lowerbound = -5
      case (25); cec2005fun_lowerbound = 2
    end select
  end function cec2005fun_lowerbound

  function cec2005fun_upperbound()
    real(8) :: cec2005fun_upperbound
    cec2005fun_upperbound = 0
    select case ( cec_tp)
      case (1);  cec2005fun_upperbound = 100
      case (2);  cec2005fun_upperbound = 100
      case (3);  cec2005fun_upperbound = 100
      case (4);  cec2005fun_upperbound = 100
      case (5);  cec2005fun_upperbound = 100
      case (6);  cec2005fun_upperbound = 100
      case (7);  cec2005fun_upperbound = 600
      case (8);  cec2005fun_upperbound = 32
      case (9);  cec2005fun_upperbound = 5
      case (10); cec2005fun_upperbound = 5
      case (11); cec2005fun_upperbound = 0.5_8
      case (12); cec2005fun_upperbound = pi
      case (13); cec2005fun_upperbound = 1
      case (14); cec2005fun_upperbound = 100
      case (15); cec2005fun_upperbound = 5
      case (16); cec2005fun_upperbound = 5
      case (17); cec2005fun_upperbound = 5
      case (18); cec2005fun_upperbound = 5
      case (19); cec2005fun_upperbound = 5
      case (20); cec2005fun_upperbound = 5
      case (21); cec2005fun_upperbound = 5
      case (22); cec2005fun_upperbound = 5
      case (23); cec2005fun_upperbound = 5
      case (24); cec2005fun_upperbound = 5
      case (25); cec2005fun_upperbound = 5
    end select
  end function cec2005fun_upperbound
  
  function cec2005fun_bound_constrained()
    logical :: cec2005fun_bound_constrained
    cec2005fun_bound_constrained = .false.
    select case ( cec_tp)
      case (1);  cec2005fun_bound_constrained = .true.
      case (2);  cec2005fun_bound_constrained = .true.
      case (3);  cec2005fun_bound_constrained = .true.
      case (4);  cec2005fun_bound_constrained = .true.
      case (5);  cec2005fun_bound_constrained = .true.
      case (6);  cec2005fun_bound_constrained = .true.
      case (7);  cec2005fun_bound_constrained = .false.
      case (8);  cec2005fun_bound_constrained = .true.
      case (9);  cec2005fun_bound_constrained = .true.
      case (10); cec2005fun_bound_constrained = .true.
      case (11); cec2005fun_bound_constrained = .true.
      case (12); cec2005fun_bound_constrained = .true.
      case (13); cec2005fun_bound_constrained = .true.
      case (14); cec2005fun_bound_constrained = .true.
      case (15); cec2005fun_bound_constrained = .true.
      case (16); cec2005fun_bound_constrained = .true.
      case (17); cec2005fun_bound_constrained = .true.
      case (18); cec2005fun_bound_constrained = .true.
      case (19); cec2005fun_bound_constrained = .true.
      case (20); cec2005fun_bound_constrained = .true.
      case (21); cec2005fun_bound_constrained = .true.
      case (22); cec2005fun_bound_constrained = .true.
      case (23); cec2005fun_bound_constrained = .true.
      case (24); cec2005fun_bound_constrained = .true.
      case (25); cec2005fun_bound_constrained = .false.
    end select
  end function cec2005fun_bound_constrained

end module cec2005problems
