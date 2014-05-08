module  cec2005problems_load_data_files
  implicit None

contains

  subroutine load_data_file(m, D, fname)
    real(8),allocatable,dimension(:),intent(inout) :: m
    integer :: D
    character(len=*) :: fname
    allocate(m(D))
    open(12,file= data_dir // fname, status='old')
    read(12,*) m 
    close(12)
  end subroutine load_data_file

  subroutine load_data_file_2D(m, n1, n2, fname)
    real(8),allocatable,intent(inout) :: m(:,:)
    integer :: n1, n2
    character(len=*) :: fname
    allocate(m(n1,n2))
    open(12,file= data_dir // fname, status='old')
    read(12,*) m 
    close(12)
  end subroutine load_data_file_2D

end module cec2005problems_load_data_files
