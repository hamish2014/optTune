
gfortran -fPIC -c cec2005problems_load_data_files.F90 -Ddata_dir="'$PWD/cec_data/'"

f2py -c -m fortran_SO random.f90 cec2005problems.f90 de.f90 pso.f90 cec2005problems_load_data_files.o