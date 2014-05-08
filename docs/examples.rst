Examples
========

The example below although written for Linux, should work on another operating system if adjusted accordingly.

Tuning DE and PSO to the CEC'05 problems under multiple OFE budgets
-------------------------------------------------------------------

For this example, a virtual machine running Ubuntu 14.04 will be setup to tune DE and PSO to the CEC'05 problems using tMOPSO.

On your new virtual machine, with a fresh Ubuntu 14.04 installed, in a terminal ::

  $ sudo apt-get install python-numpy python-scipy python-matplotlib   # python modules which optTune depends on
  $ sudo apt-get install python-setuptools   #install optTune for the local user
  $ easy_install --user optTune

f2py and batchOpenMPI are used for speed: f2py as to easily interface with the DE and PSO fortran codes, and batchOpenMPI for parallelization.
For these you will need the following software ::

  $ sudo apt-get install gfortran openmpi-bin openmpi-common libopenmpi-dev
  $ easy_install --user mpi4py
  $ easy_install --user batchOpenMPI 

Next download and extract the CEC 2005 tuning example files located at `cec05examples.tar.gz <cec05examples.tar.gz>`_ ::
  
  $ wget https://pythonhosted.org/optTune/cec05examples.tar.gz
  $ tar -xf cec05examples.tar.gz
  $ cd cec05examples/
  $ ls

The first thing to do is to compile and then validate the fortran codes ::

  $ ./compile.sh
  $ sudo apt-get install octave #octave is used in the testing scripts ...
  $ ./test_cec2005problems.py 

Examine the testing data, and if acceptable ::

  # to tune DE using 4 processors 
  $ mpirun -np 4 tune_DE_via_tMOPSO.py 
  # or to tune PSO
  $ mpirun -np 4 tune_PSO_via_tMOPSO.py 

Have fun.
