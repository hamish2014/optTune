Speeding things up
==================

Tuning algorithms is an computationaly expensive exercise, there are few approaches which can be followed to speed things up. Optune examples demonstrating each approach can be found under the examples directory, which is distrubuted with the optTune source code.

Parallel Computing
------------------

If you PC has multiple cores, or you access to super computing time, parallization is the way to go. The task of CPV tuning is *embarassingly parallizable*, and can be spead up significantly.

Use Faster Codes
----------------

If the optimization algorithm being tuned is written in a high-level, language design for functionality and not brute force, such as Python or Octave/Matlab, re write the code in a faster language such as C++ or fortran. Optimization runs can go upto a 100 times faster if this is done ...

Once your codes have been migrated to a faster language, all that remains to interface those code with Python and the optTune package. There are a variety of ways to achieve this, 

1. system calls, via subprocess or batchOpenMPI
2. compiling a shared fortran library using f2py, see :ref:`tMOPSO-f2py-example`.
3. compiling a C or C++ library, and linking to it using an Python module such as Ctypes
