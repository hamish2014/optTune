tMOPSO
======================

.. automodule:: optTune.tMOPSO_code.tMOPSO_module

.. autoclass:: optTune.tMOPSO

Examples
--------

pure Python example
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/tMOPSO_simulated_annealing.py

.. _tMOPSO-f2py-example:

f2py example
^^^^^^^^^^^^

A faster way of the doing the above examples is to use `f2py <www.f2py.com/>`_, and then call the resulting shared object from Python.

Fortran code

.. literalinclude:: ../examples/anneal.f90
   :language: fortran

Python code

.. literalinclude:: ../examples/tMOPSO_simulated_annealing_f2py.py
