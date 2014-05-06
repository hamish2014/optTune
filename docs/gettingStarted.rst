Getting Started
===============

Tuning an optimization algorithm under multiple OFE budgets using optTune consists of three parts:

#. setting up the tuning problem
#. applying the tuning algorithm
#. analysing the results


Setting up the tuning problem
-----------------------------

optTune runs the optimization algorithm using the CPV tuple being assessed as to gauge performance at multiple OFE budgets.
The assessment function for tMOPSO is in the form of

.. py:function:: optAlg(CPV_tuple, OFE_budgets, randomSeed):

   :param numpy.array CPV_tuple: CPV_tuple
   :param numpy.array OFE_budgets: OFE_budgets at which to record utility (i.e. solution error)
   :param int randomSeed: seed to be used for optimization run
   :return: Two lists : [ Utility_values, OFE_eval_made ]
	      

Other factors such as the tuning bounds, OFE budgets to tune under, resampling size, also need to be decided upon.

Example setup for tuning Differential Evolution ( DE ; see :ref:`DE_code` ) to the gerneralised Rossenbrock function.

.. literalinclude:: ../examples/DE_tuning_setup.py

Refer to the :ref:`quirks` section as to why the *get_F_vals_at_specified_OFE_budgets* function is nessary.

Applying the tuning algorithm
-----------------------------

After the tuning problem has been defined, then tuning algorithm can applied. Example:

.. literalinclude:: ../examples/tMOPSO_tuning_DE.py
   :lines: 1-2,4-11,13,17


Analysing the results
---------------------

continuing with the tMOPSO tuning DE example

.. literalinclude:: ../examples/tMOPSO_tuning_DE.py
   :lines: 3-4,21-
