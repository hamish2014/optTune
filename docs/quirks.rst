.. _quirks:

Quirks
======

Some quirks when tuning under multiple OFE budgets.

get_F_vals_at_specified_OFE_budgets
------------------------------------

The *get_F_vals_at_specified_OFE_budgets* function is nessary as the following example illustrates.
For this example an algorithm is assessed which makes use of a population size of 5, and the tuner wants utility values at [ 2, 3, 5, 7, 11, 16, 20, 30].

.. literalinclude:: ../examples/get_F_vals_at_specified_OFE_budgets_ex.py

which returns::
    
  F : [ 0.5   0.5   0.3   0.2   0.15  0.15]
  E : [ 5  7 11 16 20 30]
