import numpy
from optTune import get_F_vals_at_specified_OFE_budgets

evals_made = numpy.array( [ 5 , 10, 15, 20 ])
solution_error_achieved = numpy.array([ 0.5, 0.3, 0.2, 0.15])
OFE_budgets = numpy.array( [ 2, 3, 5, 7, 11, 16, 20, 30] )
F, E = get_F_vals_at_specified_OFE_budgets( solution_error_achieved , evals_made, OFE_budgets)

print('F : %s' % F)
print('E : %s' % E)


