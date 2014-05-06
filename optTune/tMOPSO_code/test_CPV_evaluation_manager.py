#! /usr/bin/env python

import sys, numpy
sys.path.append('../../')
from tMOPSO_module import CPV_evaluation_manager, get_F_vals_at_specified_OFE_budgets

def optAlg(y):
    ''' y = [CPV_1, OFE_budget, randomSeed] '''
    n = numpy.random.randint(5,10)
    E = numpy.arange(1,n)
    F = numpy.arange(1,n)
    return F, E

def addtobatch(y):
    pass


print('Test to check tMOPSOs mechanisms for handling algorithm which terminate before reaching the specified OFE budget')

c = CPV_evaluation_manager(numpy.array([1,2]), optAlg, 10, 8, addtobatch)

sampleSizes = [3,5]

for i, repeats in enumerate(sampleSizes):
    evals, fvals = c.results(repeats)
    print(fvals)
    print('evals_used in generating extra %i runs : %i (total evals should be %i)' % (repeats, c.OFEs_used_over_last_n_runs(repeats),sum(fvals[len(fvals)-1,:])))

print('\ntesting, tMOPSO get_F_vals_at_specified_OFE_budgets')
Fv = numpy.array([ 0.9, 0.5, 0.3, 0.2 , 0.15, 0.12 ])
Ev = numpy.arange(1,len(Fv)+1)*3
E_desired = numpy.array([3,7,10,11,14,20,25])
def sub_fun(F_in,E_in,E_d):
    print('zip(E_in,F_in) %s' %' '.join('%i,%1.2f ' % (e,f) for e,f in zip(E_in,F_in)))
    print('  E_desired %s' % ' '.join(map(str,E_d)))
    F_out, E_out =  get_F_vals_at_specified_OFE_budgets(F_in,E_in,E_d)
    print('  zip(E_out,F_out) %s' %' '.join('%i,%1.2f ' % (e,f) for e,f in zip(E_out,F_out)))
sub_fun(Fv,Ev,E_desired)
sub_fun(Fv[1:],Ev[1:],E_desired)
