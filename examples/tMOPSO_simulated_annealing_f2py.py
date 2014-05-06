"""
Tune the simulated annealing algorithm from the scipy package to the generalized Rosenbrock problem, for multiple objective function evaluation (OFE) budgets simulatenously.
Same as the other example, except a fortran version of fast sa is used.
"""
import numpy, os
from optTune import tMOPSO, get_F_vals_at_specified_OFE_budgets, linearFunction
print('Please note this example only works on Linux, and requires gfortran')
if not os.path.exists('anneal_fortran.so'):
    os.system('f2py -c -m anneal_fortran anneal.f90')
from anneal_fortran import anneal_module
D = 5 #number of dimensions for Rosenbrock problem

def anneal(CPVs, OFE_budgets, randomSeed):
    #fast_sa_run - Function signature:
    #  fast_sa_run(prob_id,x0,t0,dwell,m,n,quench,boltzmann,maxevals,lower,upper,random_seed,[d])
    anneal_module.fast_sa_run(prob_id = 1 ,
                              x0 = -2.048 + 2*2.048*numpy.random.rand(D),
                              t0 = 500.0,
                              dwell = int(CPVs[0]),
                              m = CPVs[1],
                              n = 1.0,
                              quench = 1.0,
                              boltzmann = 1.0,
                              maxevals = max(OFE_budgets),
                              lower = -2.048*numpy.ones(D),
                              upper =  2.048*numpy.ones(D),
                              random_seed = randomSeed)
    return get_F_vals_at_specified_OFE_budgets(F=anneal_module.fval_hist.copy(), E=anneal_module.eval_hist.copy(), E_desired=OFE_budgets)

def CPV_valid(CPVs, OFE_budget):
    if CPVs[0] < 5:
        return False,'dwell,CPVs[0] < 5'
    if CPVs[1] < 0.0001:
        return False,'CPVs[1] < 0.0001'
    return True,''

tuningOpt = tMOPSO( 
    optAlg = anneal, 
    CPV_lb = numpy.array([10, 0.0]), 
    CPV_ub = numpy.array([50, 5.0]),
    CPV_validity_checks = CPV_valid,
    OFE_budgets=numpy.logspace(1,3,30).astype(int),
    sampleSizes = [2,8,20], #resampling size of 30
    resampling_interruption_confidence = 0.6,
    gammaBudget = 30*1000*50, #increase to get a smoother result ...
    OFE_assessment_overshoot_function = linearFunction(2, 100 ),
    N = 10,
    printLevel=1,
    )
print(tuningOpt)

Fmin_values = [ d.fv[1] for d in tuningOpt.PFA.designs ] 
OFE_budgets = [ d.fv[0] for d in tuningOpt.PFA.designs ]
dwell_values =  [ int(d.xv[1]) for d in tuningOpt.PFA.designs ]
m_values = [ d.xv[2] for d in tuningOpt.PFA.designs ]

print('OFE budget     Fmin      dwell       m     ')
for a,b,c,d in zip(OFE_budgets, Fmin_values, dwell_values, m_values):
    print('  %i       %6.4f      %i       %4.2f' % (a,b,c,d))

from matplotlib import pyplot
p1 = pyplot.semilogx(OFE_budgets, dwell_values, 'g.')[0]
pyplot.ylabel('dwell')
pyplot.ylim( min(dwell_values) - 1, max( dwell_values) + 1)
pyplot.twinx()
p2 = pyplot.semilogx(OFE_budgets, m_values, 'bx')[0]
pyplot.ylim( 0, max(m_values)*1.1)
pyplot.ylabel('m (rate of cool)')
pyplot.legend([p1,p2],['dwell','m'], loc='best')
pyplot.xlabel('OFE budget')
pyplot.xlim(min(OFE_budgets)-1,max(OFE_budgets)+60)
pyplot.title('Optimal CPVs for different OFE budgets')

pyplot.show()
