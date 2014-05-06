import numpy, os
from optTune import MOTA,  get_F_vals_at_specified_OFE_budgets, linearFunction
from optTune import MOTA_subproblem_weighted_sum
print('Please note this example is writen for Linux, and requires gfortran')
if not os.path.exists('anneal_fortran.so'):
    os.system('f2py -c -m anneal_fortran anneal.f90')
from anneal_fortran import anneal_module
from matplotlib import pyplot

D = 12 #number of dimensions for Rosenbrock and Sphere problems
anneal_KWs = dict(
    t0 = 500.0,
    n = 1.0,
    quench = 1.0,
    boltzmann = 1.0,
)

def ros_ND( CPVs, OFE_budgets, randomSeed):
    anneal_module.fast_sa_run( 
        prob_id = 1,
        x0 = -2.048 + 2*2.048*numpy.random.rand(D),
        dwell = int(CPVs[0]), m = CPVs[1], 
        maxevals = max(OFE_budgets), 
        random_seed = randomSeed,
        lower = -2.048*numpy.ones(D),
        upper =  2.048*numpy.ones(D),
        **anneal_KWs )
    return get_F_vals_at_specified_OFE_budgets(F=anneal_module.fval_hist.copy(), E=anneal_module.eval_hist.copy(), E_desired=OFE_budgets)

def sphere_ND( CPVs, OFE_budgets, randomSeed):
    anneal_module.fast_sa_run( 
        prob_id = 2,
        x0 = -100 + 200*numpy.random.rand(D),
        dwell = int(CPVs[0]), m = CPVs[1], 
        maxevals = max(OFE_budgets), 
        random_seed = randomSeed,
        lower = -100.0*numpy.ones(D),
        upper =  100.0*numpy.ones(D),
        **anneal_KWs )
    return get_F_vals_at_specified_OFE_budgets(F=anneal_module.fval_hist.copy(), E=anneal_module.eval_hist.copy(), E_desired=OFE_budgets)

def CPV_valid(CPVs, OFE_budget):
    if CPVs[0] < 2:
        return False,'dwell,CPVs[0] < 2'
    if CPVs[1] < 0.0001:
        return False,'CPVs[1] < 0.0001'
    return True,''

subproblems =  []
for w_1 in numpy.linspace(0,1,5):
    subproblems.append( MOTA_subproblem_weighted_sum ( 
            w = numpy.array([w_1, 1-w_1]),
            target_OFE_budgets = numpy.logspace(1,3,30).astype(int),
            gammaBudget = 50*1000*50,
            updateNeighbours = [] #will do now
            ))
for i in range(len(subproblems)):
    subproblems[i].updateNeighbours = [subproblems[(i-1)%5],subproblems[(i+1)%5]]

tuningOpt = MOTA( 
    objectiveFunctions = [ros_ND, sphere_ND],
    subproblems = subproblems, 
    CPV_lb = numpy.array([30, 0.0]), 
    CPV_ub = numpy.array([50, 5.0]),
    CPV_validity_checks = CPV_valid,
    sampleSizes = [5,10,35], #resampling size of 30
    resampling_interruption_confidence = 0.6,
    printLevel=2,
    )
print(tuningOpt)

print('\nplotting optimal CPVs for different OFE budgets')

for j,S in enumerate(tuningOpt.subproblems):
    print(S)
    Fmin_values = [ d.fv[1] for d in S.PFA.designs ] 
    OFE_budgets = [ d.fv[0] for d in S.PFA.designs ]
    dwell_values =  [ int(d.xv[1]) for d in S.PFA.designs ]
    m_values = [ d.xv[2] for d in S.PFA.designs ]
    pyplot.subplot(1,5,j+1)

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
    pyplot.title('w = %s' % str(S.w))

pyplot.show()
