"""
Tune the simulated annealing algorithm from the scipy package to the 5D Rosenbrock problem, for multiple objective function evaluation budgets simulatenously.
The contorl parameter values (CPVs)  tuned are those which control are those which control the rate of exploration versus the rate of exploitation, namely 
 * m - control how quickly the Temperature cools down, i.e. higher m = higher exploration and lower exploitation.
        c = m * exp(-n * quench)
        T_new = T0 * exp(-c * k**quench)
 * dwell - evaluation at each tempreture
"""
import numpy
from scipy import optimize
from optTune import tMOPSO, evaluation_history_recording_wrapper, get_F_vals_at_specified_OFE_budgets, linearFunction


def Ros_ND(x) :
    "gerneralised Rossenbroch function"
    return sum([100*(x[ii+1]-x[ii]**2)**2 + (1-x[ii])**2 for ii in range(len(x)-1)])

def solution_valid(x):
    return ( -2.048 <= x ).all() and ( x <= 2.048 ).all()

def run_simulated_annealing(CPVs, OFE_budgets, randomSeed):
    dwell = int(CPVs[0]) #equibavent to population size in evolutionary algorithms
    func = evaluation_history_recording_wrapper( Ros_ND, dwell, solution_valid )
    optimize.anneal(func, 
                    x0 = -0.5 * numpy.random.rand(5),
                    #x0 = -2.048 + 2*2.048*numpy.random.rand(10), #if used make sure tMOPSO sample size greater than 100
                    m = CPVs[1],
                    T0 = 500.0,
                    lower= -2.048,
                    upper=  2.048,
                    dwell=dwell, 
                    maxeval = max( OFE_budgets ), #termination criteria
                    feps = 0.0, 
                    Tf = 0.0)
    return get_F_vals_at_specified_OFE_budgets(F=func.f_hist, E=func.OFE_hist, E_desired=OFE_budgets)

def CPV_valid(CPVs, OFE_budget):
    if CPVs[0] < 5:
        return False,'dwell,CPVs[0] < 5'
    if CPVs[1] < 0.0001:
        return False,'CPVs[1] < 0.0001'
    return True,''

tuningOpt = tMOPSO( 
    optAlg = run_simulated_annealing, 
    CPV_lb = numpy.array([10, 0.0]), 
    CPV_ub = numpy.array([50, 5.0]),
    CPV_validity_checks = CPV_valid,
    OFE_budgets=numpy.logspace(1,3,30).astype(int),
    sampleSizes = [2,8,20], #resampling size of 30
    resampling_interruption_confidence = 0.6,
    gammaBudget = 30*1000*50, #increase to get a smoother result ...
    OFE_assessment_overshoot_function = linearFunction(2, 100 ),
    N = 10,
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
pyplot.legend([p1,p2],['dwell','m'], loc='upper center')
pyplot.xlabel('OFE budget')
pyplot.xlim(min(OFE_budgets)-1,max(OFE_budgets)+60)
pyplot.title('Optimal CPVs for different OFE budgets')
#pyplot.figure()
#tuningOpt.plot_HV_hist()
#pyplot.title('Hyper Volume History of multi-objective tuning optimization')

pyplot.show()
