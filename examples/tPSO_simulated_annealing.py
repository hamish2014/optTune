"""
Tune the simulated annealing algorithm from the scipy package to the 5D Rosenbrock problem, for multiple objective function evaluation budgets simulatenously. The contorl parameter values (CPVs)  tuned are those which control are those which control the rate of exploration versus the rate of exploitation, namely 
 * m - control how quickly the Temperature cools down, i.e. higher m = higher exploration and lower exploitation.
        c = m * exp(-n * quench)
        T_new = T0 * exp(-c * k**quench)
 * dwell - evaluation at each tempreture
"""
import numpy
from scipy import optimize
from optTune import tPSO


def Ros_ND(x) :
    "gerneralised Rossenbrock function"
    return sum([100*(x[ii+1]-x[ii]**2)**2 + (1-x[ii])**2 for ii in range(len(x)-1)])

def run_simulated_annealing(CPVs, OFE_budget, randomSeed):
    R = optimize.anneal(Ros_ND, 
                    x0 = -2.048 + 2*2.048*numpy.random.rand(5),
                    dwell=int(CPVs[0]), 
                    m = CPVs[1],
                    T0 = 500.0,
                    lower= -2.048,
                    upper=  2.048,
                    maxeval = OFE_budget, #termination criteria
                    feps = 0.0, 
                    Tf = 0.0)
    return  Ros_ND(R[0]) # where xMin = R[0]

def CPV_valid(CPVs, OFE_budget):
    if CPVs[0] > OFE_budget:
        return False, 'dwell,CPVs[0] > OFE budget'
    if CPVs[0] < 5:
        return False,'dwell,CPVs[0] < 5'
    if CPVs[1] < 0.0001:
        return False,'CPVs[1] < 0.0001'
    return True,''

tuningOpt = tPSO( 
    optAlg = run_simulated_annealing, 
    CPV_lb = numpy.array([10, 0.0]), 
    CPV_ub = numpy.array([50, 5.0]),
    CPV_validity_checks = CPV_valid,
    OFE_budget= 500,
    sampleSizes = [2,8,20], #resampling size of 30
    resampling_interruption_confidence = 0.6,
    gammaBudget = 500*1000,
    N = 10,
    )
print(tuningOpt)
print('''Best CPVs founds for Rossenbrock 5D, given an OFE budget of %i, 
  - dwell               %f
  - cool down rate (m)  %f
''' % (tuningOpt.OFE_budget, tuningOpt.x_gb[0], tuningOpt.x_gb[1]))

