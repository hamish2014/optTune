from DE_code import DE_opt, numpy
from optTune import get_F_vals_at_specified_OFE_budgets

def Ros_ND(x) :
    "gerneralised Rossenbrock function"
    return sum([100*(x[ii+1]-x[ii]**2)**2 + (1-x[ii])**2 for ii in range(len(x)-1)])
prob_d = 6

def run_DE_on_Ros_ND(CPV_tuple, OFE_budgets, randomSeed):
    X_min, f_best_hist, X_hist, F_hist = DE_opt(
        objfun = Ros_ND,
        x_lb = -5.0 * numpy.ones(prob_d),
        x_ub =  5.0 * numpy.ones(prob_d),
        Np = int(CPV_tuple[0]),
        Cr = CPV_tuple[1],
        F = CPV_tuple[2],
        evals = max(OFE_budgets),
        printLevel=0
        )
    F =  numpy.array(f_best_hist)
    OFEs_made = int(CPV_tuple[0])*numpy.arange(1,len(X_hist)+1)
    return  get_F_vals_at_specified_OFE_budgets(F, OFEs_made, OFE_budgets)

def CPV_validity_checks(CPV_array, OFE_budget):
    'check tuning constraints'
    N, Cr, F = CPV_array
    if OFE_budget < N :
        return False, 'OFE_budget < N'
    if N < 5:
        return False, 'N < 5'
    if Cr < 0 or Cr > 1 :
        return False, 'Cr not in [0,1]'
    if F < 0:
        return False, 'F < 0'
    return True, ""

#initilization bounds
CPV_lb = numpy.array([  5, 0.0, 0.0 ]) 
CPV_ub = numpy.array([ 50, 1.0, 1.0 ]) 

OFE_budgets_to_tune_under = numpy.logspace(1,3,30).astype(int)

sampleSizes = [2,8,20]

tuningBudget = 50*1000*30 # tuning budget is equvialent of assessing 50 CPV tuples upto 1000 OFEs using 30 resampling runs each
