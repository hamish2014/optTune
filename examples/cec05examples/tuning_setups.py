import fortran_SO, numpy, batchOpenMPI
from optTune import get_F_vals_at_specified_OFE_budgets

def DE(tfun_id, tfun_d, Np, F, Cr, evals, randomSeed, DE_x=0, DE_y=1, printLevel=0) :
    """ printLevel=0 for no output """
    if evals < Np:
        raise ValueError,'fortran_SO.DE, evals < Np (%i < %i)' % (evals, Np)
    if Np <= 2*DE_y + 1:
        raise ValueError, 'fortran_SO.DE, Np < 2*DE_y + 1'
    if Np > evals:
        raise ValueError, 'fortran_SO.DE, (Np=%i) > evals ' % Np
    if Cr < 0:
        raise ValueError, 'fortran_SO.DE, (Cr=%f) < 0' % Cr

    if printLevel > 0: 
        print('DE (f_%i D%i), Np %i, F %4.2f, Cr %4.2f, evals %i, randomSeed %s'% (tfun_id, tfun_d, Np, F, Cr, evals, randomSeed))
    #print('tfun_id, tfun_d, Np, F, Cr, evals',tfun_id, tfun_d, Np, F, Cr, evals)
    fm = fortran_SO.de_module #fortran module
    fm.de_x = DE_x
    fm.de_y = DE_y  
    fm.derun(tfun_id, tfun_d, Np, F, Cr, evals, randomSeed )
    F = fm.derun_fvalhist.copy()
    OFEs_made = Np * numpy.arange(1,len(F)+1)
    return F, OFEs_made

def PSO(tfun_id, tfun_d, N, w, c_p, c_g, evals, randomSeed, printLevel=0) :
    if printLevel > 0: 
        print('PSO (f_%i D%i), N %i, w %4.2f, c_p %4.2f, c_g %4.2f, evals %i, randomSeed %i'% (tfun_id, tfun_d, N, w, c_p, c_g, evals, randomSeed))
    if evals < N:
        raise ValueError, 'fortran_SOO.PSO, evals < N'
    if N < 2:
        raise AssertionError, 'fortran_SOO.PSO, N < 2'
    fm = fortran_SO.pso_module 
    fm.psorun(tfun_id, tfun_d, N, w, w, c_p, c_g, evals, randomSeed )
    F = fm.psorun_fvalhist.copy()
    OFEs_made = N * numpy.arange(1,len(F)+1)
    return F, OFEs_made

cec_tp_mins = [0, -450, -450, -450, -450, -310, #0 added as spacer
                390, -180, -140, -330, -330,
                90, -460, -130, -300, 120,
                120, 120, 10, 10, 10, 
                360, 360, 360, 260, 260 ]
prob_ID = 6
prob_D = 30


#setting up for optTune and parallizm

class batchOpenMPI_wrapper:
    def __init__(self, batchFun, prob_id):
        self.batchFun = batchFun
        self.prob_id =  prob_id
    def prep_input( self, CPV_tuple, OFE_budgets, randomSeed):
        return [ self.prob_id ] + CPV_tuple.tolist() + [ max(OFE_budgets), randomSeed ]
    def __call__(self, CPV_tuple, OFE_budgets, randomSeed):
        F, OFEs_made = self.batchFun( self.prep_input( CPV_tuple, OFE_budgets, randomSeed))
        F = F - cec_tp_mins[self.prob_id]
        return  get_F_vals_at_specified_OFE_budgets(F, OFEs_made, OFE_budgets)
    def addtoBatch(self, CPV_tuple, OFE_budgets, randomSeed):
        self.batchFun.addtoBatch(self.prep_input( CPV_tuple, OFE_budgets, randomSeed))

def _DE_batch(x):
    return DE(tfun_id=x[0], Np=int(x[1]), Cr=x[2], F=x[3], evals=x[4], randomSeed=x[5],
              tfun_d=prob_D, DE_x=0, DE_y=1, printLevel=0)
DE_batch = batchOpenMPI.batchFunction( _DE_batch )

def DE_CPV_validity_checks(CPV_array, OFE_budget):
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

DE_CPV_lb = numpy.array([  5, 0.0, 0.0 ])  #lower initilization bound
DE_CPV_ub = numpy.array([ 50, 1.0, 1.0 ])  #upper initilization bound
DE_OFE_budgets = numpy.logspace(1,5,50).astype(int)
DE_sampleSizes = [2,8,15] #resampling size of 25
DE_alpha =  0.9 #resampling_interruption_confidence 
DE_gammaBudget = 50*100*1000*25 #tuning budget equivalent to assessing 50 CPV tuples for 100*1000 OFEs at 25 repeats.

def _PSO_batch(x):
    return PSO(tfun_id=x[0], N=int(x[1]), w=x[2], c_p=x[3], c_g=x[4],  evals=x[5], randomSeed=x[6], tfun_d=prob_D, printLevel=0)
PSO_batch = batchOpenMPI.batchFunction( _PSO_batch )

def PSO_CPV_validity_checks(CPV_array, OFE_budget):
    'check tuning constraints'
    N, w, c_p, c_g = CPV_array
    if OFE_budget < N :
        return False, 'OFE_budget < N'
    if N < 5:
        return False, 'N < 5'
    if c_p < 0 :
        return False, 'c_p < 0'
    if c_g < 0 :
        return False, 'c_g < 0'
    if w < 0 or w > 1 :
        return False, 'intertia factor not in [0,1]'
    return True, ""

PSO_CPV_lb = numpy.array([  5, 0.0, 0.0, 0.0 ]) 
PSO_CPV_ub = numpy.array([ 50, 1.0, 3.0, 3.0 ]) 
PSO_OFE_budgets = DE_OFE_budgets
PSO_sampleSizes = DE_sampleSizes
PSO_alpha = DE_alpha
PSO_gammaBudget = DE_gammaBudget
