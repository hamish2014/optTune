#! /usr/bin/env python
import numpy, tuning_setups, batchOpenMPI, plotlib
from optTune import tMOPSO
from matplotlib import pyplot

batchOpenMPI.begin_MPI_loop()
#adjust which CEC problem to tune under, etcetera in tuning_setups.py
runAlg = tuning_setups.batchOpenMPI_wrapper( tuning_setups.PSO_batch, tuning_setups.prob_ID)

tuningOpt = tMOPSO( 
    optAlg = runAlg, 
    CPV_lb = tuning_setups.PSO_CPV_lb , 
    CPV_ub = tuning_setups.PSO_CPV_ub ,
    CPV_validity_checks = tuning_setups.PSO_CPV_validity_checks,
    OFE_budgets =  tuning_setups.PSO_OFE_budgets,
    sampleSizes = tuning_setups.PSO_sampleSizes, #resampling size of 25
    resampling_interruption_confidence = tuning_setups.PSO_alpha,
    gammaBudget = tuning_setups.PSO_gammaBudget, #tuning budget
    addtoBatch = runAlg.addtoBatch, 
    processBatch = batchOpenMPI.processBatch
    )

batchOpenMPI.end_MPI_loop(print_stats=True) #release workers, and print stats

print(tuningOpt)

OFE_budgets = [ d.fv[0] for d in tuningOpt.PFA.designs ] 
Fmin_values = [ d.fv[1] for d in tuningOpt.PFA.designs ] 

log_OFE_budgets = [ d.xv[0] for d in tuningOpt.PFA.designs ]
N_values =        [ int(d.xv[1]) for d in tuningOpt.PFA.designs ]
w_values =        [ d.xv[2] for d in tuningOpt.PFA.designs ]
c_p_values =      [ d.xv[3] for d in tuningOpt.PFA.designs ]
c_g_values =      [ d.xv[4] for d in tuningOpt.PFA.designs ]

plotlib.plot_PSO_results( OFE_budgets, Fmin_values, N_values, w_values, c_p_values, c_g_values )

pyplot.show()
