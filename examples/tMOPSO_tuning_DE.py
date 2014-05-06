import DE_tuning_setup 
from optTune import tMOPSO, linearFunction
from matplotlib import pyplot

tuningOpt = tMOPSO( 
    optAlg = DE_tuning_setup.run_DE_on_Ros_ND, 
    CPV_lb = DE_tuning_setup.CPV_lb, 
    CPV_ub = DE_tuning_setup.CPV_ub,
    CPV_validity_checks = DE_tuning_setup.CPV_validity_checks,
    OFE_budgets= DE_tuning_setup.OFE_budgets_to_tune_under,
    sampleSizes =  DE_tuning_setup.sampleSizes,
    resampling_interruption_confidence = 0.6,
    gammaBudget = DE_tuning_setup.tuningBudget,
    OFE_assessment_overshoot_function = linearFunction(2, 100 ),
    N = 10,
    saveTo = 'tMOPSO_tuning_DE.data'
    )
print(tuningOpt)

#extracting data from the Pareto-optimal front Approximation
OFE_budgets = [ d.fv[0] for d in tuningOpt.PFA.designs ] 
Fmin_values = [ d.fv[1] for d in tuningOpt.PFA.designs ] 

log_OFE_budgets = [ d.xv[0] for d in tuningOpt.PFA.designs ]
N_values =        [ int(d.xv[1]) for d in tuningOpt.PFA.designs ]
Cr_values =       [ d.xv[2] for d in tuningOpt.PFA.designs ]
F_values =        [ d.xv[3] for d in tuningOpt.PFA.designs ]

line_Cr = pyplot.semilogx(OFE_budgets, Cr_values, 'b^')[0]
line_F  = pyplot.semilogx(OFE_budgets, F_values, 'rx')[0]
pyplot.ylabel('Cr, F')
pyplot.twinx()
line_N = pyplot.semilogx(OFE_budgets, N_values, 'go')[0]
pyplot.ylim( 0, max(N_values)*1.1)
pyplot.ylabel('N')
pyplot.legend([line_Cr,line_F,line_N], ['Cr','F','N'], loc='upper center')
pyplot.xlabel('OFE budget')
pyplot.xlim(min(OFE_budgets)-1,max(OFE_budgets)+60)
pyplot.title('Optimal CPVs for different OFE budgets')

pyplot.show()
