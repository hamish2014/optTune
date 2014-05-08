from matplotlib import pyplot

def _plot_pareto_front( OFE_budgets, Fmin_values ):
    pyplot.loglog(OFE_budgets, Fmin_values)
    pyplot.xlabel('OFE budget')
    pyplot.ylabel('solution error')
    pyplot.title('Pareto-optimal curve of speed versus accuracy trade-off')

def plot_DE_results( OFE_budgets, Fmin_values, N_values, Cr_values, F_values ):
    pyplot.figure()
    pyplot.subplot(1,2,1)
    _plot_pareto_front( OFE_budgets, Fmin_values )
    pyplot.subplot(1,2,2)
    line_Cr = pyplot.semilogx(OFE_budgets, Cr_values, 'b^')[0]
    line_F  = pyplot.semilogx(OFE_budgets, F_values, 'rx')[0]
    pyplot.ylabel('Cr, F')
    pyplot.twinx()
    line_N = pyplot.semilogx(OFE_budgets, N_values, 'go')[0]
    pyplot.ylim( 0, max(N_values)*1.1)
    pyplot.ylabel('N')
    pyplot.legend([line_Cr,line_F,line_N], ['Cr','F','N'], loc='upper center')
    pyplot.xlim(min(OFE_budgets),max(OFE_budgets))
    pyplot.xlabel('OFE budget')
    pyplot.title('Optimal CPVs for different OFE budgets')

def plot_PSO_results( OFE_budgets, Fmin_values, N_values, w_values, c_p_values,c_g_values):
    pyplot.figure()
    pyplot.subplot(1,2,1)
    _plot_pareto_front( OFE_budgets, Fmin_values )
    pyplot.subplot(1,2,2)
    line_w  = pyplot.semilogx(OFE_budgets, w_values, 'rx')[0]
    line_c_p = pyplot.semilogx(OFE_budgets, c_p_values, 'b^')[0]
    line_c_g = pyplot.semilogx(OFE_budgets, c_g_values, 'mv')[0]
    pyplot.ylabel('w, c_p, c_g')
    pyplot.ylim( 0, 4.2)
    pyplot.twinx()
    line_N = pyplot.semilogx(OFE_budgets, N_values, 'go')[0]
    pyplot.ylim( 0, max(N_values)*1.1)
    pyplot.ylabel('N')
    pyplot.legend([line_N, line_w, line_c_p, line_c_g], ['N','w','c_p','c_g'], loc='upper center')
    pyplot.xlim(min(OFE_budgets),max(OFE_budgets))
    pyplot.xlabel('OFE budget')
    pyplot.title('Optimal CPVs for different OFE budgets')
