"""
The tuning multi-objective particle swarm optimization (tMOPSO) algorithm tunes an optimization algorithms to a single problem for multiple objective function evaluation (OFE) budgets.
tMOPSO finds multiple control parameter value (CPV) tuples each of which is optimal for a different OFE budget.
"""

import numpy, math, pickle, datetime
from numpy.random import rand, randn
from copy import deepcopy
from optTune.paretoArchives import paretoArchive2D, PFA_history_recorder
from optTune.paretoArchives.paretoArchive2D_noise import paretoArchive2D_MWUT
from optTune.paretoArchives import paretoArchive2D
from CPV_evaluation_manager import CPV_evaluation_manager
from optTune import linearFunction, _timingWrapper, _passfunction

def to_stdout(text):
    "should not be needed in Python 3.0"
    print(text)

class tMOPSO:
    def __init__(self,
                 optAlg, 
                 CPV_lb, 
                 CPV_ub, 
                 CPV_validity_checks,
                 sampleSizes,
                 gammaBudget, 
                 OFE_budgets = None,
                 OFE_max = None,
                 extra_termination_critea = [],
                 N = 90, 
                 w = 0.0,
                 c_g = 2.0,
                 c_p = 2.0,                
                 c_beta = 0.1,
                 resampling_interruption_confidence = 0.9,
                 resampling_interruption_mode = 'reduce_max',
                 OFE_assessment_overshoot_function = linearFunction(1.5, 100 ),
                 OFE_assessment_undershoot_function = linearFunction(0, 0),
                 constrain_to_initilization_bounds = False,
                 saveTo = None,
                 saveInterval = 10,
                 paretoArchive_local_use_history_info = True,
                 printFunction = to_stdout,
                 printLevel = 2,
                 addtoBatch = _passfunction,
                 processBatch = _passfunction,
                 post_iteration_function = _passfunction ,
                 record_V_hist = True,
                 record_X_hist = True,
                 ):
        """
     Required Args
        * optAlg - function which calls optimization algorithm or numerical method and returns two lists. The first list optAlg should return the utility measure such as the solution error (i.e f_found_opt - f_min) and should be decreasing, and the second list the number of objective function evaluations (OFEs) used in order to determine each element in the first list and should be increasing. The input arguments passed from tMOPSO to optAlg are ( numpy.array([CPV_1, CPV_2, ..., ]), OFE_budgets, randomSeed ). If OFE_budgets is a numpy.array then a solution error for each value in the list is desired, else if OFE_budgets is a integer, the solutions errors for every iteration up until OFE_budgets is desired. The type of OFE_budgets depends upon the OFE_budgets == None, if so then integer, else values from OFE_budgets is passed into optAlg.
        * CPV_lb - initialization lower bound for CPV tuning, i.e.  numpy.array([CPV_1_init_lb, CPV_2_init_lb, ...])
        * CPV_ub - numpy.array([CPV_1_init_ub, CPV_2_init_ub, ...])
        * CPV_validity_checks - function used to check validity of candidate CPVs. Usage CPV_validity_checks(CPV_array, OFE_budget) returns tuple (valid, msg) where msg is string of why invalid. Should be a cheap function, as candidate CPVs are regenerated if they do not satisfy it. Use to prevent negative population sizes, populations size larger then OFE_budget checks, etcetera.
        * sampleSizes - sample sizes used to generate and refine CPV utility values. For example if the sampleSizes are [5,15,30] then all candidate CPVs will be sampled 5 times, then the possibly not dominated CPVs are then sampled another 15 times, and if still promising the for another 30 times. CPV making it to the final iteration are therefore averaged over 50 independent runs.
        * gammaBudget - the number of application layer evaluations (evaluation of the function optAlg optimizes) allocated for the tuning. NB include repeats, i.e. assessing optAlg for on OFE budget of 100 at 5 repeats, counts as a gamma of 500.
        * OFE_budgets - numpy.array of OFE budgets for which the optAlg is to be tuned under. If None then algorithm is tuned under every OFE budget upto OFE_max.
        * OFE_max - maximum OFE budget of interest. Need not if specified if OFE_budgets specified


     Optional Args
        * extra_termination_critea - termination criteria in addition to gammaBudget termination criteria.
        * N - tMOPSO population size
        * w - tMOPSO particle inertia factor
        * c_g - parameter controlling the attraction towards the global guide
        * c_p - parameter controlling the attraction towards the particles personal guide
        * c_beta - particle target OFE budget perturbation factor [0,1], influences each particle velocity in the OFE budget dimension, and the local and global guide selection points.
        * resampling_interruption_confidence - re-sampling interruption confidence level used by paretoArchive2D
        * resampling_interruption_mode - choices=['reduce_max', 'check_all']
        * OFE__assessment_overshoot_function - when assessing a CPV tuple for a OFE budget of beta, this factor is used to control overshoot, beta_actual = OFE__assessment_undershot_function(beta)
        * OFE__assessment_undershoot_function -  like OFE__assessment_overshoot_function except control minimum value
        * saveTo - save optimization to this file after the optimization has been complete, at the interval specified by save_interval, use None to disable saving
        * saveInterval - optimization is saved every `save_interval` iterations. Zero for no saving during optimization   
        * boundConstrained - should CPV be constrained between initialization bounds CPV_lb and CPV_ub
        * paretoArchive_local_use_history_info - use history info from solution error calculations to update Particles local approximations of the Pareto front. This boolean if True, may speed up tuning by increasing the quality of the each particles approximation of the Pareto Front. However, do this increase the computational overhead of tMOPSO.
        * printLevel - 0 no printing, 1 only warnings, 2 overall info, 3 lots of info, 4 verbose (not implemented) 
        * addtoBatch - optAlg inputs are passed to this function before optAlg is called, 
        * processBatch - function is called after all addtoBatch calls have been made, and before any optAlg calls have been made. If used then optAlg, should be retrieve solutions from this functions results
        * post_iteration_function - at the end of each iteration this function is called with tMOPSO instance as the only arg.
        
        """
        self.T_start = datetime.datetime.now()
        self.initializationArgs = locals()
        # required parameters
        self.optAlg = _timingWrapper(optAlg)
        assert OFE_budgets <> None or OFE_max <> None
        self.OFE_budgets =  OFE_budgets
        if OFE_budgets == None:
            self.x_init_lb = numpy.array([0] + list(CPV_lb))
            self.x_init_ub = numpy.array([numpy.log(OFE_max)] + list(CPV_ub))
        else:
            self.x_init_lb = numpy.array([ numpy.log(min(OFE_budgets))] + list(CPV_lb) )
            self.x_init_ub = numpy.array([ numpy.log(max(OFE_budgets))] + list(CPV_ub) )
        self.CPV_validity_checks = CPV_validity_checks
        self.n_dim = len(self.x_init_lb) 
        self.sampleSizes = sampleSizes
        self.gammaBudget = gammaBudget
        # optional parameters
        self.extra_termination_critea = extra_termination_critea
        self.N = N
        self.w  = w
        self.c_g = c_g
        self.c_p = c_p
        self.c_beta = c_beta
        self.icl = resampling_interruption_confidence
        self.resampling_interruption_mode = resampling_interruption_mode
        self.OFE_assessment_overshoot_function = OFE_assessment_overshoot_function
        self.OFE_assessment_undershoot_function = OFE_assessment_undershoot_function
        self.constrain_to_initilization_bounds = constrain_to_initilization_bounds
        self.PFA = paretoArchive2D_MWUT() #global guide
        self.paretoArchive_local_use_history_info = paretoArchive_local_use_history_info
        self.optAlg_addtoBatch = addtoBatch
        self.optAlg_processBatch = _timingWrapper(processBatch)
        self.saveInterval = saveInterval        
        self.printFunction = printFunction
        self.printLevel = printLevel
        if saveTo <> None :
            self.saveProgress = True
            self.saveTo = saveTo
        else :
            self.saveProgress = False
        self.post_iteration_function = post_iteration_function
        self.record_V_hist = record_V_hist
        self.record_X_hist = record_X_hist
        # additional stuff
        self.it = 0
        self.localGuides =  [ paretoArchive2D() for i in range(N) ]
        self.log_beta_max = self.x_init_ub[0]
        self.OFE_budget_max = int(numpy.exp(self.log_beta_max )) #max OFE budget for algorithm being tuned
        self.OFE_budget_min = int(numpy.exp(self.x_init_lb[0]))
        self.optAlg_evals_made = 0
        self.evaluate_candidate_designs_stats = []
        self.PFA_history = PFA_history_recorder()
        self.V_history = []
        self.X_history = []
        self.continueOptimisation()

    def printInfo(self, level, msg):
        if level <= self.printLevel:
            self.printFunction(msg)

    def beta_g(self, ind):
        return numpy.exp(self.x_part[ind][0] + self.w * self.v_part[ind][0] + 
                         self.c_beta * 0.25 * randn() *  self.log_beta_max)
    
    def _constraints_satisfied(self, x):
        OFE = int(numpy.exp( x[0] ))
        if OFE < self.OFE_budget_min :
            return False, "OFE_budget (%i) < OFE_budget_min (%i)" % (OFE, self.OFE_budget_min)
        if OFE > self.OFE_budget_max :
            return False, "OFE_budget (%i) > OFE_budget_max (%i)" % (OFE, self.OFE_budget_max)
        return self.CPV_validity_checks(x[1:], OFE)

    def continueOptimisation(self) :
        N = self.N
        n_dim = self.n_dim
        x_init_lb = self.x_init_lb 
        x_init_ub = self.x_init_ub
        if self.it == 0 : 
            self.printInfo(2, 'Starting new optimization, Generating Initial Population')
            self.x_part = []
            failCount = 0
            while len(self.x_part) < N:
                x = x_init_lb + rand(n_dim)*(x_init_ub- x_init_lb)
                candidate_design_acceptable, msg = self._constraints_satisfied(x)
                if candidate_design_acceptable:
                    self.x_part.append(x)
                else:
                    failCount = failCount + 1
                    self.printInfo(3 , 'Intializing population: failure count %i : constraints not satisfied : %s' % (failCount,msg))
                    if failCount > 1000:
                        raise RuntimeError, "Intializing population: failCount > 1000, check constraints!"
            self.evaluate_candidate_designs()
            for i, lg, x in zip(range(N),self.localGuides,self.x_part):
                if lg.N == 0:
                    raise RuntimeError,"lg.N == 0"
                    #self.printInfo(0, '  WARNING tMOPSO, local guide (%i of %i) has no entries, can occur when self.OFE_budgets <> None. OFE_budget %i CVPs %s ' % (i+1, N, numpy.exp(x[0]), ','.join(['%2.3f'%xv for xv in x[1:]])))
                    # scenario can occur where tMOPSO ask give f_min at 188 evals, for popsize of 188, but value specified in OFE bugets is 180,200. Given specified_OFE budget f_min assigned new budget of 200.Then CPV_manager does not return any output. Assigning bogus point to populate local guide rep.')
                    #lg.inspect_design(x, numpy.array([max(self.OFE_budgets),numpy.inf]))
                        #if i == 3:
                            #print('x[:4]',x[:4])
            self.v_part = [numpy.zeros(n_dim) for x in self.x_part]
            if self.PFA.N == 0 :
                raise RuntimeError , 'tMOPSO : could generate any feasible points aborting'
            self.it = 1
            if self.record_X_hist: self.X_history.append(numpy.array(self.x_part))
            if self.record_V_hist: self.V_history.append(numpy.array(self.v_part))
            self.printInfo(2, 'Finished generating Initail Population,  application layer evaluation budget used  %3.2f %%'%(100.0 * self.optAlg_evals_made /self.gammaBudget))
            self.post_iteration_function(self)

        while self.optAlg_evals_made < self.gammaBudget :
            #current_gen, total_gen, mutrate = self.it, self.max_it, 0.5 #shorthand for mutatation calc 
            for j in range(self.N) :
                candidate_design_acceptable = False
                failCount = 0
                x_j = self.x_part[j]
                v_j = self.v_part[j]
                while not candidate_design_acceptable:
                    #position update
                    v_pb=rand(n_dim)*(self.localGuides[j].best_design( f1 = self.beta_g(j))-x_j)
                    v_gb = rand(n_dim) * ( self.PFA.best_design( f1 = self.beta_g(j) ) - x_j )
                    v_c = numpy.zeros(n_dim)
                    v_c[0] = -0.5*( self.c_p+self.c_g ) *  self.w * v_j[0]
                    v = self.w * v_j + self.c_p*v_pb + self.c_g*v_gb + v_c
                    x = x_j + v
                    if self.constrain_to_initilization_bounds : #normally not used ...
                        lb_violated = x < x_init_lb
                        x[lb_violated] = x_init_lb[lb_violated]
                        v[lb_violated] = -v[lb_violated]
                        ub_violated = x > x_init_ub
                        x[ub_violated] = x_init_ub[ub_violated]
                        v[ub_violated] = -v[ub_violated]
                    candidate_design_acceptable, msg = self._constraints_satisfied(x)
                    if not candidate_design_acceptable:
                        failCount = failCount + 1
                        self.printInfo(3, '  Constraints not satisfied: gen. %i, pop. member %i, failureCount %i, msg: %s' % (self.it,j,failCount,msg))
                        if failCount > 10:
                            self.printInfo(3, '  Constraints not satisfied: failureCount %i which is > 10, re-intializing particle')
                            x = x_init_lb + rand(n_dim)*(x_init_ub- x_init_lb)
                            v = numpy.zeros(n_dim)
                            candidate_design_acceptable, msg = self._constraints_satisfied(x)
                        elif failCount > 100:
                            raise RuntimeError, "  generate x:  Constraints not satisfied for a failCount > 100, check constraints!"
                        
                self.v_part[j] = v
                self.x_part[j] = x
            self.evaluate_candidate_designs()
            self.it = self.it + 1
            self.printInfo(2, 'tMOPSO , it=%i complete, application layer evaluation budget used  %3.2f %%'%(self.it, 100.0 * self.optAlg_evals_made /self.gammaBudget))
            if self.saveInterval > 0 and self.it % self.saveInterval == 0 :
                if self.saveProgress : self.save(self.saveTo)
            if self.record_X_hist: self.X_history.append(numpy.array(self.x_part))
            if self.record_V_hist: self.V_history.append(numpy.array(self.v_part))
            self.post_iteration_function(self)
            if any( tc.satisfied(self) for tc in self.extra_termination_critea ):
                msgs = [ tc.msg for tc in self.extra_termination_critea if hasattr(tc,'msg') ]
                self.printInfo(1, 'tMOPSO terminating, %s' % (';'.join(msgs) ) )
                break

        #saving final status
        self.T_finish = datetime.datetime.now()
        if self.saveProgress : self.save(self.saveTo)

    def evaluate_candidate_designs(self):
        """
        evaluates CPVs, and updates the global Pareto front approximation, as well each particle own local Pareto Front approx.
        """
        maxSamples = sum(self.sampleSizes)
        max_OFE_x = lambda x: min( self.OFE_assessment_overshoot_function(numpy.exp(x[0])),
                                   self.OFE_budget_max )
        CPVs = [ CPV_evaluation_manager(
                x[1:], 
                self.optAlg, 
                self.OFE_budgets if self.OFE_budgets <> None else max_OFE_x(x) ,
                self.OFE_assessment_undershoot_function(numpy.exp(x[0])),
                max_OFE_x(x),
                maxSamples,
                self.printInfo,
                self.optAlg_addtoBatch,
                ) for x in self.x_part ]
        OFE_budget_stats = [] 
        no_s = 0 
        for i, repeats in enumerate(self.sampleSizes):
            OFE_budget_stats.append( [ c.max_OFE_budget() for c in CPVs ] )
            no_s = no_s + repeats
            if self.printLevel > 1:
                s1 = sum([c>0 for c in OFE_budget_stats[-1]])
                s2 = numpy.array([c for c in OFE_budget_stats[-1] if c>0]).mean() if s1 > 0 else 0.0
                self.printFunction('  evaluating/refining     %3i CPV sets \t    sample size %3i \t mean target OFE budget %6i' % (int(s1), no_s, int(s2)))
            for c in CPVs:
                if c.max_OFE_budget() > 0:
                    c.addtoBatch( repeats )
            self.optAlg_processBatch() #parrel batch processing code,
            results = []
            for c in CPVs:
                 if c.max_OFE_budget() > 0:
                     results.append( c.results( repeats ) )
                     self.optAlg_evals_made = self.optAlg_evals_made + c.OFEs_used_over_last_n_runs(n = repeats)
                 else:
                     results.append(None) #spacer
            if self.it == 0 and no_s < maxSamples: #construct intermedidate rep/archive
                PFA_interruption_checks = self.PFA.__class__()
                for r in results:
                    if r <> None:
                        for f1, f2_vals in zip(*r): #r = evals, fvals
                            PFA_interruption_checks.inspect_design( None, f1, f2_vals )
            else:
                PFA_interruption_checks = self.PFA
            for c, r, localGuide in zip(CPVs, results, self.localGuides):
                if r <> None: 
                    evals, fvals = r
                    #print('(evals, fvals)',evals, fvals)
                    if no_s < maxSamples:
                        if self.resampling_interruption_mode == 'reduce_max':
                            new_max_OFE_budget = 0
                            for f1, f2_vals in reversed( zip(evals, fvals) ):
                                if not PFA_interruption_checks.probably_dominates( f1, f2_vals, self.icl ):
                                    new_max_OFE_budget = f1
                                    break
                                elif self.paretoArchive_local_use_history_info:
                                    xv = numpy.array([ numpy.log(f1)] + c.CPVs.tolist())
                                    fv = numpy.array([ f1, f2_vals.mean() ])
                                    localGuide.inspect_design(xv, fv)
                            c.reduce_max_OFE_budget( new_max_OFE_budget )
                        elif self.resampling_interruption_mode == 'check_all' :
                            OFE_budgets_to_keep_refining = []
                            for f1, f2_vals in zip(evals, fvals):
                                if not PFA_interruption_checks.probably_dominates( f1, f2_vals, self.icl ):
                                    OFE_budgets_to_keep_refining.append(f1)
                                elif self.paretoArchive_local_use_history_info:
                                    xv = numpy.array([ numpy.log(f1)] + c.CPVs.tolist())
                                    fv = numpy.array([ f1, f2_vals.mean() ])
                                    localGuide.inspect_design(xv, fv)
                            c.set_new_OFE_budgets( OFE_budgets_to_keep_refining  )
                        else :
                            raise NotImplemented, "resampling_interruption_mode == '%s' not programmed yet" % self.resampling_interruption_mode
                    else:
                        for f1, f2_vals in  zip(evals, fvals):
                            xv = numpy.array([ numpy.log(f1)] + c.CPVs.tolist())
                            self.PFA.inspect_design(xv, f1, f2_vals)
                            if self.paretoArchive_local_use_history_info:
                                localGuide.inspect_design(xv,  numpy.array([f1,f2_vals.mean()]))
        self.evaluate_candidate_designs_stats.append(OFE_budget_stats)
        if not self.paretoArchive_local_use_history_info:
            for c,x,localGuide in zip(CPVs, self.x_part, self.localGuides):
                x_closest, f_closest = c.get_xv_fv_for_OFE_budget(numpy.exp(x[0]))
                localGuide.inspect_design(x_closest, f_closest)
        self.PFA_history.record(self.optAlg_evals_made , self.PFA)

    def save(self,fn) :
        "save current progess, so that the optimisation can be resumed, if interupted"
        if type(fn) == type('') :
            #print('saving functino to.. ' + fn)
            f = file(fn,'w')
        else :
            f = fn
        pickle.dump(self, f, protocol=1)
        if type(fn) == type('') :
            f.close()

    def plot_sampling_info(self, clrs = ['b','g','r']):
        import pylab
        wid = 0.9
        nS = len(self.sampleSizes)
        bar_plots = []
        for i in range(nS):
            h = [ sum( s_j > 0 for s_j in s[i] ) for s in self.evaluate_candidate_designs_stats]
            bp = pylab.bar( numpy.arange(len(h)) + i * wid / nS + 0.5 - (1-wid)/2, h, 
                            width=wid/nS, color=clrs[i % len(clrs)])
            bar_plots.append(bp[0])
        pylab.xlabel('tMOPSO itteration')
        pylab.ylabel('no. CPVs evaluated for')
        pylab.legend(bar_plots, ['sample size %i' % s for s in self.sampleSizes])
                
    def plot_target_OFE_distrubution(self, log, logx=True, clrs = ['b','g','r']):
        import pylab
        nS = len(self.sampleSizes)
        heights = []
        for i in range(1,nS):
            h_raw = sum([s[i] for s in self.evaluate_candidate_designs_stats],[])
            h = [ h_i for h_i in h_raw if h_i > 0 ]
            if logx:
                h = numpy.log(h)
            #print(h)
            n, bins, patches = pylab.hist( h , bins = 10, log=log,histtype='step', 
                                           normed = False, color=clrs[i % len(clrs)])
            heights.append(patches[0].xy[:,1].max())
        if logx:
            pylab.xlabel('log(target OFE)')
        else:
            pylab.xlabel('target OFE')

        pylab.ylabel('CPV focused on it')
        pylab.ylim( 0, max(heights) / 0.8 )
        pylab.legend(['refinement %i' % i for i in range(1,nS)])

    def plot_HV_hist(self, HV_bound=None, plotKey='g'):
        'plots the Hyper volume history, if HV_bound is not specified, then the bound from the final pareto front approximation is used'
        import pylab
        if HV_bound == None:
            HV_bound = self.PFA.upper_bounds()
        HVs = self.PFA_history.map( lambda f: f.hyper_volume(HV_bound) )
        line = pylab.plot(numpy.arange(len(HVs)), HVs, plotKey)
        return line

    def __repr__(self):
        t_total = (self.T_finish - self.T_start).total_seconds()
        t_alg_being_tuned = self.optAlg.total_seconds() + self.optAlg_processBatch.total_seconds()
        t_tuner = t_total - t_alg_being_tuned
        return """< tMOPSO tuning optimization: gamma budget %3.2e (usage %3.2f %%)
tMOPSO swarm parameters : N %i, w %4.2f, c_p %4.2f, c_g %4.2f, c_beta %4.2f
sample size increments: %s, interruption confidence %4.2f %%
total time: %7.2fs   time tMOPSO: %7.2fs   overhead: %3.1f%% >""" \
            % (self.gammaBudget, 100.0 * self.optAlg_evals_made /self.gammaBudget,
               self.N, self.w, self.c_p, self.c_g, self.c_beta,
               str(self.sampleSizes), self.icl*100,
               t_total, t_tuner, (t_tuner/t_alg_being_tuned)*100)
            
            
def commaFormat(i):
    s = str(i)
    s_out = ''
    for j in range(len(s)):
        if j > 0 and j % 3 == 0:
            s_out = s[len(s)-1-j] + ',' + s_out
        else:
            s_out = s[len(s)-1-j] + s_out
    return s_out


