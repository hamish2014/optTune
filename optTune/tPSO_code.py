"""
tuning particle swarm optimiser (tPSO) for tuning optimization algorithms according to single objective.
tPSO is a stripped down version of tMOPSO, which only focuses on a single OFE budget only. 
"""
import numpy, math, pickle, datetime
from numpy.random import rand, randn
from copy import deepcopy
from scipy import stats
from optTune.tMOPSO_code.tMOPSO_module import CPV_evaluation_manager, commaFormat
from optTune import _timingWrapper, _passfunction

def to_stdout(text):
    "should not be needed in Python 3.0"
    print(text)

class _optAlg_fbest_CPV_evaluation_manager_interface:
    def __init__(self, optAlg_fbest):
        self.optAlg_fbest = optAlg_fbest
    def __call__(self, CPVs, OFE_budgets, randomSeed):
        assert len(OFE_budgets) == 1
        F_out = [self.optAlg_fbest(CPVs, OFE_budgets[0], randomSeed)]
        E_out = [OFE_budgets[0]]
        return numpy.array(F_out), numpy.array(E_out)

class _addtoBatch_interface:
    def __init__(self, addtoBatch):
        self.addtoBatch_f = addtoBatch
    def __call__(self, CPVs, OFE_budgets, randomSeed):
        assert len(OFE_budgets) == 1
        self.addtoBatch_f(CPVs, OFE_budgets[0], randomSeed)

class function_evaluation_Noise :
    def __init__(self, fvals_sample = [numpy.inf] ) : 
        "pass the function value, plus the value of the inequality constaints, equality constraints (tol)"
        self.fv = fvals_sample 
    def __eq__(self,fe):
        # to be a 100% correct, but should be very slow...
        return (self.fv == fe.fv).all()
    def __lt__(self,fe):
        #contraint handling remove as optprob.respawn until valid should be on
        return numpy.mean(self.fv) < numpy.mean(fe.fv)
    def __repr__(self):
        return "< function_evaluation_Noise  %s >" % self.fv
    def copy(self) :
        return deepcopy(self)
    def update(self, fvals) :
        if numpy.mean(fvals) <= numpy.mean(self.fv) :
            self.fv = fvals
            return True
        else:
            return False
    def probability_of_being_better_than(self, fvals):
        #print('probability_of_being_better_than',self.fv, fvals)
        try:
            u, prob = stats.mannwhitneyu(self.fv, fvals)
            if numpy.mean(self.fv) < numpy.mean(fvals):
                return 1 - prob
            else:
                return prob
        except ValueError, msg:
            if str(msg) == 'All numbers are identical in amannwhitneyu': #then odds problem solved to machine percision, so take already evaluated sample
                return len(self.fv) > len(fvals)
            else:
                raise ValueError, msg

class tPSO:
    def __init__(self,
                 optAlg, 
                 OFE_budget, 
                 CPV_lb, 
                 CPV_ub, 
                 CPV_validity_checks,
                 sampleSizes,
                 gammaBudget, 
                 N = 15, 
                 w = 0.4,
                 c_g = 2.0,
                 c_p = 2.0,                
                 resampling_interruption_confidence = 0.9, # refered to as alpha in tMOPSO paper
                 constrain_to_initilization_bounds = False,
                 saveTo = None,
                 saveInterval = 10,
                 printFunction = to_stdout,
                 printLevel = 2,
                 addtoBatch = _passfunction,
                 processBatch = _passfunction,
                 post_iteration_function = _passfunction,
                 record_X_hist = True,
                 record_V_hist = True):
        """
Single-objective particle swarm optimisation for tuning optimization algorithms to single OFE  budget

     Required Args
        * optAlg - function called with args (CPVs, OFE_budget, randomSeed) and return best solution error, (i.e f_found_opt - f_min)
        * OFE_budget - OFE budget under which optAlg is tuned
        * CPV_lb - initialization lower bound for CPV tuning, i.e.  numpy.array([CPV_1_init_lb, CPV_2_init_lb, ...])
        * CPV_ub - numpy.array([CPV_1_init_ub, CPV_2_init_ub, ...])
        * CPV_validity_checks - function used to check validity of candidate CPVs. Usage CPV_validity_checks(CPV_array, OFE_budget) returns tuple (valid, msg) where msg is string of why invalid. Should be a cheap function, as candidate CPVs are regenerated if they do not satisfy it. Use to prevent negative population sizes, populations size larger then OFE_budget checks, etcetera.
        * CPV_ineq - vector which returned the amount by which each inequality constraint is violated. This should be a cheap function, as candidate CPVs are regenerated if they do not satisfy it. Use to prevent negative population sizes etcetera. Input vector of CPVs like CPV_lb 
        * sampleSizes - sample sizes used to generate and refined CPV utility values. For example if the sampleSizes are [5,15,30] all candidate CPVs will be sampled 5 times, the possible not dominated CPVs are then sampled another 15 times, and if still promising another 30 times. CPV which returned performances are therefore averaged over 50 independent instances.
        * gammaBudget - the number of application layer evaluations (evaluation of the function optAlg optimizes) allocated for the tuning. NB include repeats, i.e. assessing optAlg for on OFE budget of 100 at 5 repeats, counts as a gamma of 500.

     Optional Args
        * saveTo - filename used to save optimization progress (one save per itteration), use None to disable saving
        * N - tMOPSO population size
        * w - tMOPSO particle inertia factor
        * c_g - parameter controling the attraction towards the global guide
        * c_p - parameter controling the attraction towards the particles personal guide
        * resampling_interruption_confidence - re-sampling interuption confindence level used by noise stratergy
        * printLevel - 0 no printing, 1 only warnings, 2 overall info, 3 lots of info, 4 verbose (not implemented) 
        * addtoBatch - args for optAlg, passed to this function before optAlg is called, 
        * processBatch - function is called after all addtoBatch_function have been called. If used then optAlg, should be retrieve solutions from this functions results
        * post_iteration_function - at the end of each iteration this function is called with tPSO instance as the only arg.
        """
        self.T_start = datetime.datetime.now()
        self.initializationArgs = locals()
        # required parameters
        self.optAlg = _timingWrapper(_optAlg_fbest_CPV_evaluation_manager_interface(optAlg))
        self.x_init_lb = CPV_lb
        self.x_init_ub = CPV_ub
        self.CPV_validity_checks = CPV_validity_checks
        self.n_dim = len(self.x_init_lb) 
        self.N = N
        self.w  = w
        self.sampleSizes = sampleSizes
        self.OFE_budget = OFE_budget
        self.gammaBudget = gammaBudget
        # optional parameters
        self.c_g = c_g
        self.c_p = c_p
        self.icl = resampling_interruption_confidence
        self.constrain_to_initilization_bounds = constrain_to_initilization_bounds
        self.saveInterval = saveInterval        
        self.printFunction = printFunction
        self.printLevel = printLevel
        if saveTo <> None :
            self.saveProgress = True
            self.saveTo = saveTo
        else :
            self.saveProgress = False
        self.optAlg_addtoBatch = _addtoBatch_interface(addtoBatch)
        self.optAlg_processBatch = _timingWrapper(processBatch)
        self.post_iteration_function =  post_iteration_function
        self.record_V_hist = record_V_hist
        self.record_X_hist = record_X_hist
        # additional stuff
        self.it = 0
        self.optAlg_evals_made = 0
        self.evaluate_candidate_designs_stats = []
        self.f_part = [ function_evaluation_Noise() for i in range(N) ]
        self.f_best = function_evaluation_Noise()
        self.f_best_history = []
        self.V_history = []
        self.X_history = []
        self.continueOptimisation()

    def printInfo(self, level, msg):
        if level <= self.printLevel:
            self.printFunction(msg)

    def _constraints_satisfied(self, x):
        return self.CPV_validity_checks(x, self.OFE_budget) #return (valid, reason)

    def continueOptimisation(self) :
        N = self.N
        n_dim = self.n_dim
        x_init_lb = self.x_init_lb
        x_init_ub = self.x_init_ub
        if self.it == 0 : 
            self.printInfo(2, 'Starting new optimization, Generating Initial Population')
            self.x_part = []
            self.x_pb = []
            failCount = 0
            while len(self.x_part) < N:
                x = x_init_lb + rand(n_dim)*(x_init_ub- x_init_lb)
                x_valid, x_valid_msg = self._constraints_satisfied(x)
                if x_valid:
                    self.x_part.append(x)
                    self.x_pb.append(x.copy())
                else:
                    failCount = failCount + 1
                    self.printInfo(3 , 'Intializing population: CPV_validity_checks not satisfied, failure count %i, msg %s' % (failCount, x_valid_msg))
                    if failCount > 1000:
                        raise RuntimeError, "Intializing population: failCount > 1000, check constraints!"
            self.evaluate_candidate_designs()
            self.v_part = [numpy.zeros(n_dim) for x in self.x_part]
            self.it = 1
            if self.record_X_hist: self.X_history.append(numpy.array(self.x_part))
            if self.record_V_hist: self.V_history.append(numpy.array(self.v_part))
            self.printInfo(2, 'Finished generating Initail Population,  application layer evaluation budget used  %3.2f %%'%(100.0 * self.optAlg_evals_made /self.gammaBudget))
            if self.saveInterval > 0 and self.it % self.saveInterval == 0 :
                if self.saveProgress : self.save(self.saveTo)
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
                    v_pb = rand(n_dim) * ( self.x_pb[j] - x_j )
                    v_gb = rand(n_dim) * ( self.x_gb    - x_j )
                    v = self.w * v_j + self.c_p*v_pb + self.c_g*v_gb
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
                        self.printInfo(3, '  CPV_validity_checks not satisfied respawning: gen. %i, pop. member %i, failureCount %i, msg  %s' % (self.it,j,failCount,msg))
                        if failCount > 10:
                            self.printInfo(3, '  CPV_validity_checks failCount %i which is > 10, re-intializing particle' % failCount)
                            x = x_init_lb + rand(n_dim)*(x_init_ub- x_init_lb)
                            v = numpy.zeros(n_dim)
                            candidate_design_acceptable, msg = self._constraints_satisfied(x) 
                        elif failCount > 100:
                            raise RuntimeError, "  generate x:  optProb.g_respawn_x failCount > 100, check constraints!"                        
                self.v_part[j] = v
                self.x_part[j] = x
            self.evaluate_candidate_designs()
            self.it = self.it + 1
            self.printInfo(2, 'tPSO , it=%i complete, application layer evaluation budget used  %3.2f %%, f_best %1.3e'%(self.it, 100.0 * self.optAlg_evals_made /self.gammaBudget, 
                                                                                                                     numpy.mean(self.f_best.fv)))
            if self.saveInterval > 0 and self.it % self.saveInterval == 0 :
                if self.saveProgress : self.save(self.saveTo)
            if self.record_X_hist: self.X_history.append(numpy.array(self.x_part))
            if self.record_V_hist: self.V_history.append(numpy.array(self.v_part))
            self.post_iteration_function(self)
        #saving final status
        self.T_finish = datetime.datetime.now()
        if self.saveProgress : self.save(self.saveTo)

    def evaluate_candidate_designs(self):
        """
        evaluates CPVs, and updates the global Pareto front approximation, as well each particle own local Pareto Front approx.

        y = [CPV_1, CPV_2, ..., OFE_budget, Repeats, OFE_budget_target]
        """
        #OFE_budget_max = numpy.exp(self.optProb.populating_ub[0])
        maxSamples = sum(self.sampleSizes)

        CPVs = [ CPV_evaluation_manager( x,
                                         self.optAlg, 
                                         numpy.array([self.OFE_budget]),
                                         0,
                                         self.OFE_budget,
                                         maxSamples,
                                         self.printFunction,
                                         self.optAlg_addtoBatch)
                 for x in self.x_part ]
        OFE_budget_stats = [] 
        no_s = 0 
        for i, repeats in enumerate(self.sampleSizes):
            OFE_budget_stats.append( [ c.max_OFE_budget() for c in CPVs ] )
            no_s = no_s + repeats
            if self.printLevel > 1:
                s1 = sum([c>0 for c in OFE_budget_stats[-1]])
                self.printFunction('  evaluating/refining     %3i CPV sets \t    sample size %3i' % (int(s1), no_s))
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
            if self.it == 0 and no_s < maxSamples: #construct intermedidate f_best
                f_best_Temp = function_evaluation_Noise()
                for r in results:
                    if r <> None:
                        fvals_r = r[1][-1:][0] #disregard history information
                        f_best_Temp.update(fvals_r)
                #print(f_best_Temp)
            for c, r, ind in zip(CPVs, results, range(self.N)):
                if r <> None:
                    fvals_r = r[1][-1:][0] #disregard history information
                    #print('(evals, mean_fvals)',evals, mean_fvals)
                    if no_s < maxSamples:
                        f_best = self.f_best if self.it > 0 else f_best_Temp
                        if f_best.probability_of_being_better_than(fvals_r) > self.icl:
                            c.reduce_max_OFE_budget(0)
                            if self.f_part[ind].update(fvals_r):
                                self.x_pb[ind] = self.x_part[ind].copy()
                    else:
                        if self.f_best.update(fvals_r):
                            self.x_gb = self.x_part[ind].copy()
                        if self.f_part[ind].update(fvals_r):
                            self.x_pb[ind] = self.x_part[ind].copy()
                        
        self.evaluate_candidate_designs_stats.append(OFE_budget_stats)
        self.f_best_history.append(deepcopy(self.f_best))

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

    def application_layer_eval_hist(self):
        '''
        returns the cumalatrive number of application layer evaluation made after each iterations
        '''
        eval_hist = []
        e = 0
        for target_OFE_budgets in self.evaluate_candidate_designs_stats:
            for i, repeats in enumerate(self.sampleSizes):
                e = e + sum(target_OFE_budgets[i])*repeats
            eval_hist.append(e)
        assert len(eval_hist) == len(self.f_best_history)
        return eval_hist

    def f_best_after_given_optAlg_evals(self, optAlg_evals):
        '''
        Return a f_best after $optAlg_evals evals of the problem optAlg is optimizing.
        '''
        #print(self)
        eval_hist = self.application_layer_eval_hist()
        ind = 0
        if eval_hist[ind] > optAlg_evals:
            raise ValueError,"optTune used more evals in its first itteration (%i) then requested (%i) " % (eval_hist[ind],optAlg_evals)
        while ind + 1 < len(eval_hist) and eval_hist[ind + 1] < optAlg_evals:
            ind = ind + 1
        if ind +1 == len(eval_hist) and eval_hist[ind] < optAlg_evals:
            self.printInfo(1,'WARNING: rep_after_given_optAlg_evals, optTune used %s optAlg_evals, requested %s' %  (commaFormat(eval_hist[ind]), commaFormat(optAlg_evals)))
        self.printInfo(2,'Returning the %ith f_best (out of total of %i approximations)' % (ind+1, len(self.f_best_history)))
        return self.f_best_history[ind]

    def __repr__(self):
        t_total = (self.T_finish - self.T_start).total_seconds()
        t_alg_being_tuned = self.optAlg.total_seconds() + self.optAlg_processBatch.total_seconds()
        t_tuner = t_total - t_alg_being_tuned
        return """< tPSO tuning optimization: gamma budget %3.2e (usage %3.2f %%)
tPSO swarm parameters : N %i, w %4.2f, c_p %4.2f, c_g %4.2f
sample size increments: %s, interruption confidence %4.2f %%
total time: %7.2fs   time tPSO: %7.2fs   overhead: %3.1f%% >""" \
            % (self.gammaBudget, 100.0 * self.optAlg_evals_made /self.gammaBudget,
               self.N, self.w, self.c_p, self.c_g, 
               str(self.sampleSizes), self.icl*100,
               t_total, t_tuner, (t_tuner/t_alg_being_tuned)*100)
            
