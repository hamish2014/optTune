"""
Multi-objective tuning algorithm (MOTA)
"""
import numpy, math, pickle, datetime
from numpy.random import rand, randint, randn
from numpy import log
from numpy.linalg import norm
from copy import deepcopy
from optTune.paretoArchives import PFA_history_recorder
from paretoArchive2D_with_simlurity_checking import paretoArchive2D_polynimal_simularity_checking
from CPV_evaluation_manager import CPV_evaluation_manager
from subproblems import MOTA_subproblem,  MOTA_subproblem_aux
from optTune import linearFunction, _timingWrapper, _passfunction

def to_stdout(text):
    "should not be needed in Python 3.0"
    print(text)

class MOTA:
    """Multi-objective tuning algorithm"""
    def __init__(self, 
                 objectiveFunctions, 
                 subproblems,
                 CPV_lb, 
                 CPV_ub, 
                 CPV_validity_checks,
                 sampleSizes,

                 DE_F=2,
                 DE_Cr=0.7,
                 OFE_purtibation_factor = 0.2,
                 OFE_assessment_overshoot_function = linearFunction(1.2, 100),
                 OFE_assessment_undershoot_function = linearFunction(0.0, 0),

                 resampling_interruption_confidence = 0.80,
                 resampling_interruption_mode = 'reduce_max',

                 boundConstrained=False,
                 process_batch = _passfunction,
                 saveTo = None, 
                 saveInterval = -1,
                 printFunction = to_stdout, 
                 printLevel = 2,
                 record_X_hist = True,
                 normalize_objective_values = True,
                 post_iteration_function = _passfunction,

                 DE_F_vector_mutation=True,

                 polynomial_similarity_mode = -1,
                 simularity_exploitation_factor = 2, 
                 simularity_fitting_threshold_ratio = 0.2,
                 ):
        """
     Required Args
        * objectiveFunctions - contains the list of tuning objective functions. Each tuning objective (f) takes 3 arguments (CPV_array, assessment_OFE_budgets, randomSeed). f must returns two lists. The first list should return the utility measure such as the solution error (i.e f_found_opt - f_min) and should be decreasing, and the second list the number of objective function evaluations (OFEs) used in order to determine each element in the first list and should be increasing, and should if possible match the assessment_OFE_budgets (not required though). These lists can be thought of as the optimizer's history. If an objective function has the addtoBatch attribute, each (CPV_array, assessment_OFE_budgets, randomSeed) about to be evaluated in passed to that function. Then the process batch_function is called, after which the objective function is called with the same input given to addtoBatch.
        * subproblems - list of MOTA_sub_problems.
        * CPV_lb - initialization lower bound for CPV tuning, i.e.  numpy.array([CPV_1_init_lb, CPV_2_init_lb, ...])
        * CPV_ub - numpy.array([CPV_1_init_ub, CPV_2_init_ub, ...])
        * CPV_validity_checks - function used to check validity of candidate CPVs. Usage CPV_validity_checks(CPV_array, OFE_budget) returns tuple (valid, msg) where msg is string of why invalid. Should be a cheap function, as candidate CPVs are regenerated if they do not satisfy it. Use to prevent negative population sizes, populations size larger then OFE_budget checks, etcetera.
        * sampleSizes - sample sizes used to generate and refined CPV utility values. For example if the sampleSizes are [5,15,30] all candidate CPVs will be sampled 5 times, the possible not dominated CPVs are then sampled another 15 times, and if still promising another 30 times. CPV which returned performances are therefore averaged over 50 independent instances.

     Optional Args
        * resampling_interruption_confidence - confidence level for interrupting the sample gathering process when performing re-sampling. Must be greater than 0.5 and less than or equal to 1
        * resampling_interruption_mode - 'reduce_max' or 'check_all'. 
        * OFE_purtubation_factor - when generating new candidate CPVs, OFEs close to the target OFE budget + 0.25 * rand_gausion* OFE_purtubation_factor * (log(max OFE budget)-log(min OFE budget)) are considered
        * OFE_assessment_overshoot_function - when assessing a CPV tuple for a OFE budget of beta, this factor is used to control overshoot, beta_actual = OFE__assessment_undershot_function(beta)
        * OFE_assessment_undershoot_function - similar to OFE__assessment_overshoot_function except controls minimum value
        * DE_F - DE scaling factor
        * DE_F_vector_mutation - change 'DE_F*(x_1 - x_2)' to 'r()*DE_F*(x_1 - x_2)' where r is vector consting randomly generated elements between 0 and 1, using a uniform distibution. Recommended, else diversity problem at start of MOTA optimization.
        * DE_Cr - DE crossover probability 
        * boundConstrained - should CPV tuning search be bound constrainted between CPV_lb and CPV_ub
        * process_batch - function is called after all the add_to_batch_functions have been called. 
        * saveTo - filename used to save optimization progress (one save per itteration), use None to disable saving
        * saveInterval - number of iterations after which the progress of the tuning should be saved, negative numbers indicate do not saving during optimization.
        * printLevel - 0 no printing, 1 only warnings, 2 overall info, 3 lots of info, 4 verbose (not implemented) 
        * printFunction - output is parsed into this function
        * polynomial_similarity_mode - if an integer >= 1, then this CPV determines the order of the polynomial to be fitted for the similarity between polynomial fits of the paretoFronts (PFs) to construct the neighbourhood used for generating new candidate designs. A value of 0 indicates override_inter_PF_normalization, and use the update neighborhood for generating new candidate designs. A value of -1 indicates override_inter_PF_normalization, and every subproblem for generating new candidate designs. override_inter_PF_normalization - do not perform scaling correct based upon the fitted polynomials, during candidate CPV generation when using a CPV from a different PF to the target subproblem.
        * simularity_exploitation_factor - controls how much information sharing take place between subproblems as a function of there similarity between their PFAs. Example values, 10 sharing only when very simular PFAs, 0 share equally regardless of simularity, -5 share with PFAs most dissimular. 
          If function then, simularity_explotation_factor_iteration = simularity_exploitation_factor (subproblem.gamma / subproblem.gammaBudget)
        * simularity_fitting_threshold_ratio - set the maximum scaling range, as (CPV_ub - CPV_lb)*simularity_scaling_threshold
        * normalize_objective_values - normalize objective values so that utopia point is approximately 0 and nadir point is approximately 1 for all objectives
        * post_iteration_function - at the end of each iteration this function is called with MOTA instance as the only arg.

"""
        self.T_start = datetime.datetime.now()
        self.objectiveFunctions = [ _timingWrapper(f) for f in objectiveFunctions ]
        assert all( isinstance(sp,MOTA_subproblem) for sp in subproblems )
        assert any( sp.active() for sp in subproblems )
        self.subproblems = subproblems
        #if len(subproblems) < 4:
        #    raise ValueError, "MOTA requires at least 4 subproblems. Consider using the subproblem duplicate function. i.e. sp.duplicate(5)"
        self.n_f = len(objectiveFunctions)
        self.CPV_lb = CPV_lb
        self.CPV_ub = CPV_ub
        self.n_x = len(CPV_lb) + 1
        self.CPV_validity_checks = CPV_validity_checks
        self.sampleSizes = sampleSizes
        # optional parameters
        self.resampling_interruption_confidence = resampling_interruption_confidence
        assert resampling_interruption_mode in ['reduce_max','check_all']
        self.resampling_interruption_mode = resampling_interruption_mode
        self.simularity_exploitation_factor = simularity_exploitation_factor
        self.simularity_threshold = simularity_fitting_threshold_ratio * (CPV_ub - CPV_lb)
        self.OFE_purtibation_factor = OFE_purtibation_factor
        self.OFE_assessment_overshoot_function  = OFE_assessment_overshoot_function 
        self.OFE_assessment_undershoot_function = OFE_assessment_undershoot_function
        self.DE_F = DE_F
        self.DE_F_vector_mutation = DE_F_vector_mutation
        self.DE_Cr = DE_Cr
        self.boundConstrained = boundConstrained
        self.process_batch = _timingWrapper(process_batch)
        self.printFunction = printFunction
        self.printLevel = printLevel
        self.saveTo = saveTo
        self.saveProgress = saveTo <> None
        self.saveInterval = saveInterval
        self.record_X_hist = record_X_hist
        assert polynomial_similarity_mode in [-1, 0, 1, 2, 3, 4]
        self.polynomial_similarity_mode =  polynomial_similarity_mode
        if self.polynomial_similarity_mode >= 1 :
            for s in self.subproblems:
                s.PFA.poly_fit_order = polynomial_similarity_mode
        self.normalize_objective_values = normalize_objective_values
        self.post_iteration_function = post_iteration_function
        # initialization
        self.it = 0
        self.CPV_min_changes = numpy.abs( CPV_ub - CPV_lb ) / 10**6
        self.transform_utopia_point  =  numpy.ones(len(objectiveFunctions)) * numpy.inf
        self.transform_nadir_point  = -numpy.ones(len(objectiveFunctions)) * numpy.inf
        if record_X_hist:
            for sp in self.subproblems:
                sp.X_history = []
        self.evaluate_candidate_designs_stats = []
        self.continueOptimisation()
                            
    def printInfo(self, level, msg):
        if level <= self.printLevel:
            self.printFunction(msg)

    def _sp_bounds(self, subproblem):
        lb = numpy.array( [numpy.log(subproblem.min_target_OFE_budget)] + self.CPV_lb.tolist() )
        ub = numpy.array( [numpy.log(subproblem.max_target_OFE_budget)] + self.CPV_ub.tolist() )
        return lb, ub

    def _generate_random_valid_x(self, subproblem, max_tolerated_failures=10):
        lb, ub = self._sp_bounds(subproblem)
        failCount = 0
        candidate_design_acceptable = False
        while not candidate_design_acceptable:
            x = lb + rand(len(lb))*(ub - lb)
            candidate_design_acceptable, msg  = self._constraints_satisfied( x,  subproblem)
            if not candidate_design_acceptable:
                failCount = failCount + 1
                self.printInfo(3 , 'Generating random valid x: CPV_ineq not satisfied, failure count %i : %s' % (failCount, msg))
                if failCount > max_tolerated_failures:
                    raise RuntimeError, "Generating random valid x: failCount > max_tolerated_failures (%i > %i), check constraints! Last failure reason : %s" % (failCount, max_tolerated_failures, msg)
        return x

    def _constraints_satisfied(self, x, subproblem):
        OFE = int(numpy.exp( x[0] ))
        if OFE < subproblem.min_target_OFE_budget :
            return False, "OFE_budget (%i) < subproblem.min_target_OFE_budget" % OFE
        if OFE > subproblem.max_target_OFE_budget :
            return False, "OFE_budget (%i) > subproblem.max_target_OFE_budget" % OFE
        return self.CPV_validity_checks(x[1:], OFE)
    
    def continueOptimisation(self):
        while any( s.active() for s in self.subproblems ):
            for sp_index, sp in enumerate(self.subproblems) :
                if not sp.active():
                    continue
                if not sp.initialized :
                    self.printInfo(2, '  generating initial design: subproblem %i/%i' % (sp_index+1, len(self.subproblems)))
                    sp.x_c = self._generate_random_valid_x(sp)
                elif len(sp.PFA.designs)>1 and sp.PFA.lower_bound == sp.PFA.upper_bound:
                    self.printInfo(2, '  OFE_budget collapse subproblem %i : %i designs all targeting an OFE budget of %i, %sgenerating random CPV tuple' %  (sp_index+1,  len(sp.PFA.designs), sp.PFA.lower_bound, '(sp %i is aux subproblem) '%(sp_index + 1) if isinstance(sp,MOTA_subproblem_aux) else ''))
                    sp.x_c = self._generate_random_valid_x(sp)
                else:
                    delta_B = log( sp.max_target_OFE_budget ) - log( sp.min_target_OFE_budget )
                    if not hasattr(self.simularity_exploitation_factor,'__call__'):
                        psi = self.simularity_exploitation_factor
                    else:
                        psi = self.simularity_exploitation_factor( sp.gamma / sp.gammaBudget )
                    S_accumalation = [0]
                    for sp_i in self.subproblems:
                        if self.polynomial_similarity_mode > 0 : #use polynomial similarity to select subproblems to use in candidate generation
                            S_accumalation.append( S_accumalation[-1] + ( sp.PFA.simularity_to(sp_i.PFA,self.simularity_threshold)**psi if sp_i.initialized else 0 ) )
                        elif self.polynomial_similarity_mode == 0: # use update neighborhood as candidate generation neighorhood
                            S_accumalation.append( S_accumalation[-1] + ( 1.0 if sp_i.initialized and (sp_i in sp.updateNeighbours or sp_i == sp ) else 0 ))
                        elif self.polynomial_similarity_mode == -1:
                            S_accumalation.append( S_accumalation[-1] + ( 1.0 if sp_i.initialized else 0 ))                    
                    S_accumalation = numpy.array( S_accumalation ) # [ ... , ... ] < 1 returns False , but [ ... , ... ] < numpy.float64(4) , return boolean array
                    failCount = 0
                    candidate_design_acceptable = False
                    while not candidate_design_acceptable: #based upon the DE best/1/bin scheme
                        OFE_target = sp.target_OFE_budgets[ randint(len( sp.target_OFE_budgets)) ]
                        #generating mutant vector
                        r = [] #population indexes
                        x_r = []
                        fc2 = 0
                        while len(r) < 3:
                            r_i = sum( S_accumalation < rand()*S_accumalation[-1] ) -1
                            if True : #if r_i not in r:
                                r.append(r_i)
                                beta_r = numpy.exp( numpy.log(OFE_target) + 0.25*randn()*self.OFE_purtibation_factor*delta_B )
                                if self.polynomial_similarity_mode > 0:
                                    x_r.append(self.subproblems[r_i].PFA.recommend_for(sp.PFA, beta_r))
                                else:
                                    x_r.append(self.subproblems[r_i].PFA.best_design( f1 = beta_r) )
                            else :
                                fc2 = fc2 + 1 
                                if fc2 > 28:
                                    raise RuntimeError, "after 28 tries, still unable to generate 3 unique population indexes for DE mutant vector generation"
                        if not self.DE_F_vector_mutation:
                            x_m = x_r[0] +                     self.DE_F * ( x_r[1] - x_r[2] )
                        else:
                            x_m = x_r[0] + rand(len(x_r[0])) * self.DE_F * ( x_r[1] - x_r[2] )
                        
                        #print('      norm(x_b[1:] - x_m[1:]) %e' % (norm(x_base[1:] - x_m[1:])))
                        # crossover
                        x_base = sp.PFA.best_design( f1 = OFE_target) #mutation vector
                        x = x_base.copy()
                        crossover_inds = rand(len(x)) < self.DE_Cr
                        crossover_inds[randint(1,len(x))] = True  #index forced to change, low limit of 1 inserted as to force a dimension other the target OFE_budget to change!
                        x[crossover_inds] =  x_m[crossover_inds] 
                        x[0] = numpy.log(OFE_target) #for OFE target budget
                        #/crossover                        
                        if self.boundConstrained : #repair
                            x_lb, x_ub = _sp_bounds(sp)
                            lb_violated = x < x_lb
                            x[lb_violated] = x_lb[lb_violated]
                            ub_violated = x > x_ub
                            x[ub_violated] = x_ub[ub_violated]
                        candidate_design_acceptable, msg = self._constraints_satisfied( x, sp )
                        if candidate_design_acceptable: #interms of constraints, then check if any change between x_base and x_candidate happened
                            if (x_base[1:] - x[1:] < self.CPV_min_changes).all() : #ignore OFE dimension
                                msg = '(x_base[1:] - x[1:] < self.CPV_min_changes).all(), therefore marking x_candidate for regeneration'
                                candidate_design_acceptable = False

                        if not candidate_design_acceptable:
                            failCount = failCount + 1
                            self.printInfo(3, '  subproblem %i : generating x_c failed (count %i) %s' % (sp_index+1,  failCount, msg))
                            if failCount > 10:
                                self.printInfo(2, '  subproblem %i/%i : generating x_c failureCount %i > 10, assigning valid random values, last failure message %s' % (sp_index+1, len(self.subproblems),failCount, msg))
                                x = self._generate_random_valid_x(sp)
                                candidate_design_acceptable = True
                        if self.printLevel > 2:
                            self.printFunction('   candidate decision vector generation info, subproblem_no %i:' % sp_index)
                            self.printFunction('        target OFE budget %i' %   OFE_target) 
                            for i in range(3):
                                self.printFunction('      x_r%i OFEs %i CPVs %s' % (i, numpy.exp(x_r[i][0]), str(x_r[i][1:]).replace('\n','')) )
                            self.printFunction('      x_m  OFEs %i CPVs %s' % (numpy.exp(x_m[0]), str(x_m[1:]).replace('\n','')) )
                            self.printFunction('      x_b  OFEs %i CPVs %s' % (numpy.exp(x_base[0]), str(x_base[1:]).replace('\n','')) )
                            self.printFunction('      norm(x_b[1:] - x_m[1:]) %e' % (norm(x_base[1:] - x_m[1:])))
                            self.printFunction('      x_c  OFEs %i CPVs %s (valid %s)' % (numpy.exp(x[0]),str( x[1:]).replace('\n',''), candidate_design_acceptable) )
                    sp.x_c = x

            self.evaluate_candidate_designs()
            self.it = self.it + 1
            if self.printLevel > 1:
                gammaUsed = sum(sp.gamma for sp in self.subproblems)
                gammaBudget = sum(sp.gammaBudget for sp in self.subproblems)
                self.printInfo(2, 'MOTA , it %i complete, gamma budget used %3.2f%%' % (self.it, 100.0*gammaUsed/gammaBudget))
            if self.saveInterval > 0 and self.it % self.saveInterval == 0 :
                if self.saveProgress : self.save(self.saveTo)
            self.post_iteration_function(self)
        #saving final status
        self.T_finish = datetime.datetime.now()
        if self.saveProgress : self.save(self.saveTo)

    def evaluate_candidate_designs(self):
        """
        evaluates CPVs, and updates the global Pareto front approximation, as well each particle own local Pareto Front approx.
        """
        maxSamples = sum(self.sampleSizes)
        evalManagers = []
        for sp in self.subproblems:
            if hasattr(sp, 'x_c'):
                evalManagers.append( CPV_evaluation_manager(
                        sp, 
                        self.objectiveFunctions,
                        sp.x_c[1:],
                        self.OFE_assessment_undershoot_function(numpy.exp(sp.x_c[0])),
                        self.OFE_assessment_overshoot_function(numpy.exp(sp.x_c[0])),
                        maxSamples,
                        self.printInfo
                        ) )
                if self.record_X_hist: 
                    sp.X_history.append( sp.x_c )
                del sp.x_c
        OFE_budget_stats = [] 
        no_s = 0 
        for i, repeats in enumerate(self.sampleSizes):
            OFE_budget_stats.append( [ em.max_target_OFE_budget() for em in evalManagers] ) 
            no_s = no_s + repeats
            if self.printLevel > 1: 
                s1 = sum([e>0 for e in OFE_budget_stats[-1]])
                s2 = numpy.mean([e for e in OFE_budget_stats[-1] if e>0]) if s1 > 0 else 0.0
                active_subproblems = len(numpy.unique(id(em.subproblem) for em in evalManagers))
                self.printInfo(2,'  evaluating/refining     %3i CPV sets for %i/%i subproblems \t    sample size %3i \t mean target OFE budget %6i' % 
                                   (int(s1), active_subproblems, len(self.subproblems),no_s, int(s2)))
            for em in evalManagers:
                em.addtoBatch( repeats )
            self.process_batch() #parrel batch processing code,
            for em in evalManagers:
                em.update( repeats )
            results = [ em.results() for em in evalManagers ]

            for subproblem in self.subproblems: #for construction of initial temporary PFAs
                subproblem.flush_PFA_if_not_initialized(include_neighbours=True)

            if no_s < maxSamples:
                for em, R in zip(evalManagers, results) :#construction of temporary PFAs continued
                    em.subproblem.update_PFA_if_not_initialized(R[0], R[1])
                for em, R in zip(evalManagers, results) :
                    if self.resampling_interruption_mode == 'reduce_max':
                        em.reduce_max_target_OFE_budget( em.subproblem.largest_OFE_budget_not_likely_to_be_dominated( R[0], R[1], self.resampling_interruption_confidence ))
                    elif self.resampling_interruption_mode == 'check_all':
                        em.reduce_target_OFE_budgets( em.subproblem.OFE_budgets_not_likely_to_be_dominated( R[0], R[1], self.resampling_interruption_confidence ))
                    else: 
                        raise NotImplemented, "resampling_interruption_mode == %s not implemented" % self.resampling_interruption_mode
            else:
                if self.normalize_objective_values:
                    self.update_objective_normalization_transformations(results)
                for em, R in zip(evalManagers, results):
                    em.subproblem.update_PFAs(R[0], R[1], em.CPVs.tolist())

        for em in evalManagers:
            em.subproblem.add_gamma_usage( em.gamma_usage() )
        self.evaluate_candidate_designs_stats.append(OFE_budget_stats)
        for sp in self.subproblems:
            sp.update_PFA_history()

    def update_objective_normalization_transformations(self, results, adjustment_threshold= 0.1):
        '''
        if scaling is enabled then scale objective values between utopia and nadir points
        should only be run when no_of_samples = maxSamples

        subproblem transforms objective values (z) to 
           z' = m*z + c , 
        where the uptopia point is z' is \vec{0} and the nadir point is \vec{1}

        The logic of this procedue is based on the assumptio that the worst non-dominated objective values i.e. the nadir point values will occur at low OFE budgets. This assumption is expected to be valid for most cases, and no effect the results too much if the assumptions invalid.
        '''
        lb = self.transform_utopia_point
        ub = self.transform_nadir_point
        old_lb = lb.copy()
        old_ub = ub.copy()
        for B, U in results:
            if len(U) > 0:
                lb = numpy.where( U[-1].mean(axis=1) < lb,  U[-1].mean(axis=1), lb)
                ub = numpy.where( U[0].mean(axis=1) > ub,  U[0].mean(axis=1), ub ) 
        m = numpy.ones(len(lb))
        c = numpy.zeros(len(ub))
        def isan(v):#is a number
            return not numpy.isnan(v) and not numpy.isinf(v)
        update=False
        for i, old_lb_i, old_ub_i, new_lb_i, new_ub_i in zip(range(len(ub)), old_lb, old_ub, lb, ub):
            update_d_flag = 0 #update dimension flag
            if isan(new_lb_i) and isan(new_ub_i) and new_lb_i <> new_ub_i:
                if isan(old_lb_i) and isan(old_ub_i):
                    mp = 0.5 * (new_lb_i + new_ub_i)
                    delta = new_ub_i - new_lb_i
                    r_lb =  abs( old_lb_i - mp) / delta
                    r_ub =  abs( old_ub_i - mp) / delta
                    if abs(r_lb - 0.5) > adjustment_threshold or \
                            abs(r_ub - 0.5) > adjustment_threshold:
                        update_d_flag = 1
                    else:
                        update_d_flag = -1
                else:
                    update_d_flag = 1
            elif isan(old_lb_i) and isan(old_ub_i):
                update_d_flag = -1
            elif  new_lb_i == new_ub_i :
                self.printInfo(2, '  update_objective_normalization_transformations - not updating dimension %i, since new_lb_i == new_ub_i, new_lb_i = %s' % (i+1, new_lb_i))
            if update_d_flag  <> 0:
                if  update_d_flag == 1:
                    m[i] =   1.0/(new_ub_i - new_lb_i)
                    c[i] = -new_lb_i/(new_ub_i - new_lb_i)
                    assert new_lb_i <=  self.transform_utopia_point[i]
                    assert new_ub_i >=  self.transform_nadir_point[i]
                    self.transform_utopia_point[i] = new_lb_i
                    self.transform_nadir_point[i] = new_ub_i
                    update = True
                else: #update_d_flag == -1
                    m[i] =   1.0/(old_ub_i - old_lb_i)
                    c[i] = -old_lb_i/(old_ub_i - old_lb_i)
        #print('m %s' % m)
        #print('c %s' % c)
        if update:
            self.printInfo(2, '  MOTA , adjusting objective transforms')
            self.printInfo(2, ('    new utopia point  %s' % self.transform_utopia_point ).replace('\n',''))
            self.printInfo(2, ('    new nadir  point  %s' % self.transform_nadir_point ).replace('\n',''))

            for sp in self.subproblems:
                sp.adjust_objective_transform(m,c)


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

    def plot_target_OFE_distrubution(self, log, clrs = ['b','g','r']):
        import pylab
        nS = len(self.sampleSizes)
        heights = []
        for i in range(nS):
            h = sum([s['target_OFEs'][i] for s in self.evaluate_candidate_designs_stats],[])
            #print(h)
            n, bins, patches = pylab.hist( h , bins = 10, log=log,histtype='step', 
                                           normed = False, color=clrs[i % len(clrs)] )
            heights.append(patches[0].xy[:,1].max())
        pylab.xlabel('target OFE')
        pylab.ylabel('CPV focused on it')
        pylab.ylim( 0, max(heights) / 0.8 )
        pylab.legend(['refinement %i' % i for i in range(nS)])

    def __repr__(self):
        t_total = (self.T_finish - self.T_start).total_seconds()
        t_alg_being_tuned = sum(f.total_seconds() for f in self.objectiveFunctions) + self.process_batch.total_seconds()
        t_tuner = t_total - t_alg_being_tuned
        return """< MOTA tuning optimization: no. of subproblems %i, total gamma used %s
DE_F: %1.2f DE_Cr: %1.2f c_b: %1.2f  simularity_exploitation_factor=%s,
sample Sizes: %s interruption confidence level: %1.2f
total time: %7.2fs   time MOTA: %7.2fs   overhead: %3.1f%%>""" \
            % (len(self.subproblems),  commaFormat(sum(sp.gamma for sp in self.subproblems)),
               self.DE_F,  self.DE_Cr, self.OFE_purtibation_factor, self.simularity_exploitation_factor,
               str(self.sampleSizes), self.resampling_interruption_confidence,
               t_total, t_tuner, (t_tuner/t_alg_being_tuned)*100)
            
            
def commaFormat(i):
    s = str(int(i))
    s_out = ''
    for j in range(len(s)):
        if j > 0 and j % 3 == 0:
            s_out = s[len(s)-1-j] + ',' + s_out
        else:
            s_out = s[len(s)-1-j] + s_out
    return s_out


class RAND_MOTA(MOTA):
    """Random CPV generation using a uniform distrubution between CPV_lb adn CPV_ub"""
    def __init__(self, 
                 objectiveFunctions, 
                 subproblems,
                 CPV_lb, 
                 CPV_ub, 
                 CPV_validity_checks,
                 sampleSizes,
                 resampling_interruption_confidence = 0.80,
                 resampling_interruption_mode = 'reduce_max',
                 OFE_assessment_overshoot_function = linearFunction(1.2, 100),
                 OFE_assessment_undershoot_function = linearFunction(0.0, 0),
                 process_batch = _passfunction,
                 saveTo = None, 
                 saveInterval = -1,
                 printFunction = to_stdout, 
                 printLevel = 2,
                 record_X_hist = True,
                 normalize_objective_values = True,
                 post_iteration_function = _passfunction
                 ):
        """
        See MOTA help for information on parameters
        """
        self.T_start = datetime.datetime.now()
        self.objectiveFunctions = [ _timingWrapper(f) for f in objectiveFunctions ]
        assert all( isinstance(sp,MOTA_subproblem) for sp in subproblems )
        assert any( sp.active() for sp in subproblems )
        self.subproblems = subproblems
        #if len(subproblems) < 4:
        #    raise ValueError, "MOTA requires at least 4 subproblems. Consider using the subproblem duplicate function. i.e. sp.duplicate(5)"
        self.n_f = len(objectiveFunctions)
        self.CPV_lb = CPV_lb
        self.CPV_ub = CPV_ub
        self.n_x = len(CPV_lb) + 1
        self.CPV_validity_checks = CPV_validity_checks
        self.sampleSizes = sampleSizes
        # optional parameters
        self.resampling_interruption_confidence = resampling_interruption_confidence
        assert resampling_interruption_mode in ['reduce_max','check_all']
        self.resampling_interruption_mode = resampling_interruption_mode
        self.OFE_assessment_overshoot_function  = OFE_assessment_overshoot_function 
        self.OFE_assessment_undershoot_function = OFE_assessment_undershoot_function
        self.process_batch = _timingWrapper(process_batch)
        self.printFunction = printFunction
        self.printLevel = printLevel
        self.saveTo = saveTo
        self.saveProgress = saveTo <> None
        self.saveInterval = saveInterval
        self.record_X_hist = record_X_hist
        self.normalize_objective_values = normalize_objective_values
        self.post_iteration_function = post_iteration_function
        # initialization
        self.it = 0
        self.transform_utopia_point  =  numpy.ones(len(objectiveFunctions)) * numpy.inf
        self.transform_nadir_point  = -numpy.ones(len(objectiveFunctions)) * numpy.inf
        if record_X_hist:
            for sp in self.subproblems:
                sp.X_history = []
        self.evaluate_candidate_designs_stats = []
        self.continueOptimisation()
                            
    def continueOptimisation(self):
        while any( s.active() for s in self.subproblems ):
            for sp_index, sp in enumerate(self.subproblems) :
                if not sp.active():
                    continue
                if not sp.initialized :
                    self.printInfo(2, '  generating initial design: subproblem %i/%i' % (sp_index+1, len(self.subproblems)))
                    sp.x_c = self._generate_random_valid_x(sp)
                else:
                    OFE_target = sp.target_OFE_budgets[ randint(len( sp.target_OFE_budgets)) ]
                    failCount = 0
                    candidate_design_acceptable = False
                    while not candidate_design_acceptable:
                        x = sp.PFA.best_design( f1 = OFE_target )
                        lb = self.CPV_lb
                        ub = self.CPV_ub
                        x[1:] = x[1:] + (rand(len(lb)) - 0.5)*(ub - lb)
                        x[0] = numpy.log(OFE_target)
                        candidate_design_acceptable, msg = self._constraints_satisfied( x, sp )
                        if not candidate_design_acceptable:
                            failCount = failCount + 1
                            self.printInfo(3, '  subproblem %i : generating x_c failed (count %i) %s' % (sp_index+1,  failCount, msg))
                            if failCount > 10:
                                self.printInfo(2, '  subproblem %i/%i : generating x_c failureCount %i > 10, assigning valid random values, last failure message %s' % (sp_index+1, len(self.subproblems),failCount, msg))
                                x = self._generate_random_valid_x(sp)
                                candidate_design_acceptable = True
                    sp.x_c = x
            self.evaluate_candidate_designs()
            self.it = self.it + 1
            if self.printLevel > 1:
                gammaUsed = sum(sp.gamma for sp in self.subproblems)
                gammaBudget = sum(sp.gammaBudget for sp in self.subproblems)
                self.printInfo(2, 'RAND_MOTA , it %i complete, gamma budget used %3.2f%%' % (self.it, 100.0*gammaUsed/gammaBudget))
            if self.saveInterval > 0 and self.it % self.saveInterval == 0 :
                if self.saveProgress : self.save(self.saveTo)
            self.post_iteration_function(self)
        #saving final status
        self.T_finish = datetime.datetime.now()
        if self.saveProgress : self.save(self.saveTo)

    def __repr__(self):
        t_total = (self.T_finish - self.T_start).total_seconds()
        t_alg_being_tuned = sum(f.total_seconds() for f in self.objectiveFunctions) + self.process_batch.total_seconds()
        t_tuner = t_total - t_alg_being_tuned
        return """< RAND_M tuning optimization: no. of subproblems %i, total gamma used %s
sample Sizes: %s interruption confidence level: %1.2f
total time: %7.2fs   time MOTA: %7.2fs   overhead: %3.1f%%>""" \
            % (len(self.subproblems),  commaFormat(sum(sp.gamma for sp in self.subproblems)),
               str(self.sampleSizes), self.resampling_interruption_confidence,
               t_total, t_tuner, (t_tuner/t_alg_being_tuned)*100)
