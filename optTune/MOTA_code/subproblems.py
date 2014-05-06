if __name__ == '__main__':
    import sys
    sys.path.append('../..')

import numpy, copy
from paretoArchive2D_with_simlurity_checking import paretoArchive2D_polynimal_simularity_checking
from optTune.paretoArchives import PFA_history_recorder

#separte OFE budget adjustment scheme. Is think so!

class MOTA_subproblem:
    ''' MOTA subproblem using Tchebycheff scalarization '''
    _repr_name = 'MOTA_subproblem_Tchebycheff'
    def __init__(self, w, target_OFE_budgets, gammaBudget, updateNeighbours,
                 extra_termination_critea = [],):
        '''
Required Args
  * w - weights vector used for scalarization.
  * initial_target_OFE_budgets - OFE budgets at which tuning is desired to take place for the subproblem
  * gammaBudget - number of application layer evaluations to use on subproblem. i.e. if the algorithm being tuned is run 3 times with an OFE budget of 180 on 2 subproblem (w has 2 non-zero elements), then the gamma usage is 1080.
  * updateNeighbours - subproblems who PFAs are in the update neighbourhood when CPV tuples for this subproblem are evaluated
'''
        assert type(w) == numpy.ndarray
        assert type(target_OFE_budgets) == numpy.ndarray
        assert type(gammaBudget) == int
        assert type(updateNeighbours) == list
        self.initialized = False
        self.w = w
        self.target_OFE_budgets = copy.deepcopy(target_OFE_budgets)
        self.min_target_OFE_budget = min(target_OFE_budgets)
        self.max_target_OFE_budget = max(target_OFE_budgets)
        self.gamma = 0
        self.gammaBudget = gammaBudget
        self.updateNeighbours = updateNeighbours
        self.extra_termination_critea = extra_termination_critea
        self.terminated_due_extra_termination_critea = False
        self.PFA = paretoArchive2D_polynimal_simularity_checking() #Pareto-optimal Front Approximation
        self.PFA_history = PFA_history_recorder()
        self.objective_transform_m = numpy.ones(len(w))
        self.objective_transform_c = numpy.zeros(len(w)) # z' = m*z + c

    def add_gamma_usage(self, g):
        self.gamma = self.gamma + g

    def active(self, call_from_duplicate = False):
        if self.terminated_due_extra_termination_critea:
            return False
        if not call_from_duplicate and \
                any( tc.satisfied(self) for tc in self.extra_termination_critea ):
            #msgs = [ tc.msg for tc in self.extra_termination_critea if hasattr(tc,'msg') ]
            #self.printInfo(1, 'tMOPSO terminating, %s' % (';'.join(msgs) ) )
            self.terminated_due_extra_termination_critea = True
            return False
        return self.gamma < self.gammaBudget

    def F_mask_excluding_updateNeighbours(self):
        return self.w <> 0

    def get_F_mask(self):
        F_mask = self.F_mask_excluding_updateNeighbours()
        for n in self.updateNeighbours:
            F_mask = F_mask + n.F_mask_excluding_updateNeighbours()
        return F_mask

    def mux_target_OFE_budgets(self):
        B = self.target_OFE_budgets.tolist()
        for n in self.updateNeighbours:
            B = B + n.target_OFE_budgets.tolist()
        return numpy.unique(B) #unique also sorts from smallest to largest

    def scalarize(self, U, W, M, C):
        '''
        normalize and then scalarize utility values
        U = [[u_1_sample1, u_1_sample2,...], [u_2_sample1, u_2_sample2, ...], ...]
        '''
        return numpy.max( [w*(m*u+c) for w,u,m,c in zip(W, U, M, C) if w <> 0 ], axis=0)

    def scalarize_utility_values(self, U_i):
        '''
        calls scalarize using self.(w, objective_transform_m, objective_transform_c)
        '''
        return self.scalarize( U_i, self.w, self.objective_transform_m, self.objective_transform_c)

    def convert_results_to_local_objectives( self, B, U ):
        'B, U from CPV_evaluation_manager results'
        f1_vals, f2_arrays, U_i_arrays = [], [], []
        for b, U_i  in zip(B, U):
            if b in self.target_OFE_budgets:
                f1_vals.append(b)
                f2_arrays.append(self.scalarize_utility_values( U_i ) )
                U_i_arrays.append( U_i)
        return f1_vals, f2_arrays, U_i_arrays

    def _update_PFA(self, B, U, CPV_list):
        for f1, f2_vals, U_i in zip(*self.convert_results_to_local_objectives( B, U )):
            xv = numpy.array([numpy.log(f1)] + CPV_list)
            self.PFA.inspect_design( xv, f1, f2_vals, U_i )
            self.initialized = True 

    def update_PFAs(self, B, U, CPV_list):
        "update the self's PFA and all the PFAs in update Neighbourhood"
        self._update_PFA( B, U, CPV_list)
        for n in self.updateNeighbours:
            n._update_PFA( B, U, CPV_list)
    
    def largest_OFE_budget_not_likely_to_be_dominated(self, B, U, confidenceLevel):
        OFE_max = 0
        for b,U_i in reversed(zip(B,U)):
            if b > OFE_max: #required as break only stop top level for loop
                for sp in [self]+self.updateNeighbours:
                    if b in sp.target_OFE_budgets:
                        f2_vals = sp.scalarize_utility_values( U_i )
                        if not sp.PFA.probably_dominates(b, f2_vals, confidenceLevel):
                            OFE_max = b
                            break
            else:
                break
        return OFE_max

    def OFE_budgets_not_likely_to_be_dominated(self, B, U, confidenceLevel):
        B_out = []
        for b,U_i in zip(B,U):
            for sp in [self]+self.updateNeighbours:
                if b in sp.target_OFE_budgets:
                    f2_vals = sp.scalarize_utility_values( U_i )
                    if not sp.PFA.probably_dominates(b, f2_vals, confidenceLevel):
                        B_out.append( b )
                        break #break top level loop only
        return numpy.array(B_out)

    def flush_PFA_if_not_initialized(self, include_neighbours=True):
        if not self.initialized:
            self.PFA.flush()
        if include_neighbours:
            for n in self.updateNeighbours:
                if not n.initialized:
                    n.PFA.flush()

    def _update_PFA_no_xv(self, B, U):
        for f1, f2_vals, U_i in zip(*self.convert_results_to_local_objectives( B, U )):
            self.PFA.inspect_design( None, f1, f2_vals, U_i )

    def update_PFA_if_not_initialized(self, B, U, include_neighbours=True):
        if not self.initialized:
            self._update_PFA_no_xv( B, U)
        if include_neighbours:
            for n in self.updateNeighbours:
                if not n.initialized:
                    n._update_PFA_no_xv( B, U)

    def adjust_objective_transform(self, m, c):
        self.objective_transform_m = m
        self.objective_transform_c = c
        x_org =  [ d.xv     for d in self.PFA.designs]
        f1_org = [ d.f1_val for d in self.PFA.designs]
        U_org =  [ d.U_i    for d in self.PFA.designs]
        self.PFA.flush()
        for x,f1,U_i in zip(x_org , f1_org,  U_org):
            self.PFA.inspect_design( x, f1, self.scalarize_utility_values(U_i), U_i )
            
    def update_PFA_history(self):
        self.PFA_history.record(self.gamma, self.PFA)

    def _duplicate_sp_class(self, *args ):
        return MOTA_subproblem_aux( *args )

    def duplicate(self, n):
        '''
        duplicate subproblem as to create aider supbroblems, whose neighbourhood conists of the aid subproblem and the this subproblem. Gamma budget for subproblem divided up so that parent and aiders, have same gamma budget as the original subproblem
        '''
        #assert len(self.updateNeighbours) == 0
        if self.gammaBudget == -1:
            return []
        aux_sps = []
        for i in range(n):
            aux_sps.append( self._duplicate_sp_class( self.w, self.target_OFE_budgets, 1, [self] + self.updateNeighbours) )
        return aux_sps
    def is_aux_subproblem(self):
        return False

    def __repr__(self):
        a = numpy.array(self.PFA_history.gamma_hist)
        v = (
            self._repr_name, 
            '[%s]' %' '.join(['%1.2f' % w_i for w_i in self.w]),
            commaFormat(self.gamma),
            100.0 * self.gamma/ self.gammaBudget,
            sum(a == 0),
            sum(a < self.gammaBudget) - sum(a == 0)
            )
        return '<%s : weights %s, gamma used %s (%3.2f%% of allocated) it_start %i it_active %i>' % v

class  MOTA_subproblem_aux(MOTA_subproblem):
    "used for MOTA_subproblem.duplicate"
    _repr_name = 'MOTA_subproblem_Tchebycheff_aux'
    def is_aux_subproblem(self):
        return True
    def add_gamma_usage(self, g):
        self.updateNeighbours[0].add_gamma_usage(g)
    def active(self):
        return self.updateNeighbours[0].active(call_from_duplicate = True)
    def largest_OFE_budget_not_likely_to_be_dominated(self, B, U, confidenceLevel):
        "save computational resources"
        return self.updateNeighbours[0].largest_OFE_budget_not_likely_to_be_dominated( B, U, confidenceLevel)
    def OFE_budgets_not_likely_to_be_dominated(self, B, U, confidenceLevel):
        return self.updateNeighbours[0].OFE_budgets_not_likely_to_be_dominated( B, U, confidenceLevel)
    def update_PFA_history(self):
        pass

class MOTA_subproblem_weighted_sum(MOTA_subproblem):
    ''' MOTA subproblem using weighted sum scalarization '''
    _repr_name = 'MOTA_subproblem_weighted_sum'
    def scalarize(self, U, W, M, C):
        'U = [[u_1_sample1, u_1_samples2,...], [u_2_sample1, u_2_samples2, ...], ...]'
        return numpy.sum( [w*(m*u+c) for w,u,m,c in zip(W, U, M, C) if w <> 0 ], axis=0)
    def _duplicate_sp_class(self, *args ):
        return MOTA_subproblem_weighted_sum_aux( *args )


class  MOTA_subproblem_weighted_sum_aux(MOTA_subproblem_weighted_sum):
    _repr_name = 'MOTA_subproblem_weighted_sum_aux'
    def is_aux_subproblem(self):
        return True
    def add_gamma_usage(self, g):
        self.updateNeighbours[0].add_gamma_usage(g)
    def active(self):
        return self.updateNeighbours[0].active(call_from_duplicate = True)
    def largest_OFE_budget_not_likely_to_be_dominated(self, B, U, confidenceLevel):
        "save computational resources"
        return self.updateNeighbours[0].largest_OFE_budget_not_likely_to_be_dominated( B, U, confidenceLevel)
    def OFE_budgets_not_likely_to_be_dominated(self, B, U, confidenceLevel):
        return self.updateNeighbours[0].OFE_budgets_not_likely_to_be_dominated( B, U, confidenceLevel)
    def update_PFA_history(self):
        pass



def generate_base_subproblem_list(n_f, target_OFE_budgets, gammaBudget, extra_termination_critea = [],
                                  subproblemClass=MOTA_subproblem ):
    ''' Convenience function which return a subproblems for the weights [1,0, ..., 0], [0,1,...,0], ... [0,0,...,1] for the n_f objectives. '''
    subproblems = []
    for w in numpy.eye( n_f ):
        subproblems.append( 
            subproblemClass (
                w = w,
                target_OFE_budgets = target_OFE_budgets, 
                gammaBudget = gammaBudget,
                extra_termination_critea = extra_termination_critea,
                updateNeighbours = []
                )
            )
    return subproblems
        

def commaFormat(i):
    s = str(int(i))
    s_out = ''
    for j in range(len(s)):
        if j > 0 and j % 3 == 0:
            s_out = s[len(s)-1-j] + ',' + s_out
        else:
            s_out = s[len(s)-1-j] + s_out
    return s_out


class sp_w_similarity:
    def __init__(self, a, b, refs_to_b):
        self.w_dot = numpy.dot(a.w / numpy.linalg.norm(a.w), b.w / numpy.linalg.norm(b.w)) #favor greater
        self.b = b
        self.refs_to_b = refs_to_b #favor less
    def __lt__(self, b):
        if self.w_dot <> b.w_dot:
            return self.w_dot < b.w_dot
        else:
            return self.refs_to_b > b.refs_to_b
            


def generate_update_neighbourhoods(subproblems, T):
    'for each subproblem create a neighborhood of size T, based on simularity of w values.'
    assert T < len(subproblems)
    assert all( len(sp.updateNeighbours) == 0 for sp in subproblems )
    references = {}
    for sp in subproblems:
        references[id(sp)] = 0
    for sp in subproblems:
        D = []
        for sp_i in subproblems:
            if sp_i <> sp:
                D.append(sp_w_similarity(sp, sp_i, references[id(sp_i)]))
        Ds = sorted(D, reverse=True)
        for d in Ds[:T]:
            sp.updateNeighbours.append(d.b)
            references[id(d.b)] = references[id(d.b)] + 1



if __name__ == '__main__':
    print('Testing generate_update_neighbourhoods function')
    subproblems = generate_base_subproblem_list(5, numpy.array([5,10]), 1000, 10, 1)
    for sp in subproblems:
        print(sp)
    generate_update_neighbourhoods(subproblems, 2)
    for sp in subproblems:
        print(' - %s  neighbors %s' % (sp.w, ', '.join('%s' % sp_i.w for sp_i in sp.updateNeighbours)))
