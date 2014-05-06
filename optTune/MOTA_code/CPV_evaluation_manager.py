import numpy

inconsistent_sampling_error_msg = "MOTA objectives / optAlg -> must return the same number E for a specified CPVs and target_OFE_budgets, irespective of the randomSeeds. Ensure that the algorithm being tuned only terminates due to the OFE budget constraint."

class  CPV_evaluation_manager:
    def __init__( self, subproblem, objectiveFunctions, CPVs, min_OFE_budget, max_OFE_budget, maxSamples, 
                  printFunction ):
        self.subproblem = subproblem
        self.F = objectiveFunctions
        assert len(subproblem.w) == len(objectiveFunctions)
        self.CPVs = CPVs
        self.F_mask = subproblem.get_F_mask()
        B = subproblem.mux_target_OFE_budgets()
        self.target_OFE_budgets = B[ sum(B < min_OFE_budget) : sum(B < max_OFE_budget)+1] # max_OFE 8 , B_clean = [5, 10], +1 forces upto 10
        self.maxSamples = maxSamples
        self.randomSeeds =  [numpy.random.randint(2147483647) for i in range(self.maxSamples)] #common to all subMangers, as to allow solution sharing when tuning multi-objective algorithms
        self.printFunction = printFunction
        self.subManagers = [ _sub_manager(self.CPVs, f, maxSamples, self.randomSeeds, printFunction )
                             for f in objectiveFunctions ]

    def reduce_max_target_OFE_budget(self, new_max_OFE_budget):
        new_max_ind = sum( self.target_OFE_budgets <= new_max_OFE_budget)
        if new_max_OFE_budget == 0:
            assert new_max_ind == 0
        else:
            assert self.target_OFE_budgets[new_max_ind - 1] >=  new_max_OFE_budget
        #print('reduce_max_target_OFE_budget   new_max_OFE_budget %i   target_OFE_budgets[ : new_max_ind ] %s   target_OFE_budgets[new_max_ind ] %i' % (new_max_OFE_budget, self.target_OFE_budgets[ : new_max_ind ], self.target_OFE_budgets[min(new_max_ind, len(self.target_OFE_budgets) -1) ]))
        self.target_OFE_budgets = self.target_OFE_budgets[ : new_max_ind ]


    def reduce_target_OFE_budgets( self, new_target_OFE_budgets) :
        if not numpy.array_equal(self.target_OFE_budgets , new_target_OFE_budgets):
            #print('self.CPVs', self.CPVs)
            #print('self.target_OFE_budgets', self.target_OFE_budgets)
            #print('new_target_OFE_budgets', new_target_OFE_budgets)
            self.target_OFE_budgets = new_target_OFE_budgets
            for m_i, sm in zip(self.F_mask, self.subManagers):
                if m_i:
                    sm.set_new_OFE_budgets( new_target_OFE_budgets  )

    def max_target_OFE_budget(self):
        return self.target_OFE_budgets[-1] if len(self.target_OFE_budgets) > 0 else 0

    def addtoBatch(self, repeats):
        if len(self.target_OFE_budgets) > 0 : 
            for m_i, sm in zip(self.F_mask, self.subManagers):
                if m_i :
                    sm.target_OFE_budgets = self.target_OFE_budgets 
                    sm.addtoBatch(repeats)

    def update(self, repeats):
        if len(self.target_OFE_budgets) > 0 : 
            for m_i, sm in zip(self.F_mask, self.subManagers):
                if m_i:
                    sm.target_OFE_budgets = self.target_OFE_budgets 
                    try:
                        sm.update(repeats)
                    except ValueError,msg:
                        print('self.CPVs, self.target_OFE_budgets', self.CPVs, self.target_OFE_budgets )
                        raise ValueError,msg

    def results(self):
        '''
        returns OFE_usages, U
        where U is a list with a 2D arrays, U_i corresponding to each budget in OFE_usages
        U_i = [ [u_1_sample1, u_1_samples2, ...], [u_2_sample1, u_2_samples2, ...], ...]
        
          timeit.timeit('x = numpy.zeros([20,8])','import numpy', number=10**6) ->  0.66292405128479
          timeit.timeit('x = numpy.ones([20,8])','import numpy', number=10**6) -> 2.6216650009155273
          timeit.timeit('x = numpy.ones([20,8])*numpy.nan','import numpy', number=10**6) -> 5.883631944656372
          timeit.timeit('x = numpy.zeros([20,8])+numpy.nan','import numpy', number=10**6) -> 3.4321529865264893
          timeit.timeit('x = numpy.zeros([20,8]); x.fill(numpy.nan)','import numpy', number=10**6) -> 1.2564830780029297
        '''

        if len(self.target_OFE_budgets) == 0 : 
            return [],[]
        Y = []
        B_all = []
        for v, sm in zip(self.F_mask, self.subManagers):
            if v :
                y, B_i = sm.get_F_E( self.target_OFE_budgets )
                B_all.append(B_i)
                Y.append(y)
            else:
                Y.append(None)
        assert all( (b == B_all[0]).all() for b in B_all)
        B = B_all[0]
        no_samples = max(sm.n for sm in self.subManagers) #should return something like [0 , 0, no_samples, 0, no_samples ...]
        U = []
        for i, b in enumerate(B):
            U_i = numpy.zeros([len(self.F),no_samples]);
            U_i.fill(numpy.nan)
            for j,y in enumerate(Y):
                if y <> None:
                    U_i[j,:] = y[i,:]
            U.append(U_i)
        return B, U

    def gamma_usage(self):
        return sum( sm.OFEs_made for sm in self.subManagers) 

class _sub_manager:
    '''
    helper class to be used by MOTA to evaluate_candidate_designs
    remember 
             f( numpy.array [CPV_1, CPV_2, ..., OFE_budget, randomSeed] )
    '''
    def __init__(self, CPVs, optAlg, max_sample_size, randomSeeds, printFun):
        '''OFE_budgets=None, implies all OFE budgets'''
        self.CPVs = CPVs
        self.n = 0 #samples counter
        self.optAlg = optAlg
        self.max_sample_size = max_sample_size
        self.printFun = printFun
        #generating random seeds, 2147483647 is max of a signed 4 byte integer
        self.seeds = randomSeeds
        #some book keeping
        self.OFEs_made = 0
    
    def addtoBatch(self, repeats):
        if hasattr(self.optAlg.f, 'addtoBatch'):
            for s in self.seeds[ self.n : self.n + repeats ] :
                self.optAlg.f.addtoBatch( self.CPVs, self.target_OFE_budgets, s )

    def update(self, repeats):
        for s in self.seeds[ self.n : self.n + repeats ] :
            F, E = self.optAlg( self.CPVs, self.target_OFE_budgets, s   )
            assert len(F) == len(E)
            assert type(F) == numpy.ndarray
            assert type(F) == numpy.ndarray
            if self.n == 0:
                if not all( e in self.target_OFE_budgets for e in E):
                    raise ValueError, "algorithms performance indicator values requested for OFE budgets of %s, however optAlg call returned indicator values for OFE budgets not in this list. The optAlg call return indicator values for OFE budgets of %s " % (self.target_OFE_budgets,E)
                self.evals = E
                self.evals_ind = len(E)
                self.fvals = numpy.zeros( [ len(F), self.max_sample_size ] )
                self.fvals[:, self.n] = F
            else:
                if len(F) > len(self.evals) or not ( E == self.evals[:len(E)] ).all():
                    raise RuntimeError, inconsistent_sampling_error_msg 
                self.fvals[:len(F), self.n] = F
                assert len(E) <= self.evals_ind
                self.evals_ind = len(E)
            if self.n > 0 and hasattr(self.optAlg,'evals'):
                self.optAlg.evals = self.optAlg.evals -1 #correction for CPV eval counter
            self.n = self.n + 1
            self.OFEs_made = self.OFEs_made + max(E)

    def set_new_OFE_budgets(self, new_OFE_budgets):
        #print('submanger.evals', self.evals)
        new_F =  numpy.zeros( [ len(new_OFE_budgets), self.max_sample_size ] )
        evals_new = []
        fval_cols_new = []
        for e, f_row in zip(self.evals, self.fvals):
            if e in new_OFE_budgets:
                evals_new.append(e)
                fval_cols_new.append(f_row)
        self.evals = numpy.array(evals_new)
        #print('submanger.evals_new', evals_new)
        assert numpy.array_equal(self.evals , new_OFE_budgets)
        #exit(2)
        self.fvals = numpy.array(fval_cols_new)

    def get_F_E(self, E_desired):
        '''
        CPV_evaluation_manager combines target OFE budgets of all the subproblems in the update neighborhood in one list, target_OFE_budgets (which is reduced through preemptively terminating resampling).
        When optAlg is called it is requested to give performance indicator values for the OFE budgets in this list.
        
        
        target OFE budgets     :  [  4  6  9 12  16   25   40] #100]  removed during preemptively terminating resampling

        Values Gathered during update
        optAlg returns (N=10)  :  [          12       25   40] #self.evals[:len(self.evals_ind)]
                                  [          f_1     f_2  f_4]

        subproblem requests indicator values at
                                  [    6     12       25        100]
        which return f_vals       [          f_1     f_2           ]

        '''
        E_available = self.evals[:self.evals_ind].tolist()
        E_out = []
        F_inds = []
        for e in E_desired:
            if e in E_available:
                E_out.append(e)
                F_inds.append(E_available.index(e))
        return self.fvals[ F_inds, :self.n ], numpy.array(E_out)

if __name__ == '__main__':
    print('Test to check MOTAs CPV evaluation manager. One test checks if terminate before reaching the specified OFE budget')
    import sys
    sys.path.append('../../')
    from subproblems import MOTA_subproblem, generate_base_subproblem_list
    from matplotlib import pyplot

    class poly_optAlg:
        def __init__(self, p, noise_strength):
            self.p = p
            self.noise_strength = noise_strength
        def __call__(self, CPVs, target_OFE_budgets, randomSeed):
            evals = 0
            N = 5
            E = []
            F = []
            while evals + N <= max(target_OFE_budgets):
                evals = evals + N
                OFE_budget = min( e for e in target_OFE_budgets if e >= evals)
                #print('evals %i OFE_budget %i' % ( evals, OFE_budget ))
                if evals + N > OFE_budget:
                    E.append( OFE_budget )
                    F.append( numpy.polyval(self.p,CPVs[0]*[evals])[0] \
                                  + self.noise_strength*numpy.random.randn() )
            return numpy.array(F), numpy.array(E)
    F  = [
        poly_optAlg([-1,1], 0.0),
        poly_optAlg([-0.1, 0, 2], 0.05),
        poly_optAlg([-0.06, -0.2, 2], 0.08)
        ]
    #target_OFEs = numpy.array([1,3,6,9])
    subproblems = generate_base_subproblem_list( len(F), numpy.array([2,5,9,16]), 84, 0, 0 )
    subproblems.append( MOTA_subproblem( numpy.array([1.0,1.0,0.0]), numpy.array([1,4,5,11,20, 25]), 84, 0, 0, []))
    subproblems[-1].updateNeighbours = subproblems[0:3]
    for sp in subproblems:
        print('%s, target OFE budgets %s' % (sp.__repr__(), sp.target_OFE_budgets ))
    print('last subproblem has other subproblems in its update neighberhood')

    print('\noptAlg function simulated a optimization algorithm which uses 5 OFEs per iteration')
    print(' therefore the function values for OFE budget are requested  [ 3  5  9  16 ]')
    print(' the algorithm should return F = [ f_2  f_4 ] E = [ 5  16 ]')
    print(' algorithm actually returns E = %s' % F[0]([1],[ 3, 5, 9, 16 ],123)[1])
    print('')


    print('CPV_evaluation_manager, is constructed for all of these problems, with min_OFE_budget=4, and max_OFE_budget=20')

    for sp_em in subproblems:
        pyplot.figure()
        em = CPV_evaluation_manager(subproblem = sp_em, objectiveFunctions=F, CPVs=numpy.array([1]), min_OFE_budget=4, max_OFE_budget=20, maxSamples=15, printFunction=None )
        print(sp_em)
        em.update(15)
        B, U = em.results()
        print('  CPV target OFE budgets    %s      OFE budget_returned  %s' % (em.target_OFE_budgets, B))
        print('U[%i OFEs][:,:4] ( F_mask %s)' % (B[0], em.F_mask))
        print('%s' % U[0][:,:4])
        def aggregate_Results(w):
            return numpy.array( sum( u*w_i for u,w_i in zip(U_i,w)) for U_i in U)
        for i, sp in enumerate(subproblems):
            pyplot.subplot(2,2,i+1)
            if i == 0:
                pyplot.title('em sp w %s' % str(sp_em.w))
            p_x = sp.target_OFE_budgets[sp.target_OFE_budgets > 2]
            p_y = numpy.max([ w_i * numpy.polyval(F[j].p,p_x*numpy.array([1])) 
                              for j, w_i in enumerate(sp.w) if w_i <> 0], axis=0)
            #p_y = 0
            #for j, w_i in enumerate(sp.w):
            #    p_y = p_y + w_i * numpy.polyval(F[j].p,p_x*numpy.array([1]))
            pyplot.plot(p_x, p_y, 'b.')
            OFE_budgets, f2_arrays, U_org = sp.convert_results_to_local_objectives(B, U)
            for e, sp_F in zip(OFE_budgets, f2_arrays) :
                for f in sp_F:
                    pyplot.plot(e,f,'gx')
            pyplot.xlim(min(p_x) - 1, max(p_x) + 1)
            pyplot.xticks(sp.target_OFE_budgets)
            pyplot.ylabel('w=%s' % str(sp.w))
    print('''
What should be shown 
 * No evaluation mananager (em) should gives values for OFEs (x) == 9
 * The values for OFEs = 16 should be equal to the P(x=15), as that is x corresponding to a OFE budget 16
 * Tchebycheff scalarization, not weighted sum!
''')
    pyplot.show()
