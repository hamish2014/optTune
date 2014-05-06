import numpy

class CPV_evaluation_manager:
    '''
    helper class to be used by tMOPSO.evaluate_candidate_designs
    remember x = [OFE_budget_target, CPV_1, CPV_2, ...]
             optAlg ( [CPV_1, CPV_2, ...,], OFE_budgets, randomSeed )

    if OFE_budgets is an numpy.array, then optAlg is expected to return solution errors corresponding to that array
        E = [ OFE_budget[0] OFE_budget[1] ... OFE_budget[-1] ] 
    *NB some OFE_budget items can be omitted, if nessary. OFE_budgets is used to focus the search and save computational resource. For scenerios  such as tuning multi-objective algorithms where performance assessment expensice, the use of OFE_budgets as an numpy.array is highly recommended.

    if OFE_budgets is an integer, then optAlg is expected to return solution error values for an
        E = [  N , 2N , ...,  OFE_budget ]

    '''
    def __init__(self, CPVs, optAlg, specified_OFE_budgets, min_OFE_budget, max_OFE_budget, max_sample_size, printFun, optAlg_addtoBatch=None):
        '''OFE_budgets=None, implies all OFE budgets'''
        self.CPVs = CPVs
        self.optAlg = optAlg
        if specified_OFE_budgets <> None:
            B = specified_OFE_budgets
            assert type(B) == numpy.ndarray
            if min_OFE_budget > max_OFE_budget: #could occur for tMOPSO article, when history disabled
                max_OFE_budgt = min_OFE_budget
            self.OFE_budgets = B[ sum(B < min_OFE_budget) : sum(B < max_OFE_budget)+1]
        else:
            self.OFE_budgets = max_OFE_budget
            assert min_OFE_budget == 0
        self.max_sample_size = max_sample_size
        self.printFun = printFun
        self.optAlg_addtoBatch = optAlg_addtoBatch
        #generating random seeds, 2147483647 is max of a signed 4 byte integer
        self.seeds = [numpy.random.randint(2147483647) for i in range(self.max_sample_size)]
        #some book keeping
        self.n = 0 #samples counter
        self.OFE_budget_hist = []
        self.discarded_fvals = {} # for get_F_return, as to return values for local PF updates
    
    def reduce_max_OFE_budget(self, new_max_OFE_budget):
        if type(self.OFE_budgets) ==  numpy.ndarray:
            self.OFE_budgets = self.OFE_budgets[ : sum( self.OFE_budgets <= new_max_OFE_budget)]
        else:
            self.OFE_budgets = new_max_OFE_budget

    def set_new_OFE_budgets(self, new_OFE_budgets):
        assert type(self.OFE_budgets) ==  numpy.ndarray
        evals_new = []
        f_new = []
        for e, fvals in zip(self.evals, self.fvals ):
            if e in new_OFE_budgets:
                evals_new.append(e)
                f_new.append(fvals)
            else:
                self.discarded_fvals[e] = fvals[:self.n]
        self.evals = numpy.array( evals_new )
        assert numpy.array_equal(self.evals, new_OFE_budgets)
        self.fvals =  numpy.array( f_new )
        self.OFE_budgets = self.evals.copy()

    def max_OFE_budget(self):
        if type(self.OFE_budgets) ==  numpy.ndarray:
            return max(self.OFE_budgets) if len(self.OFE_budgets) >  0 else 0
        else:
            return self.OFE_budgets

    def addtoBatch(self, repeats):
        if self.optAlg_addtoBatch == None:
            return
        for s in self.seeds[ self.n : self.n + repeats ] :
            self.optAlg_addtoBatch( self.CPVs, self.OFE_budgets, s )

    def results(self, repeats):
        assert self.max_OFE_budget() > 0
        for s in self.seeds[ self.n : self.n + repeats ] :
            try:
                F, E = self.optAlg( self.CPVs, self.OFE_budgets, s  )
            except ValueError,msg:
                print('CPVs, OFE_budgets, randomSeed:',self.CPVs, self.OFE_budgets, s  )
                print('self.OFE_budgets', self.OFE_budgets)
                raise ValueError, msg
            assert len(F) == len(E)
            self.OFE_budget_hist.append(max(E))
            if self.n == 0:
                self.evals = E
                self.fvals = -numpy.ones( [ len(F), self.max_sample_size ] )
                self.fvals[:, self.n] = F
            elif len(F) == self.fvals.shape[0]: #fvals.shape = (rows, columns)
                assert ( E == self.evals ).all()
                self.fvals[:, self.n] = F
            elif len(F) < self.fvals.shape[0]: #multiplying last value to lengthen F, current optAlg run terminated prematurely.
                assert ( E == self.evals[:len(E)] ).all()
                self.fvals[:len(F), self.n] = F
                self.fvals[len(F):, self.n] = F[-1]
            elif (E[:len(self.evals)] == self.evals ).all() : #previous optAlg runs terminated prematurely; multiplying last value to lengthen self.fvals as to match length(F)
                if not (E[:len(self.evals)] == self.evals ).all(): #very slow
                    print('self.evals', self.evals)
                    print('E',E)
                    raise RuntimeError," not (E[:len(self.evals)] == self.evals ).all(), when multiplying last value to lengthen of self.fvals as to match length(F)"
                self.evals = E
                fvals_new = -numpy.ones( [ len(F), self.max_sample_size ] )
                fvals_new[:self.fvals.shape[0], :self.n] = self.fvals[:,:self.n]
                for i in range(self.fvals.shape[0], len(F)):
                    fvals_new[i, :self.n] = self.fvals[-1,:self.n]
                self.fvals = fvals_new
                self.fvals[:, self.n] = F
            else:
                print('CPV_evaluation manager unable to parse data from optAlg call! Info:')
                print('  optAlg call, CPVs         %s' % self.CPVs)
                print('  optAlg call, OFE_budgets  %s' % self.OFE_budgets)
                print('  optAlg call, randomSeed   %s' % s)
                print('  F returned      %s' % F)
                print('  E returned      %s' % E)
                print('  CPVEM evals    %s' % self.evals)
                raise RuntimeError
            if self.n > 0 and hasattr(self.optAlg,'evals'):
                self.optAlg.evals = self.optAlg.evals -1 #correction for CPV eval counter
            self.n = self.n + 1
        if type(self.OFE_budgets) ==  numpy.ndarray:  
            ind = sum(self.evals <= self.OFE_budgets[-1])
        else:
            ind = sum(self.evals <= self.OFE_budgets)
        return self.evals[ :ind ], self.fvals[ :ind, :self.n ]

    def OFEs_used_over_last_n_runs(self, n):
        assert n <= len(self.OFE_budget_hist) 
        return sum(self.OFE_budget_hist[i] for i in range(-n,0))

    def get_xv_fv_for_OFE_budget(self, target_OFE_budget):
        'for local PF updates,'
        def get_closest(E):
            return max([e for i,e in enumerate(E) if  e <= target_OFE_budget or i==0 ] + [-1])
        closest_evals =      get_closest(self.evals)
        closest_discarded =  get_closest(sorted(self.discarded_fvals.keys()))
        if closest_discarded == -1 and closest_evals == -1:
             print('CPVs         %s' % self.CPVs)
             print('  self.evals %s' % self.evals)
             print('  self.discarded_fvals %s' % self.discarded_fvals)
             print('  target_OFE_budget %f' % target_OFE_budget)
             raise RuntimeError
        if closest_discarded > closest_evals:
            xc = numpy.array([ numpy.log(closest_discarded)] + self.CPVs.tolist())
            fc = numpy.array([ numpy.log(closest_discarded)] + self.CPVs.tolist())
        else:
            ind = sum(self.evals <= target_OFE_budget)
            k = sum( self.evals[ind-1] <= beta for beta in self.OFE_budget_hist )
            xc = numpy.array([ numpy.log(self.evals[ind-1])] + self.CPVs.tolist())
            fc = numpy.array([ self.evals[ind-1],  self.fvals[ind-1, :k ].mean()])
        return xc, fc
                                        

if __name__ == "__main__":
    import sys, numpy
    def optAlg(CPVs, OFE_budgets, randomSeed):
        if type(OFE_budgets) <> numpy.ndarray:
            n = numpy.random.randint(5,10)
            E = numpy.arange(1,n)
        else:
            E = OFE_budgets.copy()
        F = E.copy()
        return F, E
    
    def printFun(level, text):
        print(text)

    print('Basic test for tMOPSOs CPV evaluation manager')

    sampleSizes = [3,5]
    c = CPV_evaluation_manager(CPVs=numpy.array([2]), optAlg=optAlg, specified_OFE_budgets=None, min_OFE_budget=0, max_OFE_budget=10, max_sample_size=sum(sampleSizes), printFun=printFun)


    for i, repeats in enumerate(sampleSizes):
        evals, fvals = c.results(repeats)
        print(fvals)
        print('evals_used in generating extra %i runs : %i (total evals should be %i)' % (repeats, c.OFEs_used_over_last_n_runs(repeats),sum(fvals[len(fvals)-1,:])))

    print('\ntesting, tMOPSO get_F_vals_at_specified_OFE_budgets')
    sys.path.append('../..')
    from optTune import  get_F_vals_at_specified_OFE_budgets
    
    Fv = numpy.array([ 0.9, 0.5, 0.3, 0.2 , 0.15, 0.12 ])
    Ev = numpy.arange(1,len(Fv)+1)*3
    E_desired = numpy.array([3,7,10,11,14,20,25])
    def sub_fun(F_in,E_in,E_d):
        print('zip(E_in,F_in) %s' %' '.join('%i,%1.2f ' % (e,f) for e,f in zip(E_in,F_in)))
        print('  E_desired %s' % ' '.join(map(str,E_d)))
        F_out, E_out =  get_F_vals_at_specified_OFE_budgets(F_in,E_in,E_d)
        print('  zip(E_out,F_out) %s' %' '.join('%i,%1.2f ' % (e,f) for e,f in zip(E_out,F_out)))
    sub_fun(Fv,Ev,E_desired)
    sub_fun(Fv[1:],Ev[1:],E_desired)
