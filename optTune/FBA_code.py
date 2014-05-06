"""
Flexible budget algorithm used to demonstrate the Flexible budget method, based upon 
Branke, J. and Elomari, J. (2012). Meta-optimization for parameter tuning with a flexible computing budget. In Proceedings of the 14th International Conference on Genetic and Evolutionary Computation Conference, pages 1245--1252. ACM.

Changes made
  * focuses specified OFE budgets
  * gaussian_mutation std control parameter added

Branke, J. and Elomari, J. (2012) says the rank-tie breakers of AUC (area under the curve), and AL (area lost) are essientially the same, so only implemented AUC.
"""

import sys, numpy, copy, pickle, time, datetime
from numpy.random import rand, randint, randn


def to_stdout(text):
    "should not be needed in Python 3.0"
    print(text)

class _timingWrapper:
    def __init__(self, f):
        self.f = f
        self.dts = []
    def __call__(self, *args):
        t = time.time()
        output = self.f(*args)
        self.dts.append(time.time() - t)
        return output
    def total_seconds(self):
        return sum(self.dts)
    def __repr__(self):
        return '<_timingWrapper:%s>' % self.f.__repr__()

def _passfunction(*args):
    pass

class FBA:
    def __init__(self, 
                 optAlg,
                 CPV_lb,
                 CPV_ub,
                 CPV_validity_checks,
                 OFE_budgets,
                 repeats,
                 gammaBudget,
                 N,
                 crossover_rate = 0.5,
                 mutation_strength = 0.1,
                 tiebreaking_method = 'AUC',
                 printFunction = to_stdout,
                 printLevel = 2, 
                 saveTo=None,
                 addtoBatch = _passfunction,
                 processBatch = _passfunction):
        '''
Parameters
  * optAlg - algorithm being tuned, where optAlg(CPV_tuple, OFE_budgets, randomSeed) returns F,E curve or history
  * CPV_lb - control parameter value lower bounds
  * CPV_ub - control parameter values upper bounds
  * CPV_validity_checks  - CPV_validity_checks( CPV_array, OFE_budget ) return (Valid,msg)
  * gammaBudget - gamma = optAlg_eval made, i.e. running the optAlg 3 times, using OFE budget of 40, results in a gamma of 120. 
  
'''
        self.T_start = datetime.datetime.now()
        self.optAlg = _timingWrapper(optAlg)
        self.CPV_lb = CPV_lb
        self.CPV_ub = CPV_ub
        self.CPV_validity_checks = CPV_validity_checks
        self.OFE_budgets = OFE_budgets
        self.repeats = repeats
        self.gammaBudget = gammaBudget
        self.N = N
        self.crossover_rate = crossover_rate 
        self.mutation_strength = mutation_strength
        self.mutation_std =  (CPV_ub - CPV_lb)*mutation_strength
        #optional parameters
        self.tiebreaking_method = tiebreaking_method
        self.saveTo = saveTo
        self.printFunction = printFunction
        self.printLevel = printLevel
        self.optAlg_addtoBatch = addtoBatch
        self.optAlg_processBatch = _timingWrapper(processBatch)
        #other stuff
        self.it = 0
        self.n_x = len(self.CPV_lb)
        assert self.n_x > 1 #else one point crossover wont work
        self.gamma = 0
        self.X = []
        self.curves = []  #curve is numpy 2D array of SolutionError vals, due to specifed OFE_budgets, each curve has same number of point, with same `x' values
        self.x_hist = []
        self.curve_hist = []
        self.combined_curves_hist = []
        self.gamma_hist = []
        self.X_seeds = []
        self.optimize()

    def printF(self, level, txt):
        if level <= self.printLevel:
            self.printFunction(txt)

    def optimize(self):
        while self.gamma  < self.gammaBudget:
            X_c = []
            #generating
            if len(self.X) == 0: #not initailized
                self.printF(2,'  Generating initial population')
                while len(self.X) < self.N:
                    x_c = self.CPV_lb + rand(self.n_x) * (self.CPV_ub - self.CPV_lb)
                    if self.CPV_validity_checks(x_c, max(self.OFE_budgets))[0]:
                        self.X.append(x_c)
            else:
                self.printF(2,'  Generating offspring')
                while len(self.X) < 2*self.N:
                    #perform tournament selection - size 2
                    parent_1, parent_2 = 0,0
                    while parent_1 == parent_2:
                        parent_1 = self.better_x( randint(self.N), randint(self.N) )
                        parent_2 = self.better_x( randint(self.N), randint(self.N) )
                    # one point crossover
                    c_ind = randint(1,self.N-1)
                    x_c1, x_c2  = one_point_crossover( self.X[ parent_1], self.X[ parent_2], self.crossover_rate )
                    #gaussian mutation
                    x_c1 = x_c1 + randn(self.n_x) * self.mutation_std
                    x_c2 = x_c2 + randn(self.n_x) * self.mutation_std
                    if self.CPV_validity_checks(x_c1, max(self.OFE_budgets))[0]:
                        self.X.append(x_c1)
                    if self.CPV_validity_checks(x_c2, max(self.OFE_budgets))[0]:
                        self.X.append(x_c2)
            #batch parrel processing
            for i in range(len(self.X)):
                if i >= len(self.curves):
                    self.add_X_toBatch(self.X[i])
            self.optAlg_processBatch()
            #evaluating
            curves = []
            for i in range(len(self.X)):
                if i == len(self.curves):
                    self.printF(3,'  evaluating %s' % self.X[i])
                    self.curves.append( self.evaluate_X(self.X[i]) )
                curves.append( self.curves[i] )
            #selection
            self.ranks = rank_curves( curves  )
            self.printF(2,'  max rank of results %i' % max(self.ranks) )
            tournament_rank = 1
            while sum(numpy.array(self.ranks) <= tournament_rank) < self.N:
                tournament_rank = tournament_rank + 1
            self.printF(2,'  tournament_rank %i' % tournament_rank)
            #first removing all ranks higher then tournament_rank
            for i in reversed(range(len(self.X))):
                if self.ranks[i] > tournament_rank:
                    del self.ranks[i], self.X[i], self.curves[i]
            #then tournament time    
            while len(self.X) > self.N:
                inds_tr = [ i for i,r in enumerate(self.ranks) if r == tournament_rank ]
                i1 = inds_tr.pop( randint(len(inds_tr) ) )
                i2 = inds_tr.pop( randint(len(inds_tr) ) )
                ind_best = self.better_x(i1, i2)
                if i1 == ind_best:
                    del self.ranks[i2], self.X[i2], self.curves[i2]
                else:
                    del self.ranks[i1], self.X[i1], self.curves[i1]
            self.printF(3,'  ranks after tournament %s' % ' '.join(map(str,self.ranks ) ) )
            #/ selection
            self.combined_curves = numpy.min( numpy.array(self.curve_hist), axis=0 )   
            self.gamma_hist.append(self.gamma)

            #print(self.combined_curves)
            #exit(2)
            #self.combined_curves = numpy.min( numpy.array(self.curves), axis=0 )   
            self.combined_curves_hist.append( self.combined_curves )
            self.it = self.it + 1
            self.printF(1,'it %i completed, AUC best %s  FBA progress %3.2f%%' % (self.it, AUC(self.OFE_budgets, self.combined_curves), 100.0 * self.gamma / self.gammaBudget ) )

        self.T_finish = datetime.datetime.now()
        if self.saveTo <> None:
            f = open(self.saveTo,'w')
            pickle.dump( self, f)
            f.close()

    def better_x( self, j, k):
        if j == k:
            return j
        if self.ranks[ j ] < self.ranks[ k ]:
            return j
        elif self.ranks[ j ] > self.ranks[ k ]:
            return k
        else:
            assert self.tiebreaking_method == 'AUC'
            AUC_j = AUC(self.OFE_budgets, self.curves[j] )
            AUC_k = AUC(self.OFE_budgets, self.curves[k] )
            return j if AUC_j < AUC_k else k

    def add_X_toBatch(self, x):
        seeds = [numpy.random.randint(2147483647) for i in range(self.repeats)]
        for s in seeds:
            self.optAlg_addtoBatch(x, self.OFE_budgets, s)
        self.X_seeds.append(seeds)   

    def evaluate_X(self, x):
        seeds = self.X_seeds.pop(0)
        F = []
        E_check = None
        for s in seeds:
            #y = numpy.array( x.tolist() + [max(self.OFE_budgets), s] ) 
            F_s, E = self.optAlg(x, self.OFE_budgets, s)
            assert len(F_s) == len(E)
            self.gamma = self.gamma + max(E)
            if E_check <> None and len(E) <> len(E_check):
                minLen = min(len(E),len(E_check))
                assert (E_check[:minLen] == E[:minLen]).all()
                E_check = E if len(E) > len(E_check) else E_check
            else:
                E_check = E
            F.append(F_s)
        if not(all( len(F_i) == len(F[0]) for F_i in F)):
            self.printF(1, '  evaluate x , F_s runs not all of equal length, lengthening shorter runs to E_check ...')
            for i in range(len(F)):
                if len(F[i]) < len(E_check):
                    F[i] = F[i].tolist() + [F[i][-1]] * (len(E_check) - len(F[i]))
        F = numpy.mean(numpy.array(F),axis=0)
        F_out = map_to_OFE_budgets( self.OFE_budgets, F, E_check)
        self.x_hist.append( x )
        self.curve_hist.append( F_out )
        return F_out

    def get_optimal_X(self):
        X_opt = numpy.ones([ len(self.OFE_budgets), self.n_x])*numpy.nan
        for x_ind in range(len(self.x_hist)):
            for f_ind in range(len(self.OFE_budgets)):
                if (self.curve_hist[x_ind][f_ind] <> numpy.inf and
                    self.curve_hist[x_ind][f_ind] == self.combined_curves[f_ind]):
                    X_opt[f_ind,:] = self.x_hist[x_ind]
        return X_opt

    def __repr__(self):
        t_total = (self.T_finish - self.T_start).total_seconds()
        t_alg_being_tuned = self.optAlg.total_seconds()
        t_tuner = t_total - t_alg_being_tuned - self.optAlg_processBatch.total_seconds()
        return """< FBA tuning optimization: gamma used %i ( %3.2f %% of allocated )
parameters : N %i, mutationStrength %1.2f, repeats %i
total time: %7.2fs   time FBA: %7.2fs   overhead: %3.1f%% >""" \
            % (self.gamma, 100.0 * self.gamma /self.gammaBudget,
               self.N, self.mutation_strength, self.repeats,
               t_total, t_tuner, (t_tuner/t_alg_being_tuned)*100)
        

def one_point_crossover( mom, dad, rate, verbose=False):
    n = len(mom)
    c_ind = randint(1,n)
    if verbose:
        print('    c_ind %i' % c_ind)
    bro, sis = [], []
    for i, m_x, d_x in zip(range(n), mom, dad) :
        if i < c_ind:
            bro.append(d_x)
            sis.append(m_x)
        else:
            bro.append(m_x if rand() < rate else d_x)
            sis.append(d_x if rand() < rate else m_x)
    return numpy.array(bro), numpy.array(sis)
            
def rank_curves(list_of_curves):
    curves = copy.deepcopy( list_of_curves )
    ranks = numpy.zeros( len(curves), dtype=int )
    r = 1
    while any(ranks == 0) :
        unranked_curves = numpy.array([c_i for c_i, r_i in zip(curves, ranks) if r_i == 0]) 
        c = numpy.min( unranked_curves, axis=0 )
        for i, curve_i in enumerate(curves):
            if ranks[i] == 0:
                if ( (curve_i == c) * (curve_i <> numpy.inf)).any():
                    ranks[i] = r
        r = r + 1
    return ranks.tolist()

def AUC(X, Y):
    v = 0
    for i in range( len(X)-1):
        if Y[i] <> numpy.inf :
            dx = X[i+1] - X[i]
            v = v + dx * Y[i]
    return v

def map_to_OFE_budgets( E_desired, F, E ):
    F_out = []
    i = 0
    for e in E_desired:
        while i < len(E)-1 and E[i+1] <= e:
            i = i + 1
        if E[i] <= e:
            F_out.append(F[i])
        else: #pop size bigger then e
            F_out.append(numpy.inf)
    return numpy.array(F_out)


if __name__ == "__main__":
    from matplotlib import pyplot
    print('Testing Flexible Budget Algorithm from branke2012meta article')
    print('- curve ranking - ')
    x = numpy.linspace(0,1,51)
    curves = []
    AUCs = []
    for i in range(7):
        curves.append( numpy.polyval([ rand(), -1 -rand(), 1 + rand() ], x) )
        curves[i][:3+randint(10)] = numpy.inf
        AUCs.append(AUC(x, curves[i]))
    ranks = rank_curves(curves)
    pyplot.figure()
    for c in curves:
        pyplot.plot(x,c)
    pyplot.legend(['%i   %1.3f' % (r,a) for r,a in zip(ranks,AUCs)], 
                   title='rank   AUC')
    print('- testing AUC code - ')
    OFE_budgets = numpy.array([0.0, 0.1, 0.3, 0.6, 1.0])
    F = 1 -  OFE_budgets **2
    parts = [0.1*1, 0.2*(1-0.1**2), 0.3*(1-0.3**2), 0.4 * (1-0.6 **2)]
    correct_answer = sum(parts)
    v = AUC(OFE_budgets, F)
    print('  AUC(OFE_budgets, F)   %f' % v)
    print('  correct ans           %f' % correct_answer)
    print('  diff                  %e' % (v - correct_answer))
    print('- one_point_crossover (rate=1.0) -')
    mom = numpy.array([4,3,2,1,0])
    dad = -numpy.array([5,6,7,8,9])
    print('  mom %s' % mom)
    print('  dad %s' % dad)
    for i in range(3):
        bro, sis = one_point_crossover( mom, dad, 1.0, verbose=True)
        print('    bro %s' % bro)
        print('    sis %s' % sis)
    print('- map_to_OFE_budgets -')
    e_desired = [3,5,6,8,11,15,20]
    e_actual =  [ 4, 8,12,16,20]
    F_actual =  [ 1, 2, 3, 4, 5]
    f_desired = map_to_OFE_budgets( E_desired=e_desired, F=F_actual, E=e_actual )
    print('  e_actual  %s' % e_actual)
    print('  f_actual  %s' % F_actual)
    print('  e_desired  %s' % e_desired)
    print('  f_desired  %s' % f_desired)
    print(' /correct/   [inf 1 1 2 2 3 5]')

    pyplot.show()
