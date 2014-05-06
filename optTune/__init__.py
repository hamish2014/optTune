"""
optTune - algorithms for tuning optimization algorithms under multiple objective function evaluation budgets.

    .. This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.

    .. This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    .. You should have received a copy of the GNU General Public License
       along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

__url__ = 'http://code.google.com/p/opt-tune-python-package/'

import os, pickle, math, numpy, time

def _passfunction(*args):
    pass

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

def optfromfile(fn, verbose=True) :
    if type(fn) == type('') :
        f = file(fn)
    else :
        f = fn
    if verbose:
        print('optTune: loading data file (%s , size %6.3f MB). please be patient...' % (fn ,os.path.getsize(fn)/(1024.0*1024)))
    opt = pickle.load(f)
    if type(fn) == type('') :
        f.close()
    return opt

class linearFunction:
    'convenience function which unlike lambda functions can be pickled.'
    def __init__(self, m , c):
        'y = m * x + c'
        self.m = m 
        self.c = c
    def __call__(self,x):
        return self.m * x + self.c
    def __repr__(self):
        return '<linearFunction y= %f * x  + %f >' % (self.m, self.c)
    def __eq__(self,b):
        if not isinstance(b, linearFunction):
            return False
        else:
            return self.m == b.m and self.c == b.c

from tMOPSO_code import tMOPSO
from tPSO_code import tPSO
from MOTA_code import MOTA
from MOTA_code.subproblems import MOTA_subproblem, MOTA_subproblem_weighted_sum, generate_base_subproblem_list
from FBA_code import FBA

def get_F_vals_at_specified_OFE_budgets(F, E, E_desired):
    if not type(F) == numpy.ndarray:
        F = numpy.array(F)
    if not type(E) == numpy.ndarray:
        E = numpy.array(E)
    F_out = []
    E_out = []
    if len(E) > 2 and (E[1:] - E[:-1] == E[0]).all(): #regular sampling
        N = E[1] - E[0]
        assert int(N) == N
        inds = []
        last_ind = -1
        for i, e in zip(numpy.floor((E_desired + 0.001)  / N) - 1, E_desired) :
            if 0 <= i and i < len(E) : #and i <> last_ind:
                F_out.append( F[i] )
                E_out.append( e )
            elif i >= len(E):
                F_out.append( F[-1] )
                E_out.append( e )
            last_ind = i
    else: # the slower way
        j = 0
        j_prev = -1
        for e in E_desired:
            j = max(0,j)
            while E[j] <= e and j < len(E)-1: 
                j = j + 1
            if j < len(E)-1:
                j = j - 1
            elif E[j] > e:
                j = j -1 
            if j > -1 : #and j <> j_prev: #if E[j] <= e nessary for case where e < min(E)
                F_out.append(F[j])
                E_out.append(e)
            j_prev = j
    return numpy.array(F_out), numpy.array(E_out)

class evaluation_history_recording_wrapper:
    def __init__(self, f, schedule, x_valid = lambda x:True):
        '''
        This wrapper class is written for when tMOPSO or MOTA is applied to algorithms which do NOT record their own optimization histories. It allows after the optimization is complete, the retriveal of the f_best list versus the number of evaluations used list, as required by tMOPSO method. 

        Args:
        * f - objective function to be optimize by algorithm being tuned
        * schedule - recording schedule, when applied to population based methods it should be the size of the population used.
        * x_valid - validity function, which is called with each design/decision (x) variable evaluated, and returns True if x is valid and satisfys the constraints, else if x in not valid then False.

        Apply to objective function to which the optimization algorithm being tuned is going to be applied.

        Example:
        > import numpy
        > from scipy import optimize
        > from optTune import evaluation_history_recording_wrapper
        > def objective_function(x) :
              ...
        > optFun =  evaluation_history_recording_wrapper(objective_function)
        > optimize.anneal(optFun, ...CPVs)
        > f_hist, OFE_hist = optFun.f_hist, optFun.OFE_hist

        Notes: 
        1. Written for minimization objective functions
        2. Assumes the optimization algorithm, will return the x corrensponding to best function value evaluated. This is not always the case, some times an optimization algorith return the suspected best x, without actually having evaluated it.

        '''
        self.f = f
        self.N = schedule
        self.x_valid = x_valid
        self.f_hist = []
        self.OFE_hist = []
        self.f_best = numpy.inf
        self.evalsMade = 0
    def __call__(self, x):
        self.evalsMade = self.evalsMade + 1
        r = self.f(x)
        if self.x_valid(x):
            if r < self.f_best: 
                self.f_best = r
        if self.evalsMade % self.N == 0 and self.f_best < numpy.inf:
            self.f_hist.append( self.f_best )
            self.OFE_hist.append( self.evalsMade )
        return r
