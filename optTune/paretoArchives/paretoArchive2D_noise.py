"""
2D pareto front data structure for noisy optimization environments.
More specifically, designed for bi-objective problems where
 - the first objective is noise-free, and 
 - only the second objective has noise present.
These 2D paretoArchives are design for problems which are typically encountered in multi-objective control parameter tuning applications.
"""

import numpy, copy
from scipy import stats

class _design_sample_prototype:
    def __init__(self, xv, f1_val, f2_vals):
        self.xv = xv
        self.f1_val = f1_val
        self.f2_vals = f2_vals
        self.fv = numpy.array( [ f1_val, f2_vals.mean() ] )
        self.init_extra()
    def init_extra(self):
        pass
    def __eq__(self, b):
        return (self.fv == b.fv).all() and (self.xv == b.xv).all()
    def dominates(self, design):
        'dominates according to direct mean values, could be acceptable large f2 samples...'
        a = self.fv
        b = design.fv
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0]<b[0] or a[1]<b[1]) 
    def probability_of_f2_being_better_than(self, design):
        raise notImplemented
    def f2_better(self, design, confidenceLevel):
        raise notImplemented
    def to_string(self):
        if not hasattr(self,'_string_repr'):
            v =  [self.f1_val] + [len(self.xv)] + self.xv.tolist() \
                + [len(self.f2_vals)] + self.f2_vals.tolist()
            if hasattr(self,'U_i'):
                v = v + [self.U_i.shape[0]]
                for i, r in enumerate(self.U_i):
                    if not numpy.isnan(r[0]):
                        v = v + [i] + r.tolist()
            self._string_repr = numpy.array(v).tostring()
        return self._string_repr

class _paretoArchive2D_noise_prototype:
    '''
    prototype/model/parent class, varing factor between class is the design/decision class :
    '''
    #designClass = design_sample
    #repr_name = '_rep2D_noise_prototype'
    def __init__(self):
        """
        needs to define at least 
        """
        
        self.designs = []
        self.search_list = []
        self.N = 0
        #counters - nod equals number of designs
        self.nod_inspected = 0 
        self.nod_dominance_check_only = 0
        self.nod_rejected = 0
        self.no_dominance_probability_checks = 0

    def list_loc(self, fv_dim1):
        "binary search to locate comparison point."
        search_list = self.search_list
        lb, ub = 0, len(search_list)-1
        while ub - lb > 1:
            mp = (ub + lb)/2
            if search_list[mp] < fv_dim1: 
                lb = mp #try make sure lb is always less than fv_dim1, and hence non dominated ...
            else:
                ub = mp
        if search_list[ub] == fv_dim1 and search_list[lb] < fv_dim1:
            return ub
        else: #search_list[lb] == fv_dim1
            return lb

    def add_design(self, design,  loc, adjust_bounds):
        self.designs.insert(loc, design )
        self.search_list.insert(loc, design.f1_val)
        if adjust_bounds:
            self.lower_bound = min(self.lower_bound, design.f1_val)
            self.upper_bound = max(self.upper_bound, design.f1_val)
        self.N = self.N + 1

    def del_design(self, index):
        del self.designs[index], self.search_list[index]
        self.nod_rejected = self.nod_rejected + 1
        self.N = self.N - 1

    def inspect_design(self, xv, f1_val, f2_vals):
        """
        inspects designs and returns True if design/decision is non-dominated in which cases its added to the current non-dominated set, or False is return if the design is dominated by the current non-dominated set
        """
        self._inspect_design_core(self.designClass(xv, f1_val, f2_vals))

    def _inspect_design_core(self, candidateDesign):
        self.nod_inspected = self.nod_inspected + 1
        f1_val = candidateDesign.f1_val
        if len(self.designs) == 0:
            self.designs = [ candidateDesign ]
            self.search_list = [f1_val]
            self.lower_bound = f1_val
            self.upper_bound = f1_val
            self.N = 1
            return True
        if self.lower_bound <= f1_val and f1_val <= self.upper_bound:
            ind = self.list_loc(f1_val)
            if not self.designs[ind].dominates(candidateDesign):
                if f1_val > self.designs[ind].f1_val :
                    self.add_design(candidateDesign, ind+1, False)
                    check_ind = ind + 2
                else:
                    self.add_design(candidateDesign, ind, False)
                    check_ind = ind + 1
                while check_ind < len(self.designs) and candidateDesign.dominates( self.designs[check_ind] ):
                    self.del_design(check_ind)
                if check_ind == len(self.designs):
                    self.upper_bound = f1_val
                return True
            else :
                self.nod_rejected = self.nod_rejected + 1
                return False 
        elif f1_val < self.lower_bound:
            self.add_design(candidateDesign, 0, True)
            while 1 < len(self.designs) and candidateDesign.dominates( self.designs[1] ):
                self.del_design(1)
            if len(self.designs) == 1:
                self.upper_bound = f1_val
            return True
        else: # self.upper_bound < fv[0]
            if not self.designs[-1].dominates(candidateDesign):
                self.add_design(candidateDesign, len(self.designs), True)
                return True
            else:
                self.nod_rejected = self.nod_rejected + 1

    def dominates(self, f1_val, f2_vals, confidenceLevel=0.90):
        """check if front dominates design/decision according to f2_mean value"""
        candidateDesign = self.designClass(None, f1_val, f2_vals)
        self.nod_dominance_check_only = self.nod_dominance_check_only + 1
        if len(self.designs) == 0 or f1_val < self.lower_bound:
            return False
        if self.lower_bound <= f1_val and f1_val <= self.upper_bound:
            ind = self.list_loc(f1_val)
        else: #f1_val > self.upper_bound
            ind = -1
        return self.designs[ind].dominates( candidateDesign )


    def probably_dominates(self, f1_val, f2_vals, confidenceLevel=0.90):
        """check if front dominates design/decision with f1_val and f2_vals is dominated at the prescibed confidence (0 - 1)"""
        candidateDesign = self.designClass(None, f1_val, f2_vals)
        self.no_dominance_probability_checks = self.no_dominance_probability_checks + 1
        if len(self.designs) == 0 or f1_val < self.lower_bound:
            return False
        if self.lower_bound <= f1_val and f1_val <= self.upper_bound:
            ind = self.list_loc(f1_val)
        else: #f1_val > self.upper_bound
            ind = -1
        return self.designs[ind].f2_better( candidateDesign , confidenceLevel )

    def dominance_probability(self, f1_val, f2_vals):
        'returns the likelihood of design (comprised of f1_val & f2_vals) being dominated'
        candidateDesign = self.designClass(None, f1_val, f2_vals)
        self.no_dominance_probability_checks = self.no_dominance_probability_checks + 1
        if len(self.designs) == 0 or f1_val < self.lower_bound:
            return 0.0
        if self.lower_bound <= f1_val and f1_val <= self.upper_bound:
            ind = self.list_loc(f1_val)
        else: #f1_val > self.upper_bound
            ind = -1
        return self.designs[ind].probability_of_f2_being_better_than( candidateDesign )

    def lower_bounds(self):
        return numpy.array([self.designs[0].fv[0], self.designs[-1].fv[1]])
    def upper_bounds(self):
        return numpy.array([self.designs[-1].fv[0], self.designs[0].fv[1]])

    def flush(self):
        del self.designs[:]

    def hyper_volume(self, HV_bound ):
        'Calculated the hypervolume bound between, the pareto front and an HV_bound'
        start_ind = 0
        #trimming points outside HV_bounds
        while self.designs[start_ind].fv[1] > HV_bound[1] and start_ind < len(self.designs)-1 :
            start_ind = start_ind + 1
        end_ind = len(self.designs)-1
        while self.designs[end_ind].fv[0] > HV_bound[0] and 0 < end_ind :
            end_ind = end_ind - 1
        HV = 0.0
        if start_ind < end_ind:
            for i in range(start_ind, end_ind + 1):
                if i == start_ind:
                    wid = HV_bound[1] - self.designs[i].fv[1] 
                else:
                    wid = self.designs[i-1].fv[1] - self.designs[i].fv[1]
                HV = HV + wid * ( HV_bound[0] - self.designs[i].fv[0])
        assert HV >= 0.0
        return HV

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['designs'], odict['search_list']
        odict['design_f1'] = numpy.array([ d.f1_val  for  d  in  self.designs])
        odict['design_f2'] = numpy.array([ d.f2_vals for  d  in  self.designs])
        odict['design_xv'] = numpy.array([d.xv for d in self.designs])
        return odict

    def __setstate__(self, dict):
        dict['designs'] = [ self.designClass(xv,f1,f2) for xv,f1,f2 in 
                            zip(dict['design_xv'],dict['design_f1'],dict['design_f2']) ]
        dict['search_list'] = [d.f1_val for d in dict['designs']]
        self.__dict__.update(dict) 

    def __eq__(self, b):
        'very slow ...'
        return all( b_design in self.designs for b_design in b.designs ) and  all( d in b.designs for d in self.designs )

    def __repr__(self):
        return """< %s :  size: %i,  designs inspected: %i, designs rejected: %i, dominance checks %i, probabily dominates checks %i >""" % (self.repr_name, len(self.designs), self.nod_inspected, self.nod_rejected, self.nod_dominance_check_only + self.nod_inspected, self.no_dominance_probability_checks )

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def copy_pareto_front_only(self):
        'removes f2_vals, as to return a paretoArchive greatly reduced in size'
        import copy
        r = self.__class__()
        for d in reversed(self.designs): #reversed for quick adding.
            r.inspect_design( copy.copy(d.xv), d.fv[0], numpy.array([d.fv[1]]) )
        return r

    def best_design(self, f1=None, f2=None):
        '''
        return the best decision/design vector according either f1 or f2 but not both!
        if f1, then design selected according to list_loc
        '''
        assert f1 <> f2
        if f1 <> None:
            ind = self.list_loc( f1 )
            return self.designs[ind].xv.copy()
        else:
            raise NotImplementedError,"f2 arguement not implemented"


class paretoArchive2D_WTT_design(_design_sample_prototype):
    'uses Welch T test to determine dominance_probability_f2'
    def init_extra(self):
        self.f2_mean = self.fv[1] # which equals f2_vals.mean()
        self.f2_std  = self.f2_vals.std(ddof=1)
    def probability_of_f2_being_better_than(self, design):
        m1 = self.f2_mean
        s1 = self.f2_std
        n1 = len(self.f2_vals)
        m2 = design.f2_mean
        s2 = design.f2_std
        n2 = len(design.f2_vals)
        t = -(m1 - m2) / ( s1**2/n1 + s2**2/n2 ) ** 0.5   
        return stats.zprob(t)
    def f2_better(self, design, confidenceLevel):
        return self.probability_of_f2_being_better_than(design) >= confidenceLevel

class paretoArchive2D_WTT(_paretoArchive2D_noise_prototype):
    designClass = paretoArchive2D_WTT_design
    repr_name = 'paretoArchive2D_WTT'


class paretoArchive2D_WTT_design_ev(_design_sample_prototype):
    '''
    Assume equal variance as to speed up f2_better calculations.
    '''
    def init_extra(self):
        self.f2_mean = self.fv[1]
        self.n = len(self.f2_vals)
    def probability_of_f2_being_better_than(self, design):
        if not hasattr( self, 'std'):
            self.std = self.f2_vals.std(ddof=1)
        std = self.std
        m1 = self.f2_mean
        m2 = design.f2_mean
        n1 = len(self.f2_vals)
        n2 = len(design.f2_vals)
        t = -(m1 - m2) * std / ( 1.0/n1 + 1.0/n2 ) **0.5
    def f2_better_init(self, confidenceLevel):
        self.confidenceLevel = confidenceLevel
        self.std = self.f2_vals.std(ddof=1)
        self.cut_off_values = self.f2_mean  + stats.norm.ppf(confidenceLevel) * self.std * ( 1.0 / self.n + 1.0 / numpy.arange(1.0, self.n + 1) ) ** 0.5
        #print( self.f2_mean, confidenceLevel,  self.cut_off_values )
    def f2_better(self, design, confidenceLevel):
        if not hasattr(self, 'confidenceLevel'):
            self.f2_better_init(confidenceLevel)
        else:
            assert self.confidenceLevel == confidenceLevel
        return self.cut_off_values[design.n-1] < design.f2_mean
        

class paretoArchive2D_WTT_ev(_paretoArchive2D_noise_prototype):   
    designClass = paretoArchive2D_WTT_design_ev
    repr_name = 'paretoArchive2D_WTT_ev'
    def plot_paretoArchive_CO(self, confidenceLevel, sampleSizes ):
        import pylab
        for d in self.designs:
            d.f2_better_init(confidenceLevel)
        ss_totals = [ sum(sampleSizes[:i+1]) for i in range(len(sampleSizes)-1) ]
        for i,ss in enumerate(ss_totals):
            gclr = 0.3 + 0.7 / len(ss_totals) * i
            x= [d.f1_val               for d in self.designs] 
            y= [d.cut_off_values[ss-1] for d in self.designs]
            pylab.plot(x, y, color=(0,gclr,0), label = "ss %2i" % ss)
        pylab.title('Cut off values for different sample sizes')
        pylab.ylabel('f2 (noisy)')
        pylab.xlabel('f1 (no noise)') 
        pylab.legend()


class _design_MWUT(_design_sample_prototype):
    ''' uses the Man-whitney u test for statisical tests '''
    def probability_of_f2_being_better_than(self, design):
        u, prob = stats.mannwhitneyu(self.f2_vals, design.f2_vals)
        if self.fv[1] < design.fv[1] :
            return 1 - prob
        else:
            return prob
    def f2_better(self, design, confidenceLevel):
        try:
            return self.probability_of_f2_being_better_than(design) >= confidenceLevel
        except ValueError, msg:
            if str(msg) == 'All numbers are identical in amannwhitneyu': #then odds problem solved to machine percision, so take already evaluated sample
                return len(self.f2_vals) < len(design.f2_vals)
            else:
                raise ValueError, msg

class paretoArchive2D_MWUT(_paretoArchive2D_noise_prototype):   
    ''' uses the Man-whitney u test for statisical tests '''
    designClass = _design_MWUT
    repr_name = 'paretoArchive2D_MWUT'



if __name__ == '__main__':
    print('Basic tests for the paretoArchive2D_noise module')
    import pickle, time
    from matplotlib import pyplot
    plotKeys = ['go','b+','rx','m^']
    class TestingPoint:
        def __init__(self, label, f1, f2_mean, f2_vals):
            self.label = label
            self.f1 = f1
            self.f2_mean = f2_mean
            self.f2_vals = f2_vals
    points = []
    def add_exp_curve(a, b, label, f2_noise_generator, f1_pnts=100, f1_min=0, f1_max=1):
        for f1 in numpy.linspace(f1_min, f1_max, f1_pnts):
            points.append( TestingPoint( label, f1, numpy.exp(a*f1 +b),
                                         numpy.exp(a*f1 +b) + f2_noise_generator() ))
    noise_fun =  lambda : numpy.random.randn(50)/10
    add_exp_curve( -1, 0  , 0, noise_fun, 100)
    add_exp_curve( -2, 0.2, 1, noise_fun,  90)
    add_exp_curve( -3, 0.9, 2, noise_fun,  80)
    add_exp_curve( -3, 1.0, 3, noise_fun,  80)


    pyplot.subplot(1,2,1)
    for p in points:
        pyplot.plot([p.f1],[p.f2_mean],plotKeys[p.label])
    pyplot.title('true mean values')
    pyplot.subplot(1,2,2)
    for p in points:
        pyplot.plot([p.f1],[p.f2_vals.mean()],plotKeys[p.label])
    pyplot.title('mean of f2 samples')

    def test_paretoArchive(paretoArchiveClass, sampleInc):
        #first construct paretoArchive based on mean values only
        print('testing %s' % paretoArchiveClass)
        paretoArchive = paretoArchiveClass()
        for p in points:
            paretoArchive.inspect_design( p.label, p.f1, p.f2_vals )
        print('  %i of %i designs found to be non-dominated.' % (paretoArchive.N, len(points)))
        #pyplot.figure()
        #for d in paretoArchive.designs:
        #    pyplot.plot([d.fv[0]], [d.fv[1]], plotKeys[d.xv])


        ss_max = len(paretoArchive.designs[0].f2_vals)       
        cL = 0.90 
        print('%10s    %20s     %20s    %20s' % ('confidenceLevel','evals. saved','false eliminations','time taken'))
        for cL in [0.4, 0.6, 0.75, 0.90, 0.95, 0.99]:
            ss = sampleInc
            mask = [True for p in points ]
            if hasattr(paretoArchive.designs[0],'f2_better_init'):
                 for d in paretoArchive.designs:
                     d.f2_better_init(cL)
            counter_saving = 0
            counter_miss = 0
            counter_time = 0
            while ss < ss_max:
                A = []
                t_mark = time.time()
                for p, check in zip(points, mask):
                    if check :
                        A.append( paretoArchive.probably_dominates( p.f1, p.f2_vals[:ss] , cL ))
                    else:
                        A.append( False )
                #stats
                counter_time = counter_time + time.time() - t_mark
                for p,check,a in zip(points, mask, A):
                    if check:
                        if a:
                            if paretoArchive.dominates( p.f1, p.f2_vals[:ss] ):
                                counter_saving = counter_saving + ss_max - ss
                            else:
                                counter_miss = counter_miss + 1
                ss = ss + sampleInc
           
            print('%8.2f    %20i     %20i    %24.5f' % (cL,counter_saving,counter_miss,counter_time))
        #pickling test & auditing
        assert paretoArchive == pickle.loads(pickle.dumps(paretoArchive))
        no_reals = sum([ 1 + 1 + len(d.f2_vals) + 2 for d in paretoArchive.designs ])
        pickle_s_full = pickle.dumps(paretoArchive, protocol=1)
        template = '%20s : no reals bytes %8i, no bytes pickle str %8i, effiency  %4.3f'
        print(template % ('full pickle',no_reals*8,len(pickle_s_full),
                          1.0*no_reals*8/len(pickle_s_full)))
        
        no_reals_pf = sum([ 1 +  1 + 1 + 2 for d in paretoArchive.designs ])
        pickle_pf_only = pickle.dumps(paretoArchive.copy_pareto_front_only(), protocol=1)
        print(template % ('pickle pf_only',no_reals_pf*8,len(pickle_pf_only),
                          1.0*no_reals_pf*8/len(pickle_pf_only)))

        print(paretoArchive)
        return paretoArchive

    test_paretoArchive( paretoArchive2D_WTT, 10 )
    test_paretoArchive( paretoArchive2D_WTT_ev, 10 )
    paretoArchive = test_paretoArchive( paretoArchive2D_MWUT, 10 )

    pyplot.show()
