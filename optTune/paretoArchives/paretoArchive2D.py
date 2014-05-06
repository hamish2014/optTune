"""
Basic 2D Pareto Front archive or repository values
"""

import numpy

def dominates(a,b):
    "all(a <= b) and any(a < b), no longer used"
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0]<b[0] or a[1]<b[1]) 
    #return (a <= b).all() and (a < b).any() # to slow
    #cmp_vals = [cmp(Av,Bv) for Av,Bv in zip(a,b)]
    #return  1 not in cmp_vals and -1 in cmp_vals

class _paretoArchive_design:
    "class containing information about the design."
    def __init__(self, fv, xv):
        self.fv = fv
        self.xv = xv
    def __eq__(self, b):
        return (self.fv == b.fv).all() and (self.xv == b.xv).all()

class paretoArchive2D:
    def __init__(self):
        self.designs = []
        self.search_list = []
        self.nod_inspected = 0 #nod = number of designs
        self.nod_dominance_check_only = 0
        self.nod_rejected = 0
        self.N = 0    

    def list_loc(self, fv_dim1):
        "binary search to locate comparison point."
        search_list = self.search_list
        lb, ub = 0, len(search_list)-1
        while ub - lb > 1:
            mp = (ub + lb)/2
            if search_list[mp] < fv_dim1: 
                lb = mp # make sure lb is always less than fv_dim1, and hence non dominated ...
            else:
                ub = mp
        if search_list[ub] == fv_dim1 and search_list[lb] < fv_dim1: #if an index is thrown here then len(search_list) == 0
            return ub
        else: #search_list[lb] == fv_dim1
            return lb

    def add_design(self, fv, xv, loc, adjust_bounds):
        self.designs.insert(loc, _paretoArchive_design(fv,xv))
        self.search_list.insert(loc, fv[0])
        if adjust_bounds:
            self.lower_bound = min(self.lower_bound, fv[0])
            self.upper_bound = max(self.upper_bound, fv[0])
        self.N = self.N + 1

    def del_design(self, index):
        del self.designs[index], self.search_list[index]
        self.nod_rejected = self.nod_rejected + 1
        self.N = self.N - 1

    def inspect_design(self, xv, fv):
        """
        inspects designs and returns True if design added, or False if the design in not added,
        in other words it returns if the design is non-dominated (True) or domaninated(False)
        """
        assert len(fv) == 2
        self.nod_inspected = self.nod_inspected + 1
        if len(self.designs) == 0:
            self.designs = [_paretoArchive_design(fv,xv)]
            self.search_list = [fv[0]]
            self.lower_bound = fv[0]
            self.upper_bound = fv[0]
            self.N = 1
            return True
        if self.lower_bound <= fv[0] and fv[0] <= self.upper_bound:
            ind = self.list_loc(fv[0])
            if not dominates(self.designs[ind].fv, fv):
                if fv[0] > self.designs[ind].fv[0]:
                    self.add_design(fv,xv,ind+1,False)
                    check_ind = ind+2
                else:
                    self.add_design(fv,xv,ind,False)
                    check_ind = ind+1
                while check_ind < len(self.designs) and fv[1] < self.designs[check_ind].fv[1]:
                    self.del_design(check_ind)
                if check_ind == len(self.designs):
                    self.upper_bound = fv[0]
                return True
            else :
                self.nod_rejected = self.nod_rejected + 1
                return False 
        elif fv[0] < self.lower_bound:
            self.add_design(fv,xv,0,True)
            while 1 < len(self.designs) and fv[1] <= self.designs[1].fv[1]:
                self.del_design(1)
            if len(self.designs) == 1:
                self.upper_bound = fv[0]
            return True
        else: # self.upper_bound < fv[0]
            if fv[1] < self.designs[-1].fv[1]:
                self.add_design(fv,xv,len(self.designs),True)
                return True
            else:
                self.nod_rejected = self.nod_rejected + 1
                return False

    def inspect_multiple(self, xvals, fvals):
        "inspect multiple designs many fvals and xvals. function helps to reduce expensive grid calculations"
        return [self.inspect_design(xv,fv) for xv,fv in zip(xvals,fvals)]

    def dominates(self, fv):
        "check if front dominates fv"
        assert len(fv) == 2
        self.nod_dominance_check_only = self.nod_dominance_check_only + 1
        if len(self.designs) == 0:
            return False
        if self.lower_bound <= fv[0] and fv[0] <= self.upper_bound:
            ind = self.list_loc(fv[0])
            return self.designs[ind].fv[1] < fv[1]
        elif fv[0] < self.lower_bound:
            return True
        else:
            return self.designs[-1].fv[1] < fv[1]

    def lower_bounds(self):
        return numpy.array([self.designs[0].fv[0], self.designs[-1].fv[1]])

    def upper_bounds(self):
        return numpy.array([self.designs[-1].fv[0], self.designs[0].fv[1]])

    def hyper_volume(self, HPV_bound ):
        'Calculated the hypervolume bound between, the pareto front and an HPV_bound'
        X = [d.fv[0] for d in self.designs]
        Y = [d.fv[1] for d in self.designs]
        ref_x, ref_y = HPV_bound
        start_ind = 0
        #trimming points outside HPV_bounds
        while Y[start_ind] > ref_y and start_ind < len(Y)-1 :
            start_ind = start_ind + 1
        end_ind = len(X)-1
        while X[end_ind] > ref_x and 0 < end_ind :
            end_ind = end_ind - 1
        HV = 0.0
        if start_ind < end_ind:
            for i in range(start_ind, end_ind + 1):
                if i == end_ind:
                    dX = ref_x - X[i] 
                else:
                    dX = X[i+1] - X[i]
                HV = HV + dX * ( ref_y - Y[i] )
        if not HV >= 0.0:
            print('HV', HV)
            print('X',X)
            print('Y',Y)
            print('ref_x', ref_x, 'ref_y', ref_y)
            print('start_ind, end_ind',  start_ind, end_ind)
            raise AssertionError," Hyper Volume >= 0.0, values dumped to screen before exception raised."
        return HV

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['designs']
        odict['design_fv'] = numpy.array([d.fv for d in self.designs])
        odict['design_xv'] = numpy.array([d.xv for d in self.designs])
        return odict

    def __setstate__(self, dict):
        dict['designs'] = [ _paretoArchive_design(fv,xv) for fv,xv in zip(dict['design_fv'],dict['design_xv']) ]
        self.__dict__.update(dict) 

    def __eq__(self, b):
        'very slow ...'
        return all( b_design in self.designs for b_design in b.designs ) and  all( d in b.designs for d in self.designs )

    def __repr__(self):
        return """<lossless 2D pareto front approx. :  size: %i,  designs inspected: %i, designs rejected: %i, dominance checks %i >""" % (len(self.designs), self.nod_inspected, self.nod_rejected, self.nod_dominance_check_only + self.nod_inspected )
        
    def plot(self, key='go'):
        designs = self.designs
        xv = [d.fv[0] for d in designs]
        yv = [d.fv[1] for d in designs]
        import pylab
        pylab.plot(xv,yv,key)

    def copy(self):
        import copy
        return copy.deepcopy(self)

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

if __name__ == '__main__':
    print('Basic tests for the paretoArchive2D class')
    import pickle, time
    from matplotlib import pyplot
    plotKeys = ['g.','b+','rx','m^']*2
    class TestingPoint:
        def __init__(self, label, f1, f2):
            self.label = label
            self.fv = numpy.array( [f1, f2] )
    points = []
    def add_exp_curve(a, b, label, f1_pnts=100, f1_min=0, f1_max=1):
        for f1 in numpy.linspace(f1_min, f1_max, f1_pnts):
            points.append( TestingPoint( label, f1, numpy.exp(a*f1 +b) ) )
    add_exp_curve( -1, 0  ,   0, 100)
    add_exp_curve( -2, 0.2,   1,  90)
    add_exp_curve( -3, 0.9,   2,  80)
    add_exp_curve( -4, 1.0,   3, 120)
    add_exp_curve( -2.2, 1.1, 4, 000)

    ax1 = pyplot.subplot(1,2,1)
    for p in points:
        pyplot.plot([p.fv[0]], [p.fv[1]], plotKeys[p.label])
    pyplot.title('sample points')

    paretoArchive = paretoArchive2D()
    for p in points:
        paretoArchive.inspect_design(numpy.array([p.label]), p.fv )
    print('  %i of %i designs found to be non-dominated.' % (paretoArchive.N, len(points)))
    
    #pickling test & auditing
    assert paretoArchive == pickle.loads(pickle.dumps(paretoArchive))
    no_reals =sum([ 1 + 2 for d in paretoArchive.designs ])
    pickle_s_full = pickle.dumps(paretoArchive, protocol=1)
    template = '%20s : no reals bytes %8i, no bytes pickle str %8i, effiency  %4.3f'
    print(template % ('full pickle',no_reals*8,len(pickle_s_full),
                      1.0*no_reals*8/len(pickle_s_full)))
    print(paretoArchive)
    ax2 = pyplot.subplot(1,2,2)
    paretoArchive.plot()
    pyplot.title('Pareto non-dominated front')
    ax2.set_ylim(ax1.set_ylim())

    pyplot.show()


