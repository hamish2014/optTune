'''
Object for recording the Pareto-optimal Front approximations PFA.

Issues with saving/recording PF
  - size, uses lots of disk space
  - speed, in addition to the size, dipickling the PFA can greatly increase the time (and memory) requirements when loading and saving data

Considers with regard to MOTA subproblem PFA recording
  - PFA approximations may improve without any gamma (OFEs) directly used for that sub problem. This occurs when similar sub problems solution result in a PFA improvement
  - MOTA iterations may occur, when no - or only a small change occur in the PFA
'''

import pickle, numpy, difflib, copy, time
from paretoArchive2D_noise import paretoArchive2D_MWUT

def PFA_designs_to_str_list(PFA):
    return sorted(d.to_string() for d in PFA.designs)

def _parse_design_str(s):
    y = numpy.fromstring(s, dtype=numpy.float64)
    nx = int(y[1])
    no_samples = int(y[2+nx])
    if 3+nx+no_samples == len(y):
        return y[2:2+nx], y[0], y[3+nx:3+nx+no_samples]
    else:
        j = 3+nx+no_samples 
        len_F = int(y[j])
        U_i = numpy.zeros([len_F, no_samples])
        U_i.fill(numpy.nan)
        while j + 1 < len(y):
            U_i[ int(y[j+1]), : ] = y[j+2:j+2+no_samples]
            j = j + 1 + no_samples
        return y[2:2+nx], y[0], y[3+nx:3+nx+no_samples], U_i

def PFA_from_str_list(designs_str_list, repClass=paretoArchive2D_MWUT):
    PFA = repClass()
    for s in designs_str_list:
        PFA.inspect_design(*_parse_design_str(s))
    return PFA

def PFA_str_list_delta(old, new):
    '''
    order does not matter, so only need to track insertions and removals [old and new must be sorted]
    alternative approach to this is to just use Python sets
    returns [removals, insertions]
    '''
    removals = []
    insertions = [] 
    ind_old = 0
    ind_new = 0
    while ind_old < len(old) and ind_new < len(new):
        if old[ind_old] == new[ind_new]:
            ind_old = ind_old + 1
            ind_new = ind_new + 1
        elif old[ind_old] < new[ind_new]:
            removals.append(old[ind_old])
            ind_old = ind_old + 1
        else:
            insertions.append(new[ind_new])
            ind_new = ind_new + 1
    while ind_old < len(old):
        removals.append(old[ind_old])
        ind_old = ind_old + 1
    while ind_new < len(new):
        insertions.append(new[ind_new])
        ind_new = ind_new + 1
    return removals, insertions


class PFA_history_recorder_old:
    def __init__(self):
        self.gamma_hist = []
        self.changes = []
        self.current_PFA_str_list = []
        self.t__init__ = time.time()
        self.t_records = []
        self.record_processing_times = []
    def _PFA_str_diff(self, new_PFA_str):
        return PFA_str_list_delta( self.current_PFA_str_list, new_PFA_str )
    def _rollforward(self, PFA_sl, removals, insertions):
        new = [ line for line in PFA_sl if line not in removals ]
        return sorted(new + insertions)
    def record(self, gamma, new_PFA):
        t = time.time()
        self.gamma_hist.append(gamma)
        self.t_records.append( time.time() )
        if not hasattr( self, '_PFA_class'):
            self._PFA_class = new_PFA.__class__
        else:
            assert self._PFA_class == new_PFA.__class__
        new_PFA_str = PFA_designs_to_str_list(new_PFA)
        self.changes.append( self._PFA_str_diff(new_PFA_str) )
        self.current_PFA_str_list = new_PFA_str
        self.record_processing_times.append( time.time() - t )
    def __getstate__(self):
        odict = self.__dict__.copy() 
        del odict['current_PFA_str_list']   
        return odict
    def __setstate__(self, dict):
        PFA_sl = []
        for removals, insertions in dict['changes']:
            PFA_sl = self._rollforward( PFA_sl, removals, insertions )
        self.__dict__.update(dict)   # update attributes
        self.current_PFA_str_list = PFA_sl

    def __iter__(self):
        output = []
        PFA_sl = []
        for removals, insertions in self.changes:
            PFA_sl = self._rollforward( PFA_sl, removals, insertions )
            yield PFA_from_str_list(PFA_sl, self._PFA_class)


    def map(self, f):
        '''
        effictivenly does map(f, PFA_hist). 
        Process starting from last PFA, f(PFA), revent last PFA to 2nd last PFA, f(PFA_second_last)...
        Finally returns [f(PFA_first), f(PFA_second), ..., ]
        '''
        #output = []
        #PFA_sl = []
        #for removals, insertions in self.changes:
        #    PFA_sl = self._rollforward( PFA_sl, removals, insertions )
        #    output.append(f(PFA_from_str_list(PFA_sl, self._PFA_class))) #if error with no attribure self._PFA_class, then set attribute.
        #return output
        return [ f(PFA) for PFA in self ]

    def uncompressed_history(self):
        'If results in memory problems consider using _PFA_history_recorder.map instead'
        return [ PFA for PFA in self ]

    def get_PFA_index(self, index):
        ''' zero based index, i.e. ind \in 0,1,2,...,len(self.changes)-1'''
        PFA_sl = []
        assert 0 <= index and index < len(self.changes)
        for removals, insertions in self.changes[:index+1]:
            PFA_sl = self._rollforward( PFA_sl, removals, insertions )
        return PFA_from_str_list(PFA_sl, self._PFA_class) #if error with not attribure self._PFA_class, then set attribute.
    def get_PFA_gamma(self, gamma):
        'return PFA for larget g <= gamma for g in self.gamma_hist'
        if min(self.gamma_hist) > gamma: 
            raise ValueError,"min gamma_hist is %i, gamma requested is %i" % (min(self.gamma_hist), gamma)
        elif max(self.gamma_hist) < gamma:
            raise ValueError,"max gamma_hist is %i, gamma requested is %i" % (max(self.gamma_hist), gamma)
        return self.get_PFA_index( sum( numpy.array(self.gamma_hist) <= gamma ) - 1 )
    def plot(self, marker='o',
             makersize_fun = lambda i,n_i : (6**2 + 200.0*i/(n_i-1)) ** 0.5,
             mfc_fun = lambda i,n_i : (0, 0.3 + 0.7*i/(n_i-1), 0)):
        'basisc example of howto plot PFA history, modify and personalize for own use'
        from matplotlib import pyplot
        n = len(self.changes)
        def get_fvals(PFA):
            return [d.fv.copy() for d in PFA.designs]
        for i,pnts in enumerate(self.map(get_fvals)) :
            x= [p[0] for p in pnts] 
            y= [p[1] for p in pnts]
            pyplot.plot(x, y, linestyle='None', marker=marker, 
                        markersize=makersize_fun(i,n), 
                        mfc=mfc_fun(i,n))


def PFA_delta_lean(old, new):
    removed_inds = []
    insertions = [] 
    ind_old = 0
    ind_new = 0
    while ind_old < len(old) and ind_new < len(new):
        if old[ind_old] == new[ind_new]:
            ind_old = ind_old + 1
            ind_new = ind_new + 1
        elif old[ind_old] < new[ind_new]:
            removed_inds.append(ind_old)
            ind_old = ind_old + 1
        else:
            insertions.append(new[ind_new])
            ind_new = ind_new + 1
    while ind_old < len(old):
        removed_inds.append(ind_old)
        ind_old = ind_old + 1
    while ind_new < len(new):
        insertions.append(new[ind_new])
        ind_new = ind_new + 1
    return removed_inds, insertions

class PFA_history_recorder(PFA_history_recorder_old):
    def _PFA_str_diff(self, new_PFA_str):
        return PFA_delta_lean( self.current_PFA_str_list, new_PFA_str )
    def _rollforward(self, PFA_sl, removed_inds, insertions):
        new = [ line for j,line in enumerate(PFA_sl) if j not in removed_inds ]
        return sorted(new + insertions)


if __name__ == '__main__':
    import copy
    from matplotlib import pyplot
    print('PF recorder testing script')
    print('testing PFA_str_list_delta function')
    s1 = sorted(['apples','fruits','roses','zemons'])
    s2 = sorted(['grapes','roses','yaks'])
    removals, insertions = PFA_str_list_delta(s1,s2)
    print('  s1 : %s' % str(s1))
    print('  s2 : %s' % str(s2))
    print('  removals : %s' % removals)
    print('  insertions : %s' % insertions)

    def test_PFA_recorder(PFA_recorder_class):
        print('testing PFA_recorder : %s' % str(PFA_recorder_class))
        pf = paretoArchive2D_MWUT()
        pf_hist = []
        pf_recorder = PFA_recorder_class()
        for i in range(100):
            if False:
                f2_vals = numpy.random.rand(10)
                f1 = 1 - f2_vals.mean()
                pf.inspect_design( numpy.array([i]), f1, f2_vals )
            else:
                pf.inspect_design( numpy.array([i]), numpy.random.rand(), numpy.random.rand(10) )
            if (i+1) % 10 == 0:
                print('  %i points inspected, recording PF approximation, PF size %i' % (i + 1, len(pf.designs)))
               #print(pf)
                pf_hist.append( copy.deepcopy(pf) ) 
                pf_recorder.record(i+1,pf)
        HV_bound = numpy.array([1.0, 1.0])
        HV_hist = [ p.hyper_volume(HV_bound) for p in pf_hist ]
        HV_recorder = pf_recorder.map( lambda f: f.hyper_volume(HV_bound) )
        #print('Computational Overhead')
        time_PFA = sum(pf_recorder.record_processing_times)
        time_total = pf_recorder.t_records[-1] - pf_recorder.t__init__
        print('  time processing   %f        time running %f       overhead %3.2f%%' % ( time_PFA, time_total, 100*time_PFA / (time_total - time_PFA)))
        print('Comparison between copy.deepcopy list and pf_recorder')
        print('\t\th1\t\th2\t\tdetla')
        for h1,h2 in zip(HV_hist, HV_recorder ):
            print('\t%1.2e\t%1.2e\t%f' % (h1,h2,abs(h1-h2)))
        print('  pf_hist == pf_recorder.uncompressed_history() : %s' % (pf_hist == pf_recorder.uncompressed_history()))
        pyplot.figure()
        pf_recorder.plot()
        print('pickle test')
        pf_recorder_dump = pickle.dumps(pf_recorder)
        pf_recorder_load = pickle.loads(pf_recorder_dump)
        print('  pf_recorder_load.uncompressed_history() ==  pf_hist : %s' % (pf_recorder_load.uncompressed_history() ==  pf_hist))
        print('pickle size tests')
        for p in [0,1,2]:
            len1 = len(pickle.dumps(pf_recorder,p))
            len2 = len(pickle.dumps(pf_hist,p))
            print('      len(pickle.dumps(pf_recorder,%i))      %5i       len( pickle.dumps(pf_hist,%i))      %5i       reduction   %1.2f' % (p,len1,p,len2,1.0*len2/len1))
        return pf_recorder
    test_PFA_recorder(PFA_history_recorder_old)
    pf_recorder = test_PFA_recorder(PFA_history_recorder)
    pyplot.show()
