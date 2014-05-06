'''
OptTunes standard termination crition is based on the gammaBudget, this module constrains additional termination critera
'''
import numpy, pickle
from tMOPSO_code import tMOPSO
from paretoArchives import paretoArchive2D


def noTransform(x):
    return x


class hyperVolume_stagation:
    def __init__(self, ratioThreshold , averagingIterations, f1_transform = noTransform, f2_transform =  noTransform):
        '''
        if no of iterations less then averagingIterations, then no termination
        ratioThreshold - ratio = (HV current it. - HV (it-averagingIterations)) / HV current it. 
                         HV ref used taken from upper bounds of current it. PFA
        '''
        assert ratioThreshold < 1 and ratioThreshold >= 0
        self.ratioThreshold = ratioThreshold
        self.averagingIterations = averagingIterations
        self.f1_transform = f1_transform
        self.f2_transform = f2_transform
        self.PFAs = []
    def satisfied(self, opt):
        assert len(self.PFAs) <= opt.it
        PFA_c = paretoArchive2D()
        for d in opt.PFA.designs:
            f1 = self.f1_transform(d.fv[0])
            f2 = self.f2_transform(d.fv[1])
            PFA_c.inspect_design( None, numpy.array([f1,f2]) )
        self.PFAs.append(PFA_c)
        if len(self.PFAs) > self.averagingIterations:
            del self.PFAs[0]
        if len(self.PFAs) < self.averagingIterations:
            return False
        else:
            HV_ref = PFA_c.upper_bounds()
            HV_curr = PFA_c.hyper_volume(HV_ref)
            HV_prev = self.PFAs[0].hyper_volume(HV_ref)
            ratio = (HV_curr - HV_prev) / HV_curr
            if ratio < 0:
                print("ratio of HV improvement negative! pickle.dumps follows")
                print("PFA_c", pickle.dumps(PFA_c) )
                print("self.PFAs[0]", pickle.dumps(self.PFAs[0]) )
                raise RuntimeError, "ratio of HV improvement negative!"
            if ratio <= self.ratioThreshold:
                self.msg = "hyperVolume stagation occured : ratio %1.3f ( < %1.3f ) over %i iterations" % (ratio, self.ratioThreshold, self.averagingIterations)
                return True
            else:
                #print("  hyperVolume yet to occur : ratio %1.3f ( limit %1.3f ) over %i iterations" % (ratio, self.ratioThreshold, self.averagingIterations ) )
                return False
    def __eq__(self, b):
        if not isinstance(b,hyperVolume_stagation):
            return False
        else:
            return all( self.__dict__[k] == b.__dict__[k] for k in self.__dict__.keys() if k<>'PFAs')

class hyperVolume_stagation_MOTA_sp(hyperVolume_stagation):
    def satisfied(self, sp):
        from optTune.MOTA_code import MOTA_subproblem
        assert isinstance(sp, MOTA_subproblem)
        sp_int = len(sp.PFA_history.gamma_hist)
        if sp_int <= self.averagingIterations:
            return False
        else:
            PFA_c = paretoArchive2D()
            for d in sp.PFA.designs:
                f1 = self.f1_transform(d.fv[0])
                f2 = self.f2_transform(d.fv[1])
                PFA_c.inspect_design( None, numpy.array([f1,f2]) )
            PFA_prev = paretoArchive2D()

            PFA_prev_untransformed = sp.PFA_history.get_PFA_index(sp_int-self.averagingIterations-1, sp.PFA.__class__ )
            f1_org = [ d.f1_val for d in PFA_prev_untransformed.designs]
            U_org =  [ d.U_i    for d in PFA_prev_untransformed.designs]
            for f1, U_i in zip(f1_org,  U_org):
                f2 = sp.scalarize_utility_values(U_i).mean()
                PFA_prev.inspect_design( None, numpy.array([self.f1_transform(f1),self.f2_transform(f2)]) )
            HV_ref = PFA_c.upper_bounds()
            HV_curr = PFA_c.hyper_volume(HV_ref)
            HV_prev = PFA_prev.hyper_volume(HV_ref)
            ratio = (HV_curr - HV_prev) / HV_curr
            if ratio < 0:
                print("ratio of HV improvement negative! pickle.dumps follows")
                print("PFA_c", pickle.dumps(PFA_c) )
                print("self.PFAs[0]", pickle.dumps(self.PFAs[0]) )
                raise RuntimeError, "ratio of HV improvement negative!"
            if ratio <= self.ratioThreshold:
                self.msg = "hyperVolume stagation occured : ratio %1.3f ( < %1.3f ) over %i iterations" % (ratio, self.ratioThreshold, self.averagingIterations)
                return True
            else:
                #print("  hyperVolume yet to occur : ratio %1.3f ( limit %1.3f ) over %i iterations" % (ratio, self.ratioThreshold, self.averagingIterations ) )
                return False

if __name__ == '__main__':
    print('Testing optTune termination critera')
    terminator = hyperVolume_stagation(0.1, 5)
    class optClass:
        pass
    opt = optClass()
    opt.PFA = paretoArchive2D()
    PFA_hist = []
    opt.it = 0 
    N = 5
    while True:
        for i in range(N):
            f1 = numpy.random.rand()
            f2 = numpy.random.rand()
            if f1 + f2 < 0.5:
                f1 = 1 - f1
                f2 = 1 - f2
            opt.PFA.inspect_design( None, numpy.array([f1,f2]) )

        opt.it = opt.it + 1
        print('  it %i complete, HV(opt.PFA) %f' % (opt.it,opt.PFA.hyper_volume(numpy.array([1,1])) ))
        PFA_hist.append(opt.PFA.copy())
        if terminator.satisfied(opt):
            print(terminator.msg)
            break

    #from matplotlib import pyplot
    #for i, PFA in enumerate(PFA_hist):
    #    OFEs = [ d.fv[0] for d in PFA.designs ]
    #    FVs = [ d.fv[1] for d in  PFA.designs ]
    #    pyplot.plot( OFEs, FVs, color=(0,1.0/len(PFA_hist)*(i+1), 0))
    #pyplot.xlim(0,1)
    #pyplot.ylim(0,1)
    #pyplot.show()

    print('pickle test')
    import pickle
    t2 = pickle.dumps(terminator)
    assert pickle.loads(t2) == terminator
    print('  pickle and depickle suceeded')
