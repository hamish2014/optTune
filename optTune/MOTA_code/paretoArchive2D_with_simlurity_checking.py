'''
extensions of the paretoArchive2D class, which have a simularity checking function added

simularity_checks specifally design for MOTA.
x vals is rep are expected to be of the form
numpy.array([ log(OFE_budget) , CPV_1, CPV_2, ... ])

'''

if __name__ == '__main__':
    import sys
    sys.path.append('../../')

import numpy
from optTune.paretoArchives.paretoArchive2D_noise import paretoArchive2D_MWUT

class paretoArchive2D_polynimal_simularity_checking(paretoArchive2D_MWUT):

    poly_fit_order = 2

    def inspect_design(self, xv, f1_val, f2_vals, U_i):
        """
        inspects designs and returns True if design/decision is non-dominated in which cases its added to the current non-dominated set, or False is return if the design is dominated by the current non-dominated set
        """
        d = self.designClass(xv, f1_val, f2_vals)
        d.U_i = U_i #U_i = [ [u_1_sample1, u_1_samples2, ...], [u_2_sample1, u_2_samples2, ...], ...]
        self._inspect_design_core(d)

    def _fit_polynomials_to_PF(self, min_fitting_range=None, debug=False):
        if self.__dict__.get('poly_nod_inspected_last_fit',-1) == self.nod_inspected :
            return
        if debug:
            print('  fitting polynomials to %s' % id(self))
        self.poly_nod_inspected_last_fit = self.nod_inspected
        assert len(self.designs) > 0
        X = numpy.array([d.xv[0] for d in self.designs])
        fit_order = min(self.poly_fit_order, len(numpy.unique(X))-1) #len(numpy.unique(X))-1, protects against cases such as max(X) == min(X) and len(X) > 1, also protects against rank defficient fitting.
        self.poly_x_s = 1 / (max(X)-min(X)) if max(X)-min(X)<> 0 else 1
        self.poly_x_o = - min(X)
        self.poly_fit_x = self.poly_x_s*(X + self.poly_x_o)
        self.poly_no_cpvs = len(self.designs[0].xv) -1 
        self.poly_fits = [] #normalized fits
        self.poly_s = numpy.ones(self.poly_no_cpvs)
        self.poly_o = numpy.zeros(self.poly_no_cpvs)
        for j in range(self.poly_no_cpvs):
            p = numpy.polyfit(self.poly_fit_x, [d.xv[j+1] for d in self.designs], fit_order)
            fit_vals = numpy.polyval(p, self.poly_fit_x)
            if min_fitting_range <> None and min_fitting_range[j] > max(fit_vals) - min(fit_vals):
                self.poly_s[j] = 1 / min_fitting_range[j]
                self.poly_o[j] =  - ( fit_vals.mean() - 0.5 / self.poly_s[j] )
            else:
                self.poly_o[j] = -min(fit_vals)
                self.poly_s[j] = 1 / ( max(fit_vals) - min(fit_vals))
            self.poly_fits.append( numpy.polymul( numpy.polyadd( p , numpy.array([self.poly_o[j]])),
                                            numpy.array([self.poly_s[j]]) ) )
        if any(numpy.isnan(self.poly_o)) or any(numpy.isnan(self.poly_s)):
            print('any(numpy.isnan(self.poly_o)) or any(numpy.isnan(self.poly_s))')
            print('self.poly_o   %s' % self.poly_o)
            print('self.poly_s   %s' % self.poly_s)
            print('x_transform: scaling %s, offset %s' % (self.poly_x_s, self.poly_x_o))
            print('self.designs: ')
            for i,d in enumerate(self.designs) :
                print('  %2i:   xv %s   fv %s' % (i, str(d.xv).replace('\n',''),d.fv))
            raise RuntimeError, "Polynomial fitting failed!" 
    def simularity_to(self, repCompare, min_fitting_range=None, debug=False):
        self._fit_polynomials_to_PF( min_fitting_range, debug)
        repCompare._fit_polynomials_to_PF( min_fitting_range, debug)
        errors = []
        for j in range(self.poly_no_cpvs):
            dP = numpy.polysub(self.poly_fits[j], repCompare.poly_fits[j])
            P_se = numpy.polymul(dP,dP)
            P_sei = numpy.polyint(P_se)
            errors.append( numpy.polyval(P_sei, numpy.array([1]))[0] )
            if debug:
                print('simularity_to : control parameter %i , error %e' % (j+1, errors[-1]))
        return 1 - numpy.linalg.norm(errors) / len(errors)**0.5

    def recommend_for( self, rep_target, OFE_target):
        logB = numpy.log( OFE_target )
        x_target = rep_target.poly_x_s*(logB + rep_target.poly_x_o)
        #print('x_target %f' % x_target)
        y = []
        for j in range(self.poly_no_cpvs):
            y_i_normalized = numpy.polyval(self.poly_fits[j], numpy.array([x_target]) )[0]
            y_i =  y_i_normalized  / rep_target.poly_s[j] - rep_target.poly_o[j]
            y.append(y_i)
        return numpy.array([numpy.log( OFE_target )]+y)

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['designs'], odict['search_list']
        odict['design_f1'] = numpy.array([ d.f1_val  for  d  in  self.designs])
        odict['design_f2'] = numpy.array([ d.f2_vals for  d  in  self.designs])
        odict['design_xv'] = numpy.array([ d.xv      for  d  in  self.designs])
        odict['design_U']  =             [ d.U_i     for  d  in  self.designs]
        return odict

    def __setstate__(self, dict):
        dict['designs'] = [ self.designClass(xv,f1,f2) for xv,f1,f2 in 
                            zip(dict['design_xv'],dict['design_f1'],dict['design_f2']) ]
        for d,U_i in zip(dict['designs'], dict['design_U']):
            d.U_i = U_i
        dict['search_list'] = [d.f1_val for d in dict['designs']]
        self.__dict__.update(dict) 


class paretoArchive2D_force_zero_simularity(paretoArchive2D_polynimal_simularity_checking):
    def _fit_polynomials_to_PF(self, debug=False):
        pass
    def simularity_to(self, repCompare, debug=False):
        return 1.0 if id(self) == id(repCompare) else 0.0
    def recommend_for( self, rep_target, OFE_target ):
        return self.select_rep_h2(OFE_target)

if __name__ == '__main__':
    from matplotlib import pyplot
    print('Basic test script for paretoArchive2D_with_simularity_checking.py')
    def generate_paretoArchive(Ps, OFEs, paretoArchiveClass=paretoArchive2D_polynimal_simularity_checking):
        r = paretoArchiveClass()
        cpv_vals = zip( *[numpy.polyval(p,OFEs) for p in Ps ])
        x_vals = [ numpy.array([b] + list(c)) for b,c in zip(OFEs, cpv_vals) ]
        for xv in x_vals:
            r.inspect_design( xv, xv[0], numpy.array([[-xv[0]]]))
        return r
    paretoArchive1 = generate_paretoArchive( [numpy.array([-0.1,1,0,5]), numpy.array([-0.2,1.0])], numpy.linspace(0,5))
    paretoArchive2 = generate_paretoArchive( [numpy.array([-0.2,2,0,5]), numpy.array([0.2,0.0])], numpy.linspace(1,6))
    def plot_fit(r):
        r._fit_polynomials_to_PF(debug=True)
        pyplot.figure()
        for j in range(r.poly_no_cpvs):
            pyplot.subplot( r.poly_no_cpvs, 1, j+1 )
            pyplot.plot(r.poly_fit_x / r.poly_x_s - r.poly_x_o,  [d.xv[j+1] for d in r.designs], 'go' )
            pyplot.axis('tight')
            pyplot.twinx()
            pyplot.plot(r.poly_fit_x/ r.poly_x_s - r.poly_x_o,  numpy.polyval(r.poly_fits[j], r.poly_fit_x), '-rx' )
    plot_fit(paretoArchive1)
    plot_fit(paretoArchive2)

    print('paretoArchive1.simularity_to(paretoArchive2) %f' % paretoArchive1.simularity_to(paretoArchive2, debug=True))
    print('paretoArchive1.simularity_to(paretoArchive1) %f' % paretoArchive1.simularity_to(paretoArchive1, debug=True))


    print('paretoArchive2.recommend_for(paretoArchive1, numpy.exp(2)) %s, should be about [2, 10,0.4]' % str(paretoArchive2.recommend_for(paretoArchive1, numpy.exp(2))))

    print('pickle test to if poly_fit_order survives')
    import pickle

    paretoArchive1_clone = pickle.loads( pickle.dumps(paretoArchive1) )
    
    print('paretoArchive1_clone.poly_fit_order' , paretoArchive1_clone.poly_fit_order)

    print('test paretoArchive2D_force_zero_simularity')
    paretoArchive3 = generate_paretoArchive( [numpy.array([-0.1,1,0,5]), numpy.array([-0.2,1.0])], 
                         numpy.linspace(0,5), paretoArchive2D_force_zero_simularity )
    paretoArchive4 = generate_paretoArchive( [numpy.array([-0.2,2,0,5]), numpy.array([0.2,0.0])],
                         numpy.linspace(0,5), paretoArchive2D_force_zero_simularity  )
    print('paretoArchive3.simularity_to(paretoArchive4) %f' % paretoArchive3.simularity_to(paretoArchive4, debug=True))
    print('paretoArchive3.simularity_to(paretoArchive3) %f' % paretoArchive3.simularity_to(paretoArchive3, debug=True))

    pyplot.show()
