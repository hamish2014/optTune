"""
DE module, implementation of 

@article{Storn1997differential,
    journal = {Journal of Global Optimization},
    title  = {Differential Evolution - A Simple and Efficient Heuristic for global Optimization over Continuous Spaces},
    volume  = {11},
    number = {4},
    pages = {341--359},
    year = {1997},
    author = {R. Storn and K. Price}
}
"""

import numpy, random, copy
from numpy.random import rand

def DE_opt( objfun, x_lb, x_ub, Np, F, Cr, evals, 
            x_setting='best', no_diff_vec=2, boundConstrained=False, printLevel=2):
    'returns X_min, f_best_hist, X_hist, F_hist'
    #intialization stuff
    n_dim = len( x_lb )
    def printF(level, txt):
        if level <= printLevel:
            print(txt)
    assert not boundConstrained
    it = 0
    max_it = evals / Np
    #book keeping
    X_hist = []
    F_hist = []
    f_best_hist = []
    #optimization
    printF(1,'begginning optimisation')
    while it < max_it:
        if it == 0 : #generating initial population
            X_pop = [x_lb + rand(n_dim)*(x_ub- x_lb) for j in range(Np)] #population designs
            F_pop = [objfun(x) for x in X_pop]
        else : #main loop
            x_candidates = []            
            for i in range(Np) :
                valid = False
                failCount = 0
                while not valid:
                    #mutation
                    if x_setting == 'rand' :
                        r = [] #complete random pop selection
                    elif x_setting == 'best':
                        r = [i_best]
                    else:
                        raise ValueError,'unsupported xsetting! xsetting passed in is %s' %(str(x_setting))
                    while len(r) < (1+no_diff_vec*2) :
                        r_candidate = random.randint(0,Np-1)
                        while any([r_candidate == rv for rv in r]) :
                            r_candidate = random.randint(0,Np-1)
                        r.append(r_candidate)
                    vec_mutation = numpy.zeros(n_dim)
                    for j in range(no_diff_vec) :
                        vec_mutation = vec_mutation + X_pop[r[j*2+1]] - X_pop[r[j*2+2]]
                    v = X_pop[r[0]] + F*vec_mutation
                    #cross-over
                    rv = rand(n_dim)
                    rv[random.randint(0,n_dim-1)] = 0 #index to force crossover
                    u = X_pop[i].copy()
                    u[rv<Cr] = v[rv<Cr]
                    valid = (x_lb <= u).all() and (u <= x_ub).all()
                    failCount = failCount + 1
                    if failCount == 10 and not valid:
                        u = x_lb + rand(n_dim)*(x_ub- x_lb)
                        valid = True
                x_candidates.append(u)
            for i, x_c in enumerate(x_candidates):# natural selection using greedy algorithm
                fv = objfun(x_c)
                if fv < F_pop[i] :
                    F_pop[i] = fv
                    X_pop[i] = x_c
        i_best = [ind for ind,f in enumerate(F_pop) if f == min(F_pop)][0]
        it = it + 1
        #book keeping
        X_hist.append(copy.deepcopy(X_pop))
        F_hist.append(copy.deepcopy(F_pop))
        f_best_hist.append( F_pop[i_best] )
        printF(1,'DE it %6i/%i      xdiv %f      fval %f' %( it, max_it , population_diversity(X_pop), f_best_hist[-1]) )
    return X_pop[i_best], f_best_hist, X_hist, F_hist

def population_diversity(x) :
    "take a list of x-postions and returns a diversity measure"
    xs = numpy.array(x) 
    std_vec = xs.std(axis=0)
    return numpy.linalg.norm(std_vec)

if __name__ == '__main__':
    print('tesing DE on 2D Rossenbrock')
    def Ros_ND(x) :
        "gerneralised Rossenbrock function"
        return sum([100*(x[ii+1]-x[ii]**2)**2 + (1-x[ii])**2 for ii in range(len(x)-1)])
    X_min, f_best_hist, X_hist, F_hist = DE_opt( 
        objfun=Ros_ND,
        x_lb=numpy.ones(2)*-5, 
        x_ub=numpy.zeros(2)+2, 
        Np=10, 
        F=0.5, 
        Cr=0.9, 
        evals=500 
        )
    print('X_min should be [1,1]')
    print('return %s' % X_min)
    
    from matplotlib import pyplot
    pyplot.axes([0.06,0.1,0.54,0.8])      #axes(rect) where *rect* = [left, bottom, width, height]

    delta = 0.025
    X, Y = numpy.meshgrid(numpy.arange(-5.0, 4.0, delta), numpy.arange(-5.0, 4.0, delta))
    Z = 100*(Y-X**2)**2 + (1-X)**2
    pyplot.contour(X, Y, Z, levels=[0.5, 1.0, 5, 100, 1000, 10000])
    for x in sum(X_hist, []):
        pyplot.plot(x[0],x[1],'gx')

    OFEs = 10*numpy.arange(1,len(X_hist)+1)
    pyplot.subplot(2,3,3)
    pyplot.semilogy(OFEs, f_best_hist)
    pyplot.ylabel('f_best_hist')
    pyplot.subplot(2,3,6)
    pyplot.semilogy(OFEs, [population_diversity(X) for X in X_hist] )
    pyplot.ylabel('x_div hist')
    pyplot.show()
