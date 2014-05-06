"""
problem definitions for population based algorithms

This module contains all the problem specific data. That is the objective function, the constaint functions, the required tolerances.

Algorithm data specific data, is not stored here. That is parameters specific to the algorithm, such as the number of itteration allowed. 

"""

import numpy, copy

def noConstraints(x) :
    "always returns an empty set, use this for no constaints"
    return numpy.zeros((0))

def printFunction(msg):
    print(msg)

def passfunction(*args):
    pass

class probDef :
    def __init__(self, f, populating_lb, populating_ub, g=noConstraints,
                 g_respawn_x=noConstraints, h=noConstraints, tol_x=0.0, 
                 tol_h=10.0**-4, box_population=False,
                 undefined_exceptions=(), undefined_print_function=printFunction,
                 addtoBatch=passfunction, processBatch=passfunction, xOpt=None):
        """
        contains problem description for gradient based optimisation problem. 
        All gradients by default, are determined using finite differences.
        
        required parameters :
          f - objective function, single or multiple
          populating_lb - hyper-rectangle lowwer bounds for generating intial population
          populating_ub - hyper-rectangle upper bounds for generating intial population           

        optional parameters :
          g - inequality constraint violation function, needs to return numpy 1D array
          g_respawn_x - CHEAP inequality constraints which if not satisfied, force the candidate design to discarded and a new one to be spawned.
          h - equality constrain function
          tol_x - solution tolerance required (def 0.0 )
          tol_h - equality constraint tolerance, (def 10.0**-4) 
          box_population - constraint particle to rectangle defined by populating bounds. Only supported by certian methods.
          undefined_exceptions - tuple of exceptions which if raised during evaluation of f,g or h constitute an undefined desig.
          undefined_print_function - print error message if exception in undefined_exceptions is raise during (f,g,h) evaluation.
          allowable_exceptions - 'try:' exceptions that constitute undefined designs. **still not fully implemented!
          addtoBatch - function to add setting for parallel processing
          processBatch - function called to process jobs from addtoBatch. f,g,h function should when called look up the results for these settings.        

        Example of use :
        prob1 = probdef(fun,numpy.zeros((30)),numpy.ones((30)),tol_x=10.0 **-3 , g=gfun ,  ...)

        'prob1' can then be passed into any population  based algorithm presented in this package
        """
        self.f =  eval_counter(f)
        self.populating_lb = populating_lb.copy()
        self.populating_ub = populating_ub.copy()
        #optional parameters
        self.g = g
        self.g_respawn_x = g_respawn_x
        self.h = h
        self.tol_x =  tol_x
        self.tol_h =  tol_h
        self.box_population = box_population
        self.undefined_exceptions = undefined_exceptions
        self.undefined_print_function = undefined_print_function
        self.addtoBatch = addtoBatch
        self.processBatch = processBatch
        if xOpt <> None:
            self.xOpt = xOpt
    def copy(self) :
        """ Assigning one instance to the other merly create a 
        another pointer to the same section of memory  """
        #return copy.deepcopy(self)
        return copy.copy(self)
    def __repr__(self) :
        s = "problemDef population-based optimisation"
        items = self.__dict__.items()
        items.sort()
        for item in items :
            s = s + "\n  " + item[0] + " : " + str(item[1])
        return s


class eval_counter :
    def __init__(self,f) :
        " wrapper class which act to count the evaluation made"
        self.f = f
        self.evals = 0
    def __call__(self,*args) :
        self.evals = self.evals + 1
        return self.f(*args)
    def __eq__(self,b):
        return self.f == b.f
    def __repr__(self) :
        if self.f.__doc__ <> None :
            return '<eval counter> for "'+ str(self.f.__doc__) + '"'
        else :
            return '<eval counter> for '+ str(self.f)

#
# testing functions
#


from numpy import arctan,sin,cos,exp,pi
test_functions_SO = []

#def Ros_ND(x) :
#    "gerneralised Rossenbroch function (30-D)"
#    return sum([100*(x[2*ii+1]-x[2*ii]**2)**2 + (1-x[2*ii])**2 for ii in range(len(x)/2)])
def Ros_ND(x) :
    "gerneralised Rossenbroch function (30-D)"
    return sum([100*(x[ii+1]-x[ii]**2)**2 + (1-x[ii])**2 for ii in range(len(x)-1)])
test_functions_SO.append(probDef(Ros_ND, 
                             numpy.ones((30)) * -2.048, 
                             numpy.ones((30)) *  2.048,
                             xOpt=numpy.ones((30))))

def Quadric_ND(x) :
    "quadratic function (30-D)"
    return sum( [ sum(x[0:ii+1])**2 for ii in range(len(x))])
test_functions_SO.append(probDef(Quadric_ND, 
                             numpy.ones((30)) * -100.0,
                             numpy.ones((30)) *  100.0,
                             xOpt=numpy.zeros((30))))

def Ackley_ND(x) :
    "Ackley testing function (30-D)"
    n = len(x)
    return -20.0 * exp(-0.2*(1.0/n *x**2).sum()**0.5) - exp(1.0/n *cos(2*pi*x).sum()) +20 + exp(1.0)
test_functions_SO.append(probDef(Ackley_ND, 
                             numpy.ones((30)) * -30,
                             numpy.ones((30)) *  30,
                             xOpt=numpy.zeros((30))))

def Rastrigin_ND(x) :
    "Rastrigin 30-D"
    return (x**2 - 10*cos(2*pi*x)+10.0).sum()
test_functions_SO.append(probDef(Rastrigin_ND, 
                             numpy.ones((30)) * -5.12,
                             numpy.ones((30)) *  5.12,
                             xOpt=numpy.zeros((30))))

def Griewank_ND(x) :
    "Griewank test function 30-D"
    return 1/4000.0 * (x ** 2).sum() - cos(x / (numpy.arange(len(x))+1)**0.5).prod() + 1.0
test_functions_SO.append(probDef(Griewank_ND, 
                             numpy.ones((30)) * -600.0,
                             numpy.ones((30)) *  600.0,
                             xOpt=numpy.zeros((30))))

def Schewefel_ND(x) :
    "Schewel testing function 30-D"
    return 418.9829 * len(x)  - (sin(abs(x))**0.5).sum()

test_functions_SO.append(probDef(Schewefel_ND, 
                             numpy.ones((30)) * -500.0,
                             numpy.ones((30)) *  500.0,
                             xOpt=numpy.zeros((30)),
                             box_population=True))

# multi-objective testings functions from MOPSO article

test_functions_MO = []

#Test function 1 , Kita mirror on F1 = -F2 axis
def Kita(x) :
    "Kita multi-objective test problem"
    k_org = numpy.array([   -x[0]**2 + x[1] ,
                   1/2*x[0]    + x[1] + 1])# the solution is multiplied by -1 as to change it to a minise
    return numpy.array([-k_org[0],-k_org[1]])
def Kita_g(x) :
    "Kita_g"
    return numpy.array([1.0/6*x[0] + x[1] - 13.0 / 2,
                          0.5*x[0] + x[1] - 15.0/ 2,
                            5*x[0] + x[1] - 30.0])
test_functions_MO.append(probDef(Kita,
                                 numpy.array([0.0,0.0]), #populating lowwer bound
                                 numpy.array([7.0,7.0]), #populating upper bound
                                 g=Kita_g,
                                 box_population=True))


#Test function 2
def Kursawe(x) :
    "Kursawe"
    if (x[0:len(x)-1] + x[1:len(x)] >= 0.0).all() :
        return numpy.array([ (-10.0*exp(-0.2*(x[0:len(x)-1] + x[1:len(x)])**0.5)).sum(),
                             (abs(x) ** 0.8 + 5 * sin(x)**3).sum()])
    else :
        raise ValueError,"Kursawe in undefined region"
Kursawe_pop_lb = numpy.ones((3)) * -5.0
Kursawe_pop_ub = numpy.ones((3)) * 5.0
test_functions_MO.append(probDef(Kursawe, Kursawe_pop_lb,
                                 Kursawe_pop_ub,  box_population=True,
                                 undefined_exceptions=(ValueError,)))


#Test function 3
def Deb(x) :
    "Deb test function"
    return numpy.array([ deb_f1(x) ,
                         deb_g_f(x) * deb_h(x)])
def deb_f1(x) :
    return x[0] 
def deb_g_f(x) :
    return numpy.array([11.0 + x[1]**2 - 10.0 * cos ( 2.0 * pi * x[1] )])
def deb_h(x) : 
    v = 1 - (deb_f1(x)/deb_g_f(x)) ** 0.5
    if not numpy.isnan(v) and deb_f1(x) <= deb_g_f(x):
        return v
    else :
        return 0.0
Deb_pop_lb = numpy.array([0,-30.0])
Deb_pop_ub = numpy.array([1.0,30.0])
test_functions_MO.append(probDef(Deb,
                                 Deb_pop_lb,
                                 Deb_pop_ub,
                                 box_population=True))


#Test function 4
def Deb2(x) :
    "Deb2 test function"
    if x[0] == 0.0 :
        raise OverflowError,'division by zero'
    return numpy.array([ x[0],
                         deb2_g_f(x) / x[0] ])
def deb2_g_f(x) :
    return 2.0 - exp(-((x[1]-0.2)/0.004)**2) - 0.8*exp(-( (x[1]-0.6)/0.4)**2)
Deb2_pop_lb = numpy.array([0,0.0])
Deb2_pop_ub = numpy.array([1.0,1.0])
test_functions_MO.append(probDef(Deb2,
                                 Deb2_pop_lb,
                                 Deb2_pop_ub,
                                 box_population=True,
                                 undefined_exceptions=(OverflowError,)))


#Test function 5
F,E,L,sig = 10.0, 2*10.0**5,200.0,10.0
def Fourbar(x) :
    "Fourbar test function"
    if x[1] < 0 :
        raise ValueError, "fourbar - x[1] less than 0"
    if x[2] < 0 :
        raise ValueError, "fourbar - x[2] less than 0"
    return numpy.array([ L * (2*x[0] + (2*x[1])**0.5 + (x[2])**0.5 + x[3]) ,
                         F*L/E * (2 / x[1] + 2*2.0**0.5 / x[1] - 2*2.0**0.5 / x[2] + 2 / x[3])])

sqrt2 = 2.0 ** 0.5

Fourbar_pop_lb = numpy.array([0.05,0.1/sqrt2,0.1/sqrt2,0.05 ])
Fourbar_pop_ub = numpy.array([0.15,0.15,0.15,0.15 ])
test_functions_MO.append(probDef(Fourbar,
                                 Fourbar_pop_lb,
                                 Fourbar_pop_ub,
                                 box_population=True,
                                 undefined_exceptions=(ValueError,)))

if __name__ == "__main__" :
    print("testing problem definitions")
    for tf in test_functions_SO :
        print(tf)
        print('  f(xOpt) : \t' + str(tf.f(tf.xOpt)))
        print('\n')
