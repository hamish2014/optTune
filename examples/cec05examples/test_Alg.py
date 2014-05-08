#!/usr/bin/env python

import os, sys, numpy, time, tuning_formulations
from matplotlib import pyplot

algs = {
    'DE': { 'Np':40, 'F':0.4, 'Cr': 0.9 , 'evals':300000 },
    'PSO' : { 'N':100, 'w':0.7, 'c1': 2.0, 'c2': 2.0, 'evals':300000 },
    }

if len(sys.argv) <> 2:
    print('incorrect usage. Correct usage : test_Algs %s' % (sorted(algs.keys())))
    exit()

CPVs = algs[sys.argv[1]]
alg = {'DE':tuning_formulations.DE, 'PSO':tuning_formulations.PSO}[sys.argv[1]]
randomSeeds =  [123, 456, 987, 54321] #one run for each seed
tp_IDs = [3,5,6,8,10]

print('\nTest to see if the ``%s" implementation runs' %sys.argv[1])

for i, tp_id in enumerate(tp_IDs) :
    t = time.time()
    F = []
    for randomSeed in randomSeeds:
        F_i, OFEs_made = alg( tp_id, tfun_d=30, randomSeed=randomSeed, printLevel=1,**CPVs)
        F.append(F_i)
    timeTaken = time.time() - t
    fh = numpy.mean( F, axis = 0 ) - tuning_formulations.cec_tp_mins[tp_id]
    print('cec tp %i : fmin %14.10f' % (tp_id, fh[-1]))
    print('estimated time to do 100 runs : %i (s)' % (timeTaken*100/len(randomSeeds)))
    pyplot.subplot(2, 3, i+1)
    try:
        pyplot.loglog(OFEs_made, fh)
    except (OverflowError,ValueError), msg:
        pass
    pyplot.title("cec'05 problem %i" % tp_id)
    pyplot.xlabel('OFEs made')    
    pyplot.ylabel('fval')

pyplot.show()
