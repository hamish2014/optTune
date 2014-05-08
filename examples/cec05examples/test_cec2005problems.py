#!/usr/bin/env python

import os, sys, numpy, math, subprocess
sys.path.append('../')
import fortran_SO

print('Test script for the fortran implementations of the cec2005 problems')
print('requires octave...')
test_problems = range(1,10+1)

def compare_to_python_implementation(f_python):
    fI.fortran_SO.cec2005problems.cec_tp = tp_id
    
def cec_test_data(tp):
    fname = 'cec_data/test_data_func%i.txt' % tp
    d = numpy.fromfile(fname, sep=' ')
    assert d.shape == (510,)
    inputs = d[0:500].reshape(10,50)
    outputs = d[500:]
    return inputs, outputs

def octave_ans(tp, x):
    wdir = os.path.dirname(os.path.abspath(__file__))+'/matlab-files/'
    x.tofile(wdir+'x_input.py',sep=' ', format='%80.40f')
    stdout,stderr = subprocess.Popen('octave -q analyse_x_input.m %i' % tp,shell=True, 
                 cwd=wdir, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()
    #print(stdout)
    return float(stdout.strip().split(' ')[-1])

for tp_id in test_problems : # 1 <= tp_id <= 25
    print('Testing Cec problem %i' % (tp_id))
    fortran_SO.cec2005problems.cec_tp = tp_id
    inputs, outputs =  cec_test_data(tp_id)
    o_fortran = [ fortran_SO.cec2005problems.cec2005fun(i,len(i)) for i in  inputs ]
    o_octave =  [ octave_ans(tp_id, i)  for i in  inputs ]
    print('     correct   ' + '   '.join('%12.4f' % o for o in  outputs ) )
    print('diff fortran   ' + '   '.join('%1.4e' % o for o in (outputs - numpy.array(o_fortran))))
    print('diff octave    ' + '   '.join('%1.4e' % o for o in (outputs - numpy.array(o_octave) ) ) )

print('\n An error should be observered on problem 4, since problem 4 has noise present ...')
