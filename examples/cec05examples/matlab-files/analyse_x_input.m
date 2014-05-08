%
% octave script to test fortran implementation answers against matlab version
%
format long
tp = str2num(argv(){1});
load 'x_input.py'
x_input;
global initial_flag
initial_flag = 0;
benchmark_func(x_input, tp)