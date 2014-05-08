#! /usr/bin/octave -q

global initial_flag
%a = argv
%a{2}
load test_data.mat

for i=1:25 
  i
  eval(strcat('x=x', num2str(i),';'))
  eval(strcat('f=f', num2str(i),';'))
  initial_flag = 0;
  f_actual = benchmark_func(x,i);
  error = norm(f - f_actual)
end
