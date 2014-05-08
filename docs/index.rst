.. optTune documentation master file, created by
   sphinx-quickstart on Wed Jan 11 12:14:27 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to optTune's documentation!
===================================

The optimization tuning (optTune) package contains various tuning algorithms.
Many of the tuning algorithms are designed for tuning an optimization algorithm under multiple objective function evaluation (OFE) budgets.
The motivation behind optTune, is that many numerical methods have a speed-versus-accuracy trade-off, were depending upon the accuracy of the solution required different parameter settings work better.
Many optimization algorithms fall into this category, therefore the development of optTune whose algorithms are capable of tuning against this speed-versus-accuracy trade-off.

Contents:

.. toctree::
   :maxdepth: 1
 
   gettingStarted
   examples
   tMOPSO
   tPSO
   FBM
   MOTA
   speed
   quirks


Git repository : https://github.com/hamish2014/optTune

References
----------

The intended audience for optTune is the scientific and research communities.
If you have used any of the optTune codes in your work, please reference as follows:

* for MOPSO & tPSO - Dymond, A.S.D., Engelbrecht, A.P., Kok., S., and Heyns, P.S. (2014). *Tuning optimization algorithms under multiple objective function evaluation budgets*. IEEE Transactions on Evolutionary Computation. 
* for FBM/FBA - Branke, J. and Elomari, J. (2012). *Meta-optimization for parameter tuning with a flexible computing budget*. In Proceedings of the 14th International Conference on Genetic and Evolutionary Computation Conference, pages 1245-1252. ACM. 
* MOTA - still in the process of being published ...

Motivation
--------------------

Numerical optimization forms a pivotal part of many design processes.
The optimization process can be broadly broken up into the three parts of modeling, searching for the optimum of the generated model, and validation.
Searching for the optimum of the generated model, or solving the optimization problem, is the aspect of numerical optimization which is the focus here.

To solve an optimization problem, a practitioner often applies an optimization algorithm to search for the optimal design(s) or decision vector(s).
Numerous factors need to be considered when selecting an optimization algorithm and that algorithm's control parameter values (CPVs).
These factors include the characteristics of the objective function of the optimization problem, the constraints imposed, and the termination criteria used.
The characteristics of the objective function, which refer to properties such as dimensionality, degree of multi-modality, scaling, and noise presence, need to be considered as search mechanisms which are useful for certain characteristics are detrimental for others.
Sensitivity to termination criteria, which typically is imposed in the form of an objective function evaluation (OFE) budget, warrants consideration since depending on the application, OFE budgets vary widely.
Moreover, OFE budgets are influenced by the computational cost of an OFE, and the computational resources available to the practitioner, where computational resources consists of computing power multiplied by the computing time.
For success at solving an optimization problem, an optimization practitioner therefore needs to select an optimization algorithm and CPVs which are effective for the objective function, constraints, and termination criteria of the problem at hand.

Selecting an appropriate optimization algorithm and CPVs is not a trivial task, however.
Optimization algorithms and their default CPVs are typically benchmarked on standardized problems.
These problems, although exhibiting challenging and varying numerical characteristics, are not necessarily representative of the optimization problem a practitioner is engaged with.
If possible therefore, an optimization practitioner should rather aim to use an automated algorithm configuration approach, as to determine CPVs which work well on testing problems representative of the problem at hand.
Specifically, representative testing problems have similar numerical characteristics to the problem to be tackled (in terms of dimensionality, level of modality, etcetera), have similar constraints imposed, and are numerically cheap.
Central to automated algorithm configuration and other tools for performing CPV studies, is the use of tuning algorithms, i.e. optTune.
optTune provide tuning algorithm as to aid optimization practitioners in the task of selecting an optimization algorithm and CPVs which are appropriate for the problem they are engaged with.


