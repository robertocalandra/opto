# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import opto
from opto.functions import *
from dotmap import DotMap
from opto.opto.acq_func import *
import opto.regression as rregression
import matplotlib.pyplot as plt

import time

start = time.time()

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('example.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

task = Branin()
# task = Michalewicz()
stopCriteria = opto.opto.classes.StopCriteria(maxEvals=20)

p = DotMap()
p.verbosity = 1
# p.acq_func = UCB(model=[], logs=[], parameters={'alpha': 0.1})
p.acq_func = EI(model=None, logs=None)
# p.optimizer = opto.CMAES
p.visualize = True
p.model = rregression.GP
opt = opto.BO(parameters=p, task=task, stopCriteria=stopCriteria)
opt.optimize()
logs = opt.get_logs()
print('Number evaluations: %d' % logs.get_n_evals())
# print(logs.get_time())
print(logs.get_parameters())


if task.get_n_parameters() == 2:

    # Plot sampling points
    plt.figure()
    task.visualize(cmap=plt.cm.Greys)
    x = logs.get_parameters()
    fx = np.array(logs.get_objectives())
    color = np.arange(fx.shape[1])
    plt.scatter(np.array(x[0]), np.array(x[1]), c=color, cmap=plt.cm.coolwarm, s=50)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xlim(task.get_bounds().to_list(0))
    plt.ylim(task.get_bounds().to_list(1))

    # Plot models over time (and next point to sample)


# Parameters
plt.figure()
plt.ioff()
logs.plot_parameters()

# Optimization curve
plt.figure()
opt.plot_optimization_curve()
plt.show()
