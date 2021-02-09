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
# task = Quadratic()
stopCriteria = opto.opto.classes.StopCriteria(maxEvals=20)


past_evals_X = np.array([[3.25756593,  1.54801184,  3.69779537,  0.07129755, -4.3186305,  -2.01051478,
   2.17360641,  5.68157622,  7.74553544,  3.79344446,  3.49123278,  3.25410127,
   6.68321546,  9.99999999,  7.2747764,   2.00450321,  6.5702423,   2.32240699,
   8.38683213,  5.0192675,   9.98593681,  2.27797444,  9.30821467,  2.21877436,
   9.99787699,  2.19235485,  3.41863485,  8.86911543,  5.08139912],
 [ 6.27597831,  9.48462245,  0.58786974, 12.51885047, 10.7847739,   6.58999514,
   4.6797018,  3.97279437,  2.76220558, 10.18053024,  1.14119365,  1.17218615,
   9.48774774, 15.,         10.81440079,  1.51050438, 11.16514301,  6.87273832,
  14.76657083,  8.83099103, 13.07655928,  1.73959835, 12.74263831,  1.6961578,
  14.99993425,  1.72707805,  7.77383538, 12.82768001, 10.35160522]])
past_evals_Y = np.array([[17.18790054,  42.01470982,   3.51796741,  63.55672947,  26.54718098,
   15.72210312,   6.89310808,  26.01132585,  12.8332667,   72.2394182,
    1.74785729,   1.49229153,  88.63904462, 145.87219095, 106.56943,
    9.27103865, 120.11847773,  18.43743451, 174.84906122,  70.12763559,
  103.62873441,   5.46527705, 107.87982565,  6.18833768, 145.91007102,
    6.38155138,  33.3107582, 118.08338632,  96.33310399]])

p = DotMap()
p.verbosity = 1
p.past_evals = [past_evals_X, past_evals_Y]
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
print(logs.get_objectives())


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
