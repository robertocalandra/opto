# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import opto
import numpy as np
from opto.functions import *
from dotmap import DotMap
import matplotlib.pyplot as plt
import time
import logging

# start = time.time()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('example.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

task = Quadratic()
stopCriteria = opto.opto.classes.StopCriteria(maxEvals=10000)

p = DotMap()
p.verbosity = 1
opt = opto.RandomSearch(parameters=p, task=task, stopCriteria=stopCriteria)
opt.optimize()
logs = opt.get_logs()
print('Number evaluations: %d' % logs.get_n_evals())
print('Optimization completed in %f[s]' %(logs.get_final_time()))
# print(logs.get_parameters())

# Parameters
plt.ioff()
# plt.figure()
logs.plot_parameters()

#
plt.figure()
x = logs.get_parameters()
fx = np.array(logs.get_objectives()).T
plt.scatter(x[0], x[1], c=fx, cmap=plt.cm.jet, s=50)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.xlim(task.get_bounds().to_list(0))
plt.ylim(task.get_bounds().to_list(1))

#
plt.figure()
logs.plot_optimization_curve()
plt.show()
