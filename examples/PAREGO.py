# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import opto
from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
import opto.data as rdata
import opto.utils as rutils
from opto.functions import *

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

NAMEFILE = 'PAREGO'
rutils.create_folder(nameFolder=NAMEFILE)

# To log file
fh = logging.FileHandler(NAMEFILE + '/logs.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

task = MOP2()
stopCriteria = opto.opto.classes.StopCriteria(maxEvals=50)

p = DotMap()
p.verbosity = 1
p.visualize = 1
opt = opto.PAREGO(parameters=p, task=task, stopCriteria=stopCriteria)
opt.optimize()
logs = opt.get_logs()
logs.save(NAMEFILE + '/optimization.log')

logs = []
logs = rdata.load(NAMEFILE + '/optimization.log')

fx = logs.get_objectives()
x = logs.get_parameters()

PF_fx, PF_x = opto.opto.utils.paretoFront(fx, parameters=x)  # Compute PF
# print(PF_x)
H = opto.opto.utils.HyperVolume(fx, referencePoint=task.get_hypervolume_reference_point())  # Hypervolume
print('Hypervolume: %f' % (H))

print('Elapsed Time: %f [s]' % (logs.get_final_time()))

if task.get_n_objectives() == 2:
    plt.figure()
    plt.ioff()
    plt.scatter(fx[0], fx[1])
    opto.opto.plot.paretoFront(PF_fx, drawConnectingLines=False)
    plt.xlabel('Obj.Func. 1')
    plt.ylabel('Obj.Func. 2')

# Only works for 2D functions
if task.get_n_parameters() == 2:
    plt.figure()
    plt.scatter(np.array(x[0]).squeeze(), np.array(x[1]).squeeze(), color='blue', clip_on=False, s=50)
    plt.scatter(np.array(PF_x[0]).squeeze(), np.array(PF_x[1]).squeeze(), color='red', clip_on=False, s=80)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xlim(task.get_bounds().to_list(0))
    plt.ylim(task.get_bounds().to_list(1))
    plt.show()
