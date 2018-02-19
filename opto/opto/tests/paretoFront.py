# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

from R.opto.utils.paretoFront import paretoFront

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('example.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# y = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.5, 0.5], [0.0, 0.9], [0.9, 0.0]]).T
y = np.random.rand(2, 50000)
print(y.shape)
pf = paretoFront(objectives=y, func='minimize')
print(pf)
