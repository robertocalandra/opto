# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

from R.opto.utils.bestof import bestof

x = np.matrix([0,1,1,0.5,1,2,0,-3,3,4,5]).T
out = bestof(x)

# plt.figure()
plt.ioff()
plt.plot(x, c='blue')
plt.plot(out, c='red')
plt.show()
