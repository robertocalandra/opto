# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt


def tradeoff(PF, color='red', size=50):
    nDim = PF.shape[0]
    nPoints = PF.shape[1]
    assert nDim == 2, 'This function can only plot 2D Pareto fronts'

    # Plot points
    h = plt.scatter(PF[0], PF[1], s=size, c=color, clip_on=False)

    return h