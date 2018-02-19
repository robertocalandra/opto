# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt


# TODO: check if functions are vectorazible:


import R.opto.functions as functions

for i in functions:
    print(i)