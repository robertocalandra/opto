# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

# Single-Objective Optimization
from .Branin import Branin
from .Quadratic import Quadratic

# Multi-Objective Optimization
from .MOP2 import MOP2