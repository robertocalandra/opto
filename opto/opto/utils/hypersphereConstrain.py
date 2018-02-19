# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt


class Constrain(object):
    def evaluate(self, x):
        pass


class hypersphereConstrain(Constrain):
    """
    Inequality constraint within an hypersphere
    """
    def __init__(self, center, radius):
        self.center = np.matrix(center)
        self.radius = radius
        #  TODO: assert self.radius is scalar

    def evaluate(self, x):
        nx = np.matrix(x)
        assert self.center.shape[0] == nx.shape[0]
        # TODO: working with multiple query points
        return -np.sum(np.power(nx - self.center, 2)) + np.power(self.radius, 2)

    def to_scipy(self):
        return {'type': 'ineq', 'fun': self.evaluate}


class constrainOR(Constrain):
    """
    inequality constraints
    """
    def __init__(self, list_constrains):
        self._c = list_constrains

    def evaluate(self, x):
        out = -np.inf
        for constrain in self._c:
            out = np.max((out, constrain.evaluate(x)))
        return out

    def to_scipy(self):
        return {'type': 'ineq', 'fun': self.evaluate}

