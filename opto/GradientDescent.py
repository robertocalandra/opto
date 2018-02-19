# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

from .opto.classes.IterativeOptimizer import IterativeOptimizer


class GradientDescent(IterativeOptimizer):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        Grid Search
        :param task:
        :param parameters:
            stepsize: scalar. Step size
            x0: np.array. initial parameters configuration.
            momentum: scalar. 
        """
        super(GradientDescent, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        self.name = 'Gradient Descent'
        self.order = 1
        self.stepsize = parameters.get('stepsize', 0.001)
        self.x0 = np.matrix(parameters.get('x0'))
        self.momentum = parameters.get('momentum', 0)
        assert self.stepsize > 0, 'stepsize must be >0'
        assert self.momentum >= 0, 'momentum must be >=0'
        assert self.x0 is not None, 'x0 must be provided!'
        assert self.x0.size == (self.task.get_n_parameters()), 'Invalid x0'
        self._delta = 0

    def _select_parameters(self):
        if self._iter == 0:
            return self.x0
        else:
            self._delta = self.stepsize * self.last_gx[0].T + self.momentum * self._delta
            return self.last_x - self._delta
