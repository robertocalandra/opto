# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
from R.opto.classes.Optimizer import Optimizer
from R.opto.utils.paretoFront import paretoFront
import sys
import os


class RandomSearch(Optimizer):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """

        :param task:
        :param parameters:
        """
        super(RandomSearch, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        self.name = 'Random Search'
        self.order = 0

    def _optimize(self):
        """
        Start the optimization process
        :return: Optimizer parameters
        """
        self.stopCriteria.startTime()

        # Compute parameters to evaluate and evaluate them
        parameters = self.task.bounds.sample_uniform((self.stopCriteria.get_n_maxEvals(), self.task.get_n_parameters()))
        fx = self._evaluate(parameters)

        # Select best parameters
        if self.task.get_n_objectives() == 1:
            # SOO
            if self.task.task == {'minimize'}:
                idx_best = np.argmin(fx)
            if self.task.task == {'maximize'}:
                idx_best = np.argmax(fx)
            out = parameters[idx_best, :]
        else:
            # MOO --> return PF
            PF, out = paretoFront(objectives=fx, parameters=parameters.T)

        # Fill logs
        self._logs.data.time = np.matrix(self.stopCriteria.get_time())

        return out
