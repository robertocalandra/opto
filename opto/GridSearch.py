# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
from R.opto.classes.Optimizer import Optimizer
import sys
import os


class GridSearch(Optimizer):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        Grid Search
        :param task:
        :param parameters:
            resolution: scalar or np.array. define the resolution of the grid to be evaluated.
            randomize_evaluation_order: Bool. if True, randomize the order in which the points of the grid are evaluated.
                Useful for purposes of comparing grid search against other approaches.

        """
        super(GridSearch, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        self.name = 'Grid Search'
        self.order = 0
        self.resolution = np.array(parameters.get('resolution'))
        self.randomize_evaluation_order = parameters.get('randomize_evaluation_order', False)

        # make sure that self.resolution is a np.array with size self.task.n_parameters. if instead is a scalar, tile.
        if self.resolution.size == self.task.get_n_parameters():
            pass
        else:
            self.resolution = np.tile(self.resolution, self.task.get_n_parameters())

    def _optimize(self):
        """
        Start the optimization process
        :return: Optimizer parameters
        """
        self.stopCriteria.startTime()

        # TODO: compute parameters
        n_evals = np.prod(self.resolution)
        # parameters = self.task.bounds.sample_uniform((self.stopCriteria.get_n_maxEvals(), self.task.get_n_parameters()))
        result = np.mgrid[[slice(self.task.bounds.get_min(i), self.task.bounds.get_max(i), self.resolution[i]*1j) for i in range(self.task.get_n_parameters())]]

        parameters = result.reshape(self.task.get_n_parameters(), n_evals).T

        if self.randomize_evaluation_order:
            idx = np.random.permutation(n_evals)
            fx = self._evaluate(parameters[idx])
        else:
            fx = self._evaluate(parameters)
        idx_best = np.argmin(fx)
        out = parameters[idx_best, :]

        self._logs.x = parameters
        self._logs.fx = fx
        self._logs.xOpt = out
        self._logs.fOpt = np.array(fx[:, idx_best])
        self._logs.nEvals = n_evals  # TODO: fix me
        self._logs.time = np.matrix(self.stopCriteria.get_time())

        return out