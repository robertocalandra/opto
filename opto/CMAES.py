# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import cma
import numpy as np
from dotmap import DotMap
from opto.opto.classes.Optimizer import Optimizer
import os


class CMAES(Optimizer):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """

        :param task:
        :param parameters:
        """
        super(CMAES, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        self.name = 'CMAES'
        self.order = 0
        # ----------
        self.x0 = parameters.get('x0', None)  # Initial mean
        self.sigma = parameters.get('sigma', 0.6)
        self.popsize = parameters.get('popsize', 50)  # Population size (number of evaluations at each iteration)
        self.nRestarts = parameters.get('nRestarts', 1)  # TODO: Not implemented

    def _optimize(self):
        """
        Start the optimization process
        :return: Optimizer parameters
        """
        self.stopCriteria.startTime()

        up = self.task.bounds.get_max().tolist()
        lb = self.task.bounds.get_min().tolist()
        if self.x0 is None:
            self.x0 = self.task.bounds.sample_uniform((1,))  # Randomly sample mean distribution

        def objfunc(parameters):
            return np.array(self._evaluate(np.matrix(parameters)))[:, 0][0]  # Deal with transformations from/to np.matrix

        res = cma.fmin(objfunc, self.x0.tolist(), self.sigma,
                       options={"bounds": [lb, up], "verbose": -1, "verb_disp": False,
                                "maxfevals": self.stopCriteria.get_n_maxEvals(), "popsize": self.popsize})

        # Delete log file optimizer (pretty much useless)
        try:
            os.remove('outcmaesaxlen.dat')
            os.remove('outcmaesaxlencorr.dat')
            os.remove('outcmaesfit.dat')
            os.remove('outcmaesstddev.dat')
            os.remove('outcmaesxmean.dat')
            os.remove('outcmaesxrecentbest.dat')
        except:
            # Something went wrong
            pass

        # Logs
        # self._logs.data.n_evals = res[3]
        self._logs.data.xOpt = res[0]
        self._logs.data.fOpt = np.array(res[1])
        self._logs.data.time = np.matrix(self.stopCriteria.get_time())

        # self._logs.add_evals(x=np.matrix(res[0]).T, fx=np.matrix(res[1]),
        #                      opt_x=np.matrix(res[0]).T, opt_fx=np.matrix(res[1]),
        #                      time=np.matrix(self.stopCriteria.get_time())
        #                      )
        # self._logs.n_evals = res[3]

        out = np.array(res[0])

        return out


