# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import DIRECT as direct
import numpy as np
from dotmap import DotMap
from R.opto.classes.Optimizer import Optimizer
import sys
import os

import logging
logger = logging.getLogger(__name__)


class DIRECT_wrap(Optimizer):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        DIRECT optimizer
        :param task:
        :param parameters:
        """
        super(DIRECT_wrap, self).__init__(task=task, parameters=parameters, stopCriteria=stopCriteria)
        self.name = 'DIRECT'
        self.order = 0

    def _optimize(self):
        """
        Start the optimization process
        :return: Optimizer parameters
        """

        self.stopCriteria.startTime()  # Start the clock

        def objfunc(parameters, user_data):
            return self._evaluate(np.matrix(parameters)), 0  # Deal with transformations from/to np.matrix

        try:
            if self.verbosity < 2:
                fileno = sys.stdout.fileno()
                with os.fdopen(os.dup(fileno), 'wb') as stdout:
                    with os.fdopen(os.open(os.devnull, os.O_WRONLY), 'wb') as devnull:
                        sys.stdout.flush()
                        os.dup2(devnull.fileno(), fileno)  # redirect
                        x, fx, _ = direct.solve(objfunc,
                                                l=[self.task.bounds.get_min()],
                                                u=[self.task.bounds.get_max()],
                                                maxT=600,  # self.stopCriteria.get_n_maxIters()
                                                maxf=self.stopCriteria.get_n_maxEvals(),
                                                )
                    sys.stdout.flush()
                    os.dup2(stdout.fileno(), fileno)  # restore

            else:
                x, fx, _ = direct.solve(objfunc,
                                        l=[self.task.bounds.get_min()],
                                        u=[self.task.bounds.get_max()],
                                        maxT=600,  # self.stopCriteria.get_n_maxIters()
                                        maxf=self.stopCriteria.get_n_maxEvals(),
                                        )
        except:
            logging.warning('DIRECT had some problem and failed...')

        # Delete log file optimizer (pretty much useless)
        try:
            os.remove('DIRresults.txt')
        except:
            logging.warning('Unable to remove file: DIRresults.txt')

        # Logs
        # self._logs.xOpt = x
        # self._logs.fOpt = np.array(fx)
        # self._logs.time = np.matrix(self.stopCriteria.get_time())
        # self._logs.add_evals(x=np.matrix(x).T, fx=np.matrix(fx),
        #                      opt_x=np.matrix(x).T, opt_fx=np.matrix(fx),
        #                      time=np.matrix(self.stopCriteria.get_time())
        #                      )

        return x


