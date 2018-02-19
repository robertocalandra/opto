# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
from .Logs import Logs
from timeit import default_timer as timer

import logging
logger = logging.getLogger(__name__)

__author__ = 'Roberto Calandra'


class Optimizer(object):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        Abstract class
        :param task: R.opto.classes.OptTask -- Task to be optimized.
        :param stopCriteria: R.opto.classes.StopCriteria -- The stopping criteria for the optimization
        :param parameters: DotMap() or list of additional parameters for the optimizer:
            verbosity: 0=nothing, 1=Minimizing, 2=terminated time, 3=progress bar, 4=Iters, 5=more...
            visualize:
            indent:
        """
        # Arguments
        self.task = task  # Task to be optimized.
        self.stopCriteria = stopCriteria  # StopCriteria
        # Optional
        self.verbosity = parameters.get('verbosity', 1)
        self.visualize = parameters.get('visualize', 0)  # Warning: might substantially reduce performance!
        self.indent = parameters.get('indent', 0)
        # Internal
        self.name = 'Optimizer'  # Name of the Optimizer
        self.order = 0  # Order of the Optimizer
        self.MOO = False  # Single or multi-objective optimizer?

        self.store_full_x = True
        self._startTime = None
        self._logs = Logs(store_x=self.store_full_x)  # Structure that keep the Logs of the optimization process and related events

    def optimize(self):
        """
        Interface to start the optimization process
        :return: optimized parameters (as a 1D np.array)
        """
        if self.verbosity > 0:
            if self.task.isSOO():
                logging.info('%s %d parameters using %s' % ('Minimizing', self.task.get_n_parameters(), self.name))  # TODO: minimizing/maximizing
            else:
                logging.info('Optimizing %d objective / %d parameters using %s' % (self.task.get_n_objectives(), self.task.get_n_parameters(), self.name))
        logging.info('Optimization started')

        self._startTime = timer()

        out = self._optimize()
        # TODO: assert size parameters
        # if self.task.get_n_parameters() == 1:
        #     out = np.array(out).flatten()
        #     assert out.shape == (self.task.get_n_parameters(),), 'Internal error: wrong dimension'
        # else:
        #     assert out.shape[0] == self.task.get_n_parameters(), 'Internal error: wrong dimension'

        end = timer()
        logging.info('Optimization completed in %f[s]' % (end - self._startTime))
        logging.info('Optimization ended with flag: ')  # TODO: self.stopCriteria

        return out

    def _evaluate(self, x):
        """
        Evaluate the objective function and update the logs
        :return: 
        """

        def update_best():
            if self.task.isSOO():
                if self.task.task == {'minimize'}:
                    idx = np.argmin(self.task.get_n_objectives())
                else:
                    idx = np.argmax(self.task.get_n_objectives())
                self._logs.data.opt_fx = self._logs.get_objectives()[0, idx]
                self._logs.data.opt_x = self._logs.get_parameters()[:, idx]
            else:
                # MOO
                pass

        if self.order == 0:
            fx = self.task.evaluate(x, order=0)
            self._logs.add_evals(x=x.T, fx=fx, time=self.stopCriteria.get_time())  # store logs
            update_best()
            return fx
        if self.order == 1:
            fx, gx = self.task.evaluate(x, order=1)
            self._logs.add_evals(x=x.T, fx=fx, time=self.stopCriteria.get_time())  # store logs
            update_best()
            return fx, gx
        if self.order == 2:
            fx, gx, hx = self.task.evaluate(x, order=2)
            self._logs.add_evals(x=x.T, fx=fx, time=self.stopCriteria.get_time())  # store logs
            update_best()
            return fx, gx, hx

    def _optimize(self):
        """
        Wrapper optimization process
        :return:
        """
        raise NotImplementedError('Implement in subclass')

    # def get_f(self):
    #     """
    #     return the objective function. (LEGACY) please use self._evaluate() instead to get access to the obj.func.
    #     :return:
    #     """
    #     return self.task.f

    def get_logs(self):
        """
        Return the logs collected during the optimization
        :return:
        """
        return self._logs
