# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
from .Optimizer import Optimizer

import logging
logger = logging.getLogger(__name__)


class IterativeOptimizer(Optimizer):
    """
    This is an abstract class for all iterative optimizers (i.e., most of the optimizers out there)
    """
    def __init__(self, task, stopCriteria, parameters=DotMap()):
        super(IterativeOptimizer, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        # IterativeOptimizer collect a few useful info for easy use.
        self._iter = 0  # Number of iterations
        self.last_x = None  # store the last parameters evaluated
        self.last_fx = None  # and the corresponding obj.func.
        self.last_gx = None  # and eventually also the gradients (not stored in the logs)
        self.last_hx = None  # and eventually eventually also the hessian (not stored in the logs)

        self._fig = []  # Pointer to the figure draw if visualize=True
        self._objectives_curve = []
        self._parameters_curve = []

    def _select_parameters(self):
        """
        Wrapper to be implemented in the subclass
        :return: 
        """
        raise NotImplementedError('Implement in subclass')

    def _optimize(self):
        """
        
        :return: 
        """
        if self.visualize:
            self._fig = plt.figure()

        self.stopCriteria.startTime()

        for self._iter in range(self.stopCriteria.get_max_evals_iters()):
            logging.info('Iteration %d' % self._iter)

            # Select candidate parameters
            parameters = self._select_parameters()
            self.last_x = parameters

            # Evaluate parameters
            if self.order == 0:
                self.last_fx = self._evaluate(parameters)
            if self.order == 1:
                self.last_fx, self.last_gx = self._evaluate(parameters)
            if self.order == 2:
                self.last_fx, self.last_gx, self.last_gx = self._evaluate(parameters)

            # Log the iter corresponding to each parameter evaluated
            n_parameters = self.last_fx.size
            if self._iter == 0:
                self._logs.data.evals_n_iters = np.array([n_parameters*[self._iter+1]])
            else:
                self._logs.data.evals_n_iters = np.hstack((self._logs.data.evals_n_iters, np.array([n_parameters*[self._iter+1]])))

            # Logs
            idx_best = np.argmin(self.last_fx)

            if self.visualize:
                self.f_visualize()

        self._logs.data.evals_n_iters = np.array(self._logs.data.evals_n_iters)
        out = self.last_x

            # self._logs.x = parameters
            # self._logs.fx = fx
            # self._logs.data.xOpt = out
            # self._logs.data.fOpt = np.array(self.last_fx[:, idx_best])
            # self._logs.nEvals += parameters.shape[0]
            # self._logs.time = np.matrix(self.stopCriteria.get_time())

        return out

    def f_visualize(self):
        """
        Default visualization
        :return: 
        """
        if self.task.get_n_objectives() == 1:
            # SOO
            if self._iter == 0:


                # self._objectives_curve, = plt.plot(self.get_logs().get_objectives().T, linewidth=2)
                # plt.ylabel('Obj.Func.')
                # plt.xlabel('Evaluations')

                self._fig, axarr = plt.subplots(2, sharex=True)
                self._objectives_curve, = self._fig.axes[0].plot(self.get_logs().get_objectives().T, linewidth=2)
                self._fig.axes[0].set_ylabel('Obj.Func.')
                self._fig.axes[0].set_xlabel('Evaluations')
                try:
                    self._fig.axes[0].set_yscale('log')
                except:
                    pass
                self._parameters_curves = self._fig.axes[1].plot(self.get_logs().get_parameters().T, linewidth=2)
                self._fig.axes[1].set_ylabel('Parameters')
                self._fig.axes[1].set_xlabel('Evaluations')
            else:
                self._objectives_curve.set_data(np.arange(self.get_logs().get_n_evals()), self.get_logs().get_objectives().T)
                par = self.get_logs().get_parameters().T
                for i, curve in enumerate(self._parameters_curves):
                    curve.set_data(np.arange(self.get_logs().get_n_evals()), par[:, i])
                self._fig.axes[0].set_xlim(left=0, right=self.get_logs().get_n_evals())
                self._fig.axes[0].set_ylim((np.min(self.get_logs().get_objectives()), np.max(self.get_logs().get_objectives())))
                self._fig.canvas.draw()
        else:
            # MOO
            if self.task.get_n_objectives() == 2:
                # TODO: plot tradeoff curve
                pass
            else:
                # TODO: plot PF in some form
                pass

        # try:
        #     # Log scale, if possible (e.g., Obj.func. >0)
        #     ax = plt.gca()
        #     ax.set_yscale('log')
        #     plt.show()
        # except:
        #     pass

    def plot_optimization_curve(self, scale='log', plotDelta=True):
        import scipyplot as spp

        logs = self.get_logs()
        fig = plt.figure()
        # logs.plot_optimization_curve()

        if (self.task.opt_obj is None) and (plotDelta is True):
            h = plt.plot(logs.get_objectives().T, c='red', linewidth=2)
            plt.ylabel('Obj.Func.')
            # n_evals = logs.data.m.shape[0]
            # x = np.arange(start=logs.get_n_evals() - n_evals, stop=logs.get_n_evals())
            # spp.gauss_1D(y=logs.data.m, variance=logs.data.v, x=x, color='blue')
            # if self.log_best_mean:
            #     spp.gauss_1D(y=logs.data.best_m, variance=logs.data.best_v, x=x, color='green')
        else:
            h = plt.plot(logs.get_objectives().T - self.task.opt_obj, c='red', linewidth=2)
            plt.ylabel('Optimality gap')
            # n_evals = logs.data.m.shape[0]
            # x = np.arange(start=logs.get_n_evals() - n_evals, stop=logs.get_n_evals())
            # spp.gauss_1D(y=logs.data.m - self.task.opt_obj, variance=logs.data.v, x=x, color='blue')
            # if self.log_best_mean:
            #     spp.gauss_1D(y=logs.data.best_m - self.task.opt_obj, variance=logs.data.best_v, x=x, color='green')

        plt.xlabel('Evaluations')
        plt.legend(h, 'Evaluated obj.func.')
        if scale == 'log':
            ax = plt.gca()
            ax.set_yscale('log')

        # TODO: best performance expected
        # if self.log_best_mean:
        #     plt.legend(['Performance evaluated', 'performance expected', 'Best performance expected'])
        # else:
        #     plt.legend(['Performance evaluated', 'performance expected'])
        plt.show()
        return fig
