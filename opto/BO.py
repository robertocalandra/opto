# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
from dotmap import DotMap
import opto.regression as rregression
import opto.data as rdata
from opto.CMAES import CMAES
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes.StopCriteria import StopCriteria
from opto.opto.acq_func import *
import opto.utils as rutils
from .opto.classes.IterativeOptimizer import IterativeOptimizer
import matplotlib.pyplot as plt
import scipyplot as spp

import logging
logger = logging.getLogger(__name__)


class BO(IterativeOptimizer):
    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        Bayesian optimization
        :param task:
        :param parameters:
        """
        super(BO, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        self.name = 'Bayesian Optimization'
        self.order = 0
        # ---
        self.acq_func = parameters.get('acq_func', EI(model=None, logs=None))
        self.optimizer = DotMap()
        self.optimizer.optimizer = parameters.get('optimizer', CMAES)
        self.optimizer.maxEvals = 20000
        self.model = parameters.get('model', rregression.GP)
        self.past_evals = parameters.get('past_evals', None)
        self.n_initial_evals = parameters.get('n_initial_evals', 10)
        self.log_best_mean = False  # optimize mean acq_func at each step

        self.store_model = True  # Should we store all models for logging purposes?
        self._model = None  # Current model
        self._logs.data.m = None
        self._logs.data.v = None
        self._logs.data.model = None

    def _select_parameters(self):
        """
        Select the next set of parameters to evaluate on the objective function
        :return: parameters: np.matrix
        """
        if (self._iter == 0) and (self.past_evals is None):
            # If there are no past evaluations, randomly initialize
            logging.info('Initializing with %d random evaluations' % self.n_initial_evals)
            self._logs.data.model = [None]
            return self.task.get_bounds().sample_uniform((self.n_initial_evals, self.task.get_n_parameters()))
        else:
            # TODO: use past_evals
            # Create model
            logging.info('Fitting response surface')
            dataset = rdata.dataset(data_input=self._logs.get_parameters(), data_output=self._logs.get_objectives())
            # print(dataset)
            p = DotMap()
            p.verbosity = 0
            self._model = self.model(parameters=p)
            self._model.train(train_set=dataset)

            # Update acquisition function
            self.acq_func.update(model=self._model, logs=self._logs)

            # Optimize acquisition function
            logging.info('Optimizing the acquisition function')
            task = OptTask(f=self.acq_func.evaluate,
                           n_parameters=self.task.get_n_parameters(),
                           n_objectives=1,
                           order=0,
                           bounds=self.task.get_bounds(),
                           name='Acquisition Function',
                           task={'minimize'},
                           labels_param=None, labels_obj=None,
                           vectorized=True)
            stopCriteria = StopCriteria(maxEvals=self.optimizer.maxEvals)
            p = DotMap()
            p.verbosity = 1
            acq_opt = self.optimizer.optimizer(parameters=p, task=task, stopCriteria=stopCriteria)
            x = np.matrix(acq_opt.optimize())  # Optimize
            fx = self._model.predict(dataset=x.T)

            # Log stuff
            if self._logs.data.m is None:
                self._logs.data.m = np.matrix(fx[0])
                self._logs.data.v = np.matrix(fx[1])
            else:
                self._logs.data.m = np.concatenate((self._logs.data.m, fx[0]), axis=0)
                self._logs.data.v = np.concatenate((self._logs.data.v, fx[1]), axis=0)
            if self.store_model:
                if self._logs.data.model is None:
                    self._logs.data.model = [self._model]
                else:
                    self._logs.data.model.append(self._model)

            # Optimize mean function (for logging purposes)
            if self.log_best_mean:
                logging.info('Optimizing the mean function')
                task = OptTask(f=self._model.predict_mean,
                               n_parameters=self.task.get_n_parameters(),
                               n_objectives=1,
                               order=0,
                               bounds=self.task.get_bounds(),
                               name='Mean Function',
                               task={'minimize'},
                               labels_param=None, labels_obj=None,
                               vectorized=True)
                stopCriteria = StopCriteria(maxEvals=self.optimizer.maxEvals)
                p = DotMap()
                p.verbosity = 1
                mean_opt = self.optimizer.optimizer(parameters=p, task=task, stopCriteria=stopCriteria)
                best_x = np.matrix(acq_opt.optimize())  # Optimize
                best_fx = self._model.predict(dataset=best_x.T)
                if self._iter == 1:
                    self._logs.data.best_m = np.matrix(best_fx[0])
                    self._logs.data.best_v = np.matrix(best_fx[1])
                else:
                    self._logs.data.best_m = np.concatenate((self._logs.data.best_m, best_fx[0]), axis=0)
                    self._logs.data.best_v = np.concatenate((self._logs.data.best_v, best_fx[1]), axis=0)

            return x

    def f_visualize(self):
        # TODO: plot also model (see plot_optimization_curve)
        if self._iter == 0:
            self._objectives_curve, = plt.plot(self.get_logs().get_objectives().T, linewidth=2, color='blue')
            # self._objectives_curve, = plt.plot(self.get_logs().get_objectives().T, linewidth=2, color='red')
            plt.ylabel('Obj.Func.')
        else:
            self._objectives_curve.set_data(np.arange(self.get_logs().get_n_evals()),
                                            self.get_logs().get_objectives().T)
            self._fig.canvas.draw()
            plt.xlim([0, self.get_logs().get_n_evals()])
            plt.ylim([np.min(self.get_logs().get_objectives()), np.max(self.get_logs().get_objectives())])

    def plot_optimization_curve(self, scale='log', plotDelta=True):
        import scipyplot as spp

        logs = self.get_logs()
        plt.figure()
        # logs.plot_optimization_curve()

        if (self.task.opt_obj is None) and (plotDelta is True):
            plt.plot(logs.get_objectives().T, c='red', linewidth=2)
            plt.ylabel('Obj.Func.')
            n_evals = logs.data.m.shape[0]
            x = np.arange(start=logs.get_n_evals() - n_evals, stop=logs.get_n_evals())
            spp.gauss_1D(y=logs.data.m, variance=logs.data.v, x=x, color='blue')
            if self.log_best_mean:
                spp.gauss_1D(y=logs.data.best_m, variance=logs.data.best_v, x=x, color='green')
        else:
            plt.plot(logs.get_objectives().T-self.task.opt_obj, c='red', linewidth=2)
            plt.ylabel('Optimality gap')
            n_evals = logs.data.m.shape[0]
            x = np.arange(start=logs.get_n_evals() - n_evals, stop=logs.get_n_evals())
            spp.gauss_1D(y=logs.data.m-self.task.opt_obj, variance=logs.data.v, x=x, color='blue')
            if self.log_best_mean:
                spp.gauss_1D(y=logs.data.best_m-self.task.opt_obj, variance=logs.data.best_v, x=x, color='green')

        plt.xlabel('Evaluations')
        if scale == 'log':
            ax = plt.gca()
            ax.set_yscale('log')


        # TODO: best performance expected
        # if self.log_best_mean:
        #     plt.legend(['Performance evaluated', 'performance expected', 'Best performance expected'])
        # else:
        #     plt.legend(['Performance evaluated', 'performance expected'])
        plt.show()
