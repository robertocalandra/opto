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


class PAREGO(IterativeOptimizer):

    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        Bayesian optimization
        :param task:
        :param parameters:
        """
        super(PAREGO, self).__init__(task=task, stopCriteria=stopCriteria, parameters=parameters)
        self.name = 'ParEGO'
        self.order = 0
        self.MOO = True
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

    def scalarization(self, dataset):
        """
        Scalarize the dataset using the tchebycheff function
        :return:
        """
        logging.info('Computing Tchebycheff scalarization')
        fx = dataset.get_output()
        fx_min = np.array(np.min(fx, axis=1)).squeeze()
        fx_max = np.array(np.max(fx, axis=1)).squeeze()
        rescaled_fx = rutils.bounds(min=fx_min, max=fx_max).transform_01().transform(fx).T  # Rescale fx axis to 0-1

        lambdas = np.random.rand(self.task.get_n_objectives())
        lambdas = lambdas/np.sum(lambdas)  # Rescale such that np.lambdas == 1
        new_fx = self.tchebycheff_function(fx=rescaled_fx, lambdas=lambdas)
        out = rdata.dataset(data_input=dataset.get_input(), data_output=new_fx)
        return out

    def tchebycheff_function(self, fx, lambdas, ro=0.05):
        """
        Tchebycheff scalarization
        :param fx: matrix [N_OBJECTIVES, N_DATA]
        :param lambdas: array [N_OBJECTIVES]
        :param ro: 0.05 is the default value from [1]
        :return: Array [N_DATA]
        """
        lambda_f = lambdas * np.array(fx).T
        return np.max(lambda_f, axis=1) + ro * np.sum(lambda_f, axis=1)

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
            dataset = self.scalarization(dataset)  # Scalarize
            # print(dataset)
            p = DotMap()
            p.verbosity = 0
            self._model = self.model(parameters=p)
            self._model.train(train_set=dataset)

            # Update acquisition function
            p_acq = DotMap()
            p_acq.target = np.max(dataset.get_output())
            self.acq_func.update(model=self._model, logs=self._logs, parameters=p_acq)

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

            # # Optimize mean function (for logging purposes)
            # if self.log_best_mean:
            #     logging.info('Optimizing the mean function')
            #     task = OptTask(f=self._model.predict_mean,
            #                    n_parameters=self.task.get_n_parameters(),
            #                    n_objectives=1,
            #                    order=0,
            #                    bounds=self.task.get_bounds(),
            #                    name='Mean Function',
            #                    task={'minimize'},
            #                    labels_param=None, labels_obj=None,
            #                    vectorized=True)
            #     stopCriteria = StopCriteria(maxEvals=self.optimizer.maxEvals)
            #     p = DotMap()
            #     p.verbosity = 1
            #     mean_opt = self.optimizer.optimizer(parameters=p, task=task, stopCriteria=stopCriteria)
            #     best_x = np.matrix(acq_opt.optimize())  # Optimize
            #     best_fx = self._model.predict(dataset=best_x.T)
            #     if self._iter == 1:
            #         self._logs.data.best_m = np.matrix(best_fx[0])
            #         self._logs.data.best_v = np.matrix(best_fx[1])
            #     else:
            #         self._logs.data.best_m = np.concatenate((self._logs.data.best_m, best_fx[0]), axis=0)
            #         self._logs.data.best_v = np.concatenate((self._logs.data.best_v, best_fx[1]), axis=0)

            return x
