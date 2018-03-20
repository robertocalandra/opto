# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from opto.regression.classes.model import model
import GPy
from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

import logging
logger = logging.getLogger(__name__)


class GP(model):
    def __init__(self, parameters=DotMap()):
        model.__init__(self)
        self.name = 'GP'  # Default value
        self.probabilistic = True  # Default value
        self.verbosity = parameters.get('verbosity', 3)
        self.indent = parameters.get('indent', 0)
        self.n_inputs = None
        self.n_outputs = None
        self.kernel = parameters.get('kernel', 'Matern52')
        self.ARD = parameters.get('ARD', True)
        self.fixNoise = parameters.get('fixNoise', None)

        self.normalizeOutput = parameters.get('normalizeOutput', False)  # TODO
        self.t_output = None  # Store the output transformation

        self._kernel = []
        self._model = []
        self._logs = []
        self._startTime = None

    def train(self, train_set):
        logging.info('Training GP')
        self._startTime = timer()

        if self.normalizeOutput is True:
            self.t_output = train_set.normalize_output()
        self.n_inputs = train_set.get_dim_input()
        self.n_outputs = train_set.get_dim_output()
        logging.info('Dataset %d -> %d with %d data' % (self.n_inputs, self.n_outputs, train_set.get_n_data()))

        for i in range(self.n_outputs):
            if self.kernel == 'Matern52':
                self._kernel.append(GPy.kern.Matern52(input_dim=train_set.get_dim_input(), ARD=self.ARD))
            if self.kernel == 'Linear':
                self._kernel.append(GPy.kern.Linear(input_dim=train_set.get_dim_input(), ARD=self.ARD))
            self._model.append(GPy.models.GPRegression(train_set.get_input().T, train_set.get_output()[i, :].T, kernel=self._kernel[i]))
            if self.fixNoise is not None:
                self._model[i].likelihood.variance.fix(self.fixNoise)
            self._model[i].optimize_restarts(num_restarts=10, verbose=False)  # , parallel=True, num_processes=5
            # self._model.optimize(messages=True)

            # hmc = GPy.inference.mcmc.HMC(self._model, stepsize=2e-1)
            # # run hmc
            # t = hmc.sample(num_samples=20000, hmc_iters=20)

        # self._model.plot()
        # plt.show()
        end = timer()
        logging.info('Training completed in %f[s]' % (end - self._startTime))

    def _predict(self, dataset):
        n_data = dataset.get_n_data()
        mean = np.zeros((n_data, self.n_outputs))
        var = np.zeros((n_data, self.n_outputs))
        for i in range(self.n_outputs):
            t_mean, t_var = self._model[i].predict(np.array(dataset.get_input().T))
            mean[:, i] = t_mean.T
            var[:, i] = t_var.T
        if np.any(var < 0):
            # logging.info('Variance is negative...')
            var[var < 0] = 0  # Make sure that variance is always positive
        return mean, var

    def get_hyperparameters(self):
        return self._model._param_array_

