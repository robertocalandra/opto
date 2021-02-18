from opto.regression.classes.model import model
from dotmap import DotMap
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

import logging
logger = logging.getLogger(__name__)


class GP(model):
    def __init__(self, parameters=DotMap()):
        model.__init__(self)

        # def __init__(self, cfg: DictConfig):
        #     self._model = None
        #     self._likelihood = None
        #     self._cfg = cfg

        self.name = 'GP'  # Default value
        self.probabilistic = True  # Default value
        self.verbosity = parameters.get('verbosity', 3)
        self.indent = parameters.get('indent', 0)
        self.n_inputs = None
        self.n_outputs = None
        # self.kernel = parameters.get('kernel', 'RBF')
        # self.ARD = parameters.get('ARD', True)
        # self.fixNoise = parameters.get('fixNoise', None)

        self.normalizeOutput = parameters.get('normalizeOutput', False)  # TODO
        self.t_output = None  # Store the output transformation

        self._kernel = None
        self._model = None
        self._likelihood = None
        # self._cfg = parameters
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

        train_X = torch.from_numpy(train_set.get_input().T)
        train_Y = torch.from_numpy(train_set.get_output()).squeeze()

        # We will use the simplest form of GP model, exact inference
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[-1]))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # initialize likelihood and model
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = ExactGPModel(train_X, train_Y, self._likelihood)

        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.2)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
        training_iter = 1000

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(train_X)
            # Calc loss and backprop gradients
            loss = -mll(output, train_Y)
            loss.backward()
            logging.info('Iter %d/%d - Loss: %.3f   lengthscale1: %.3f  lengthscale2: %.3f  noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self._model.covar_module.base_kernel.lengthscale[0][0].item(),
                self._model.covar_module.base_kernel.lengthscale[0][1].item(),
                self._model.likelihood.noise.item()
            ))
            optimizer.step()

        # self._model.plot()
        # plt.show()
        end = timer()
        logging.info('Training completed in %f[s]' % (end - self._startTime))

    def _predict(self, dataset):
        n_data = dataset.get_n_data()
        # mean = np.zeros((n_data, self.n_outputs))
        # var = np.zeros((n_data, self.n_outputs))

        pred_X = torch.from_numpy(dataset.get_input().T)
        # Get into evaluation (predictive posterior) mode
        self._model.eval()
        self._likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(pred_X))
            mean = np.expand_dims(pred.mean.detach().numpy(), axis=1)
            var = np.expand_dims(pred.variance.detach().numpy(), axis=1)

        if np.any(var < 0):
            logging.warning('Variance is negative...')
            var[var < 0] = 0  # Make sure that variance is always positive

        return mean, var

    def get_hyperparameters(self):
        logging.error('Not implemented')
        return 0

