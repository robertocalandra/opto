# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from past.builtins import basestring
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import time
# import jsocket
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# ---
import opto.data as rdata
import opto.log as rlog
import opto.utils as rutils
import scipyplot as spp
# from R.regression.stats import stats

from dotmap import DotMap

import logging
logger = logging.getLogger(__name__)

__author__ = 'Roberto Calandra'
__version__ = '0.3'


class model(object):

    def __init__(self, parameters={}):
        if isinstance(parameters, basestring):
            # If parameters is a string, load model from file
            return self.load(parameters)
        else:
            p = parameters
            self.name = 'model'  # Default value
            self.probabilistic = False  # Default value
            self.verbosity = p.get('verbosity', 3)
            self.indent = p.get('indent', 0)
            self.n_inputs = None
            self.n_outputs = None

            # self.trainer = DotMap()
            #
            # self.path = p.get('path', '/home/rcalandra/Dropbox/Research/py-code/experiments')
            # self.checkpointName = p.get('checkpointName', os.path.join(self.path, 'tf_model'))  # Name checkpoint file
            # self.checkpointIterDelta = p.get('checkpointIterDelta', 10000)  # Number of Iterations between checkpoints
            # self.checkpointTimeDelta = p.get('checkpointTimeDelta', 600)  # Number of Seconds between checkpoints

            # self._p = None  # Pointer to parameters model
            # self._model = None  # Pointer to the model
            # self._X = None  # Pointer to input data
            # self._Y = None  # Pointer to output data
            # self._pred = None  # Pointer to output prediction
            # self._graph = None  # Pointer to the graph
            # self._saver = None  # Pointer to the saver

            # self.remoteLogger = False  # Connect to logger via TCP
            # self.remoteLoggerServer = '127.0.0.1'  # Address logger
            # self.remoteLoggerPort = 5489  # Port logger
            # self._remoteLoggerConnected = False
            # self._remoteLogger = None
        return self

    def __str__(self):
        return 'Model: "' + self.name + '" ' + str(self.n_inputs) + '->' + str(self.n_outputs)

    def isprobabilistic(self):
        return self.probabilistic

    def predict(self, dataset):
        """
        Predict the output for the given input
        :param dataset: either a Dataset.dataset or a np.matrix N_data x N_inputs
        :return: (in not probabilistic) a numpy matrix N_data x N_outputs. (if probabilistic) two matrix N_data x N_outputs
        """
        # TODO: self.model(training=False)
        # logging.info('Predicting')
        # if self.verbosity > 1:
        #     print('Predicting')
        dataset = rdata.data2dataset(dataset)  # Convert to dataset
        assert dataset.get_dim_input() == self.n_inputs, \
            'Number of covariates does not match the model %d -> %d' % (dataset.get_dim_input(), self.n_inputs)
        n_data = dataset.get_n_data()

        pred = self._predict(dataset=dataset)  # Predict

        if self.isprobabilistic():
            assert pred[0].shape == (n_data, self.n_outputs)
            assert pred[1].shape == (n_data, self.n_outputs)
        else:
            assert pred.shape == (n_data, self.n_outputs)
        return pred

    def predict_mean(self, dataset):
        """

        :param dataset:
        :return:
        """
        if self.isprobabilistic():
            mu, var = self.predict(dataset=dataset)
        else:
            mu = self.predict(dataset=dataset)
        return mu

    def save(self, nameFile, verbosity=0):
        """
        Save to file
        :param nameFile:
        :return:
        """
        rdata.save(self.__dict__, nameFile, verbosity=verbosity)

    def load(self, nameFile):
        """
        Load from file
        :param nameFile:
        :return:
        """
        self.__dict__ = rdata.load(nameFile)
        return self

    def compute_error(self, dataset, metric, probabilistic_metric=False):
        """
        
        :param dataset: 
        :param metric: 
        :param probabilistic_metric: 
        :return: 
        """
        p = self.isprobabilistic()
        out = self.predict(dataset)
        if probabilistic_metric:
            pass
            # TODO: implement me!!!
        else:
            if p:
                error = metric(dataset.output.T, out[0])
            else:
                error = metric(dataset.output.T, out)
        return error


    def plot_prediction_1D(self, x0, bounds, idx_output=0, idx_input=0, resolution_prediction=1000):
        """
        Plot the model sliced (along each dimension) around a pivot point x0
        :param x0: pivot point
        :param bounds: bounds for the predictions
        :param idx_input:
        :param resolution_prediction:
        :param interactive:
        :return:
        """
        import matplotlib.pyplot as plt
        import scipyplot as spp

        # TODO:assert bounds ot type utils.bounds
        # TODO: THIS IS not goirg to work FOR MULTI_DIMENSIONAL MODELS!!!!

        idx = idx_input
        if bounds.get_n_dim() == 1:
            X = np.matrix(np.linspace(bounds.get_min(idx).flatten(), bounds.get_max(idx).flatten(), num=resolution_prediction))
        else:
            X = np.matrix(np.tile(x0, (resolution_prediction, 1)))
            X[:, idx] = np.linspace(bounds.get_min(idx).flatten(), bounds.get_max(idx).flatten(), num=resolution_prediction)
        prediction = self.predict(X)
        if self.isprobabilistic():
            h = spp.gauss_1D(x=X[idx, :], y=prediction[0][:, idx_output], variance=prediction[1][:, idx_output],
                             color='b') # X[:, idx]
        else:
            h = plt.plot(X[idx, :], prediction[:, idx_output], color='b', linestyle='-', linewidth=2,
                         label='Prediction')
        # plt.axvline(x=x0[idx], linestyle='--', linewidth=2, color='r')

    def plot_prediction_2D(self, x0, bounds, idx_output=[0, 1], idx_input=0, resolution_prediction=100):
        """
        Plot the model sliced (along each dimension) around a pivot point x0
        :param x0: pivot point
        :param bounds: bounds for the predictions
        :param idx_input:
        :param resolution_prediction:
        :param interactive:
        :return:
        """
        import matplotlib.pyplot as plt
        import scipyplot as spp

        # TODO: assert bounds oF type utils.bounds
        # TODO: THIS IS not goirg to work FOR MULTI_DIMENSIONAL MODELS!!!!
        # TODO: implement me!!!
        # idx = idx_input
        # X = np.tile(x0, (resolution_prediction, 1))
        # X[:, idx] = np.linspace(bounds.get_min(idx), bounds.get_max(idx), num=resolution_prediction)
        # prediction = self.predict(X)
        # if self.isprobabilistic():
        #     h = spp.gauss_1D(x=X[:, idx], y=prediction[0][:, idx_output], variance=prediction[1][:, idx_output],
        #                      color='b')
        # else:
        #     h = plt.plot(X[:, idx], prediction[:, idx_output], color='b', linestyle='-', linewidth=2,
        #                  label='Prediction')
        # # plt.axvline(x=x0[idx], linestyle='--', linewidth=2, color='r')

    def visualize_prediction_1D(self, x0, bounds, idx_output=0, idx_input=None, resolution_prediction=1000, interactive=True):
        """
        Plot the model sliced (along each dimension) around a pivot point x0
        :param x0: pivot point
        :param bounds: bounds for the predictions
        :param idx_input:
        :param resolution_prediction:
        :param interactive:
        :return:
        """
        import matplotlib.pyplot as plt
        import scipyplot as spp

        # TODO:assert bounds ot type utils.bounds
        # TODO: THIS IS not goirg to work FOR MULTI_DIMENSIONAL MODELS!!!!

        def plotComparison(idx):
            if self.n_inputs > 1:
                X = np.tile(x0, (resolution_prediction, 1))
                X[:, idx] = np.linspace(bounds.get_min(idx), bounds.get_max(idx), num=resolution_prediction)
                prediction = self.predict(X)
                if self.isprobabilistic():
                    h = spp.gauss_1D(x=X[:, idx], y=prediction[0][:, idx_output], variance=prediction[1][:, idx_output],
                                     color='b')
                else:
                    h = plt.plot(X[:, idx], prediction[:, idx_output], color='b', linestyle='-', linewidth=2,
                                 label='Prediction')
                plt.axvline(x=x0[idx], linestyle='--', linewidth=2, color='r')
            else:
                X = np.linspace(bounds.get_min(), bounds.get_max(), num=resolution_prediction).T
                prediction = self.predict(X)
                if self.isprobabilistic():
                    h = spp.gauss_1D(x=X, y=prediction[0], variance=prediction[1], color='b')
                else:
                    h = plt.plot(X[:, idx], prediction, color='b', linestyle='-', linewidth=2, label='Prediction')
            # plt.legend()
            # plt.title(dataset.name)
            plt.title('%d' % (idx_output))
            plt.xlabel('%d' % (idx))
            # plt.ylabel(dataset.get_label_input(idx))

        nplots = self.n_inputs
        if interactive:
            spp.utils.interactivePlot(plotComparison, nplots=nplots)
        else:
            h = []
            # Visualize
            if idx_input is None:
                idx_input = np.arange(nplots)
            for idx in idx_input:
                fig = plt.figure()
                h.append(fig)
                ax = fig.add_subplot(1, 1, 1)
                plotComparison(idx)
            return h

    def plot_prediction_dataset(self, dataset, idx_output=None, interactive=False):
        """

        :param dataset:
        :param idx_output:
        :param interactive: Binary flag to activate the interactive mode (i.e., one single window controlled by arrows)
        :return:
        """
        import matplotlib.pyplot as plt
        import scipyplot as spp

        prediction = self.predict(dataset)  # Predict dataset

        def plotComparison(idx):
            handle = [None, None]
            handle[0] = plt.plot(dataset.output[idx, :].T, color='g', linestyle='-', linewidth=2, marker='o', label='Groundtruth')
            if self.isprobabilistic():
                handle[1] = spp.gauss_1D(y=prediction[0][:, idx], variance=prediction[1][:, idx], color='b')
                # handle[1] = plt.plot(prediction[0][:, idx], color='b', linestyle='-', linewidth=2, label='Prediction')
            else:
                handle[1] = plt.plot(prediction[:, idx], color='b', linestyle='-', linewidth=2, label='Prediction')
            plt.legend()
            plt.title(dataset.name)
            plt.ylabel(dataset.get_label_input(idx))

        nplots = dataset.get_dim_output()
        if interactive:
            spp.utils.interactivePlot(plotComparison, nplots=nplots)
        else:
            h = []
            # Visualize
            if idx_output is None:
                idx_output = np.arange(nplots)
            for idx in idx_output:
                fig = plt.figure()
                h.append(fig)
                ax = fig.add_subplot(1, 1, 1)
                plotComparison(idx)
                plt.show()
            return h

    def linearize(self, x, numeric=False):
        """

        :param x:
        :return: given the form y = Ax+B. we are returning A, B. C is the full covariance matrix in case of uncertainty
        """
        x = rdata.data2dataset(x)
        if numeric:
            pass
            # TODO: use finite differences
        else:
            A = self.gradient(x.input)
            B = self.predict(x.input) - np.multiply(A, x.input.T)
            if self.probabilistic is True:
                C = []
                # TODO: implement me
            else:
                C = []
                # TODO: implement me
            return A, B, C
