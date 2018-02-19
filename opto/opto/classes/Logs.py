# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dotmap import DotMap
import opto.data as rdata


class Logs(object):

    def __init__(self, store_x=True, store_fx=True, store_gx=False):
        """
        Class to store all the logs from an optimization process
        """

        self.data = DotMap()
        self.data.x = None  # matrix with all the parameters evaluated
        self.data.fx = None  # matrix with the corresponding obj.func.
        self.data.opt_x = None  # best parameters find so far
        self.data.opt_fx = None  # best obj. value find so far
        self.data.n_evals = 0  # number of evaluations performed
        self.data.evals_n_iters = None  # store the iteration at which each evaluation was performed
        self.data.time = None
        # More fields can be added to self.data dynamically at running time, depending on specific needs

        self.store_x = store_x  # Should we store the full history of variables x evaluated?
        self.store_fx = store_fx  # Should we store the full history of Obj.values fx measured?
        self.store_gx = store_gx  # Should we store the full history of gradients measured?

    def get_last_evals(self):
        return self.last_fx, self.last_x

    def add_evals(self, x=None, fx=None, gx=None, opt_x=None, opt_fx=None, nIter=None, time=None, opt_criteria='minima'):
        """

        :param x: matrix (N_PARAMETERS x N_DATA)
        :param fx: matrix (N_OBJ_FUNC x N_DATA)
        :param opt_fx: optimum fx_ so far
        :param opt_x: optimum x_ so far
        :param nIter: number of the current iteration
        :param time
        :return:
        """
        assert x.ndim == 2, 'x must be a np.matrix (N_PARAMETERS x N_DATA)'
        assert x.shape[1] == fx.shape[1]
        n_evals = x.shape[1]
        if fx is not None:
            assert fx.ndim == 2, 'fx must be a np.matrix (N_OBJ_FUNC x N_DATA)'
        if gx is not None:
            assert gx.ndim == 3, 'fx must be a np.matrix (N_OBJ_FUNC x N_PARAMETERS x N_DATA)'

        if (self.data.x is None) and (self.data.fx is None):  # TODO: fix this condition
            # Initialize
            if self.store_x:
                self.data.x = x
            if self.store_fx:
                self.data.fx = fx
            if self.store_gx:
                self.data.gx = gx
        else:
            if self.store_x:
                self.data.x = np.concatenate((self.data.x, x), axis=1)
            if self.store_fx:
                self.data.fx = np.concatenate((self.data.fx, fx), axis=1)
            if self.store_gx:
                self.data.gx = np.concatenate((self.data.gx, gx), axis=2)

        self.data.n_evals += n_evals

        if opt_x is not None:
            self.data.opt_x = opt_x
        if opt_fx is not None:
            self.data.opt_fx = opt_fx

        if nIter is not None:
            if self.data.evals_n_iters is None:
                self.data.evals_n_iters = np.array([nIter] * n_evals)
            else:
                self.data.evals_n_iters = np.hstack(np.array([nIter] * n_evals))

        if time is not None:
            if self.data.time is None:
                self.data.time = np.array(time)
            else:
                self.data.time = np.hstack((self.data.time, time))

    def get_parameters(self, iter=None):
        """
        Return all the parameters evaluated, or if iter is specified, only the parameters evaluated at the given iteration
        :param: iter: scalar 
        :return: np.matrix 
        """

        if iter is None:
            return self.data.x
        else:
            return self.data.x[:, (self.data.evals_n_iters == iter).flatten()]

    def get_parameters_till_iter(self, iter):
        """
        Return all the parameters evaluated, or if iter is specified, only the parameters evaluated at the given iteration
        :param: iter: scalar
        :return: np.matrix
        """
        return self.data.x[:, (self.data.evals_n_iters <= iter).flatten()]

    def get_objectives(self, iter=None):
        """

        :return: np.matrix
        """
        if iter is None:
            return self.data.fx
        else:
            pass  # TODO: implement me!!!

    def get_time(self):
        return self.data.time

    def get_final_time(self):
        return self.data.time[-1]

    def get_n_evals(self):
        return self.data.n_evals

    def get_best_parameters(self):
        return self.data.opt_x

    def get_best_objectives(self):
        return self.data.opt_fx

    def plot_parameters(self):
        plt.plot(self.get_parameters().T, linewidth=2)
        # TODO: Normalize parameters
        plt.ylabel('Normalized Parameters')
        plt.xlabel('Number of Evaluations')

    def plot_optimization_curve(self, scale='log'):
        # TODO: Check that it is SOO
        plt.plot(self.get_objectives().T)
        plt.xlabel('Evaluations')
        plt.ylabel('Obj.Func.')
        if scale == 'log':
            try:
                ax = plt.gca()
                ax.set_yscale('log')
            except:
                print('log scale is not possible')

    def save(self, filename, verbosity=1):
        rdata.save(self, fileName=filename, verbosity=verbosity)


