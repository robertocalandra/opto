# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

__version__ = '0.1'
__author__ = 'Roberto Calandra'


class OptTask(object):

    def __init__(self, f, n_parameters, n_objectives=1, order=0, bounds=None, name='OptTask', task={'minimize'},
                 labels_param=None, labels_obj=None, vectorized=False, referencePoint=None, opt_parameters=None,
                 opt_obj=None, info=None, optimal_hypervolume=None, fprime=None, fhess=None, constraints=None,
                 codomain=None, n_contexts=0, fcontext=None):
        """
        Definition of the task to be optimized
        :param f: function that accept np.matrix(1, n_parameters) and return a np.matrix(n_objectives, 1)
        :param fprime: function that accept np.matrix(1, n_parameters) and return a np.matrix(1, n_ n_objectives)
        :param n_parameters: integer
        :param bounds: object of type utils.bounds
        :param name: Name of the objective function
        :param task: NOTE: currently unused.
        :param labels_param: List of the labels for the parameter(s)
        :param labels_obj: List of the labels for the objective(s)
        :param vectorized: if the function can also accept a np.matrix(n_data, n_parameters) and return a
                np.matrix(n_objectives, n_data), turn this flag to True.
        """
        self.f = f  # Objective function
        self.fprime = fprime  # Derivative of the objective function wrt the parameters
        self.fhess = fhess  # Hessian of the objective funciton wrt the parameters
        self.fcontext = fcontext  # Function that return the context of an evaluation
        self.order = order  # Order of the objective function [0-2]
        self.name = name  # Name of the objective function !CURRENTLY UNUSED!
        self.vectorized = vectorized  # Is the objective function vectorized?
        self.task = task  # !CURRENTLY UNUSED!
        self.n_parameters = n_parameters  # Number of dimensions
        self.n_objectives = n_objectives  # Number of objectives
        self.n_contexts = n_contexts  # Number of contextual variables

        self.bounds = bounds  # (Optional) hyperrectangle bounds
        self.constraints = constraints  # Experimentl!!!
        self.labels_param = labels_param  # (Optional)
        self.labels_obj = labels_obj  # (Optional)

        self.referencePoint = referencePoint  # Reference point for the Hypervolume -- Multi-objective Optimization only
        self.optimal_hypervolume = optimal_hypervolume  # -- Multi-objective Optimization only

        self.opt_parameters = opt_parameters  # Optimal parameters
        self.opt_obj = opt_obj  # = OBJ.evaluate_func(OBJ.opt_parameters{1},0); %0;  % Objective function at the minimum

        self.codomain = codomain
        self.info = info  # Additional info (e.g., continuous, multimodal, etc)

        self.accept_nan = False  # Make sure that evaluations cannot be nan
        self.accept_inf = False  # Make sure that evaluations cannot be inf

        assert self.n_parameters > 0
        assert self.n_objectives > 0
        assert 0 <= self.order <= 2, 'Invalid order of the objective function'
        if self.order > 0:
            assert self.fprime is not None, 'Error'
        if self.order > 1:
            assert self.fhess is not None, 'Error'

    def get_context(self):
        c = self.fcontext()
        assert c
        return self.fcontext()

    def evaluate(self, parameters, order=0):
        """
        Evaluate the objective function at the given parameters
        :param parameters: np.matrix (N_PARAMETERS x N_DATA)
        :param order: int. desired order of the evaluation 0=fx, 1=fx,gx, 2=fx,gx,hx
        :return: np.matrix (N_OBJ_FUNC x N_DATA)
        """
        parameters = np.matrix(parameters)
        assert parameters.ndim == 2, 'Parameters must return a np.matrix (N_DATA x N_PARAMETERS)'
        assert parameters.shape[1] == self.n_parameters, 'Parameters must return a np.matrix (N_DATA x N_PARAMETERS)'
        assert np.all(np.invert(np.isnan(parameters))), 'Parameters cannot be nan'
        assert np.all(np.invert(np.isinf(parameters))), 'Parameters cannot be inf'
        n_points = parameters.shape[0]

        assert 0 <= order <= self.order, 'Invalid order'

        if self.vectorized is True:
            # Function accept vectorized input, give them all at once
            if order >= 0:
                fx = self.f(parameters)
            if order >= 1:
                gx = self.fprime(parameters)
            if order >= 2:
                hx = self.fhess(parameters)
            if self.n_contexts > 0:
                c = self.fcontext()
        else:
            # Function does not accept vectorized input, feed them one by one
            if order >= 0:
                fx = np.empty((self.n_objectives, n_points))
                for i in range(n_points):
                    fx[:, i] = self.f(np.matrix(parameters[i])).flatten()
            if order >= 1:
                gx = np.empty((self.n_objectives, self.n_parameters, n_points))
                for i in range(n_points):
                    gx[:, :, i] = self.fprime(np.matrix(parameters[i])).flatten()
            if order >= 2:
                assert False, 'not implemented yet'
                # gx = np.empty((self.n_objectives, self.n_parameters, n_points))
                # for i in range(n_points):
                #     gx[:, :, i] = self.fprime(np.matrix(parameters[i])).flatten()

        if not self.accept_nan:
            assert np.all(np.invert(np.isnan(fx))), 'fx cannot be nan'
        if not self.accept_inf:
            assert np.all(np.invert(np.isinf(fx))), 'fx cannot be inf'
        assert fx.ndim == 2, 'Objective function must return a np.matrix (N_OBJ_FUNC x N_DATA)'
        assert fx.shape == (self.n_objectives, n_points), 'Objective function must return a np.matrix (N_OBJ_FUNC x N_DATA)'
        if order >= 1:
            assert gx.ndim == 3, 'Objective function must return a np.matrix (N_OBJ_FUNC x N_PARAMETERS x N_DATA)'
            assert gx.shape == (self.n_objectives, self.n_parameters, n_points), 'Objective function must return a np.matrix (N_OBJ_FUNC x N_PARAMETERS x N_DATA)'
        if order >= 2:
            assert hx.ndim == 4, 'Objective function must return a np.matrix (N_OBJ_FUNC x N_PARAMETERS x N_DATA)'

        if (self.n_objectives == 1) and (self.opt_obj is not None):
            assert np.all(self.opt_obj <= fx), 'Invalid value of the obj.func.: value below the optimal obj.func. %f < %f' % (fx, self.opt_obj)

        if order == 0:
            if self.n_contexts > 0:
                return fx, c
            else:
                return fx
        if order == 1:
            if self.n_contexts > 0:
                return fx, gx, c
            else:
                return fx, gx
        if order == 2:
            if self.n_contexts > 0:
                return fx, gx, hx, c
            else:
                return fx, gx, hx

    def test_gradient(self, points=None):
        """

        :param points:
        :return:
        """
        if points is None:
            assert self.bounds is not None, 'Either bounds or points must be defined'
            n_points = 10000
            points = self.bounds.sample_uniform(n_points)
        # TODO: implement me!
        
    def test_hessian(self, points=None):
        pass

    def test_vectorized(self, points=None):
        pass
        
    def visualize(self, bounds=None, cmap=cm.coolwarm, interactive=False):
        """
        Visualize the objective function
        :param bounds:
        :param cmap:
        :param interactive:
        :return:
        """
        if self.n_parameters == 1:
            self.plot_1D(bounds=bounds, interactive=interactive)
        if self.n_parameters == 2:
            self.plot_2D(bounds=bounds, cmap=cmap, interactive=interactive)
        if self.n_parameters > 2:
            print('Not implemented yet!')
        
    def plot_1D(self, bounds=None, interactive=True, resolution=1000):
        """
        Plot a 1D function
        :param bounds:
        :return:
        """
        assert (bounds is None) and (self.bounds is None), 'Either self.bounds or bounds need to be defined'
        assert self.n_parameters == 1
        if bounds is None:
            bounds = self.bounds
        for i in range(self.n_objectives):
            x = np.linspace(bounds.get_min(), bounds.get_max(), num=resolution)
            y = self.evaluate(x)
            plt.figure()
            plt.plot(x, y)
        plt.show()

    def plot_slice_1D(self, point, dim, constraints):
        pass

    def plot_2D(self, bounds=None, interactive=True, resolution=100, plotType='2D', cmap=cm.coolwarm):
        """

        :param bounds:
        :param interactive:
        :param resolution:
        :param plot: '2D' or '3D'
        :return:
        """
        assert (bounds is None) or (self.bounds is None), 'Either self.bounds or bounds need to be defined'
        assert self.n_parameters == 2
        if bounds is None:
            bounds = self.bounds
        for i in range(self.n_objectives):
            # x = np.linspace(bounds.get_min(i), bounds.get_max(i), num=resolution)
            # y = self.evaluate(x)

            xlist = np.linspace(bounds.get_min(0), bounds.get_max(0), num=resolution)  # Create 1-D arrays for x,y dimensions
            ylist = np.linspace(bounds.get_min(1), bounds.get_max(1), num=resolution)
            X, Y = np.meshgrid(xlist, ylist)  # Create 2-D grid xlist,ylist values
            Z = self.evaluate(np.vstack((X.flatten(), Y.flatten())).T)  # Compute function values on the grid
            Z = Z.reshape(X.shape)
            # TODO: make sure that it works for MOO

            for i in range(self.n_objectives):
                # fig = plt.figure()
                # niceFigure()
                fig = plt.gcf()
                if plotType is '3D':
                    ax = Axes3D(fig)
                    surf = ax.plot_surface(X, Y, Z,
                                           cmap=cmap,
                                           linewidth=0,
                                           antialiased=True,
                                           rstride=1,
                                           cstride=1,
                                           # alpha=0.5,
                                           # vmin=0,
                                           # vmax=350,
                                           # shade=True,
                                           # rasterized=True
                                           )
                    ax.set_xlim3d(bounds.get_min(0), bounds.get_max(0))
                    ax.set_ylim3d(bounds.get_min(1), bounds.get_max(1))
                else:
                    ax = plt.gca()
                    plt.imshow(Z, origin='lower',
                               extent=[bounds.get_min(0), bounds.get_max(0), bounds.get_min(1), bounds.get_max(1)],
                               cmap=cmap)
                    # plt.contourf(X, Y, Z, cmap=cmap, antialiased=True)
                    plt.xlabel('Parameter 1')
                    plt.xlabel('Parameter 2')
                    ax.set_xlim(bounds.get_min(0), bounds.get_max(0))
                    ax.set_ylim(bounds.get_min(1), bounds.get_max(1))
                    plt.colorbar()
            if interactive:
                plt.show()
        return fig

    def plot_slice_2D(self, point, dims, constraints):
        pass

    def get_order(self):
        return self.order
        
    def get_n_objectives(self):
        return self.n_objectives

    def get_n_parameters(self):
        return self.n_parameters

    def isMOO(self):
        return self.get_n_objectives() != 1

    def isSOO(self):
        return self.get_n_objectives() == 1

    def isBounded(self):
        return self.bounds is not None

    def isConstrained(self):
        return self.constraints is not None

    def get_constraints(self):
        return self.constraints

    def add_constraints(self, constraints):
        self.constraints = constraints

    def get_name(self):
        return self.name

    def get_bounds(self):
        return self.bounds

    def get_hypervolume_reference_point(self):
        return self.referencePoint

    def get_labels_parameters(self, idx=None):
        """
        Return a list with the labels of the parameters specified by idx (on all the parameters if idx=None)
        :param idx:
        :return:
        """
        if self.labels_param is None:
            if idx is None:
                idx = np.arange(self.n_parameters)
            labels = []
            for i in idx:
                labels.append('Parameter %d' % i)
        else:
            if idx is None:
                labels = self.labels_param
            else:
                labels = self.labels_param[idx]
        return labels

    def get_labels_objectives(self, idx=None):
        """
        Return a list with the labels of the objectives specified by idx (on all the parameters if idx=None)
        :param idx:
        :return:
        """
        if self.labels_param is None:
            if idx is None:
                idx = np.arange(self.n_objectives)
            labels = []
            for i in idx:
                labels.append('Obj.Func. %d' % i)
        else:
            if idx is None:
                labels = self.labels_obj
            else:
                labels = self.labels_obj[idx]
        return labels
