# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:48:31 2015

@author: Roberto Calandra
"""
from __future__ import division, print_function, absolute_import
from past.builtins import basestring
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
import h5py
# import R.log as log
import opto.utils as rutils
import copy
from numbers import Number

__author__ = 'Roberto Calandra'
__version__ = '0.0.3'
# $Source$

DEBUG = False


class dataset(object):

    def __init__(self, data_input, data_output=None, name='Dataset', labels_input=None, labels_output=None, dim_input=None):
        """

        :param data_input: Numpy matrix N_INPUT_DIMENSIONS x N_DATA
        :param data_output: Numpy matrix N_OUTPUT_DIMENSIONS x N_DATA
        :param name: string name of the dataset
        :param labels_input: list of strings with the names of the input variables
        :param labels_output: list of strings with the names of the output variables
        """
        if DEBUG:
            print(data_input.shape)
            print(data_output.shape)

        if isinstance(data_input, basestring):
            # Load dataset from file
            self.loadFromFile(data_input)
            pass
        else:
            # Create dataset
            if isinstance(data_input, np.ndarray):
                data_input = np.matrix(data_input)
            if isinstance(data_output, np.ndarray):
                data_output = np.matrix(data_output)
            # TODO: deal with vectors:

            if data_output is not None:
                self._hasLabels = True
            else:
                self._hasLabels = False
            if self._hasLabels:
                assert data_input.shape[1] == data_output.shape[1], 'Mismatch in the number of input and output data: %r - %r' \
                                                                % (data_input.shape[1], data_output.shape[1])
                self.output = data_output
                self.dim_output = data_output.shape[0]
                self.labels_output = labels_output

            self.name = np.string_(name)
            self.input = data_input
            self.n_data = data_input.shape[-1]
            self.dim_input = data_input.shape[0]
            self.labels_input = labels_input
            self.verbosity = 0

    def __str__(self):
        return 'Dataset: "' + self.name + '" ' + str(self.get_dim_input()) + '->' + str(self.get_dim_output()) + \
               ' with ' + str(self.get_n_data()) + ' datapoints'

    def get_n_data(self):
        """

        :return: scalar. number of datapoints
        """
        return self.n_data

    def get_dim_input(self):
        """
        """
        return self.dim_input

    def get_dim_output(self):
        """
        """
        return self.dim_output

    def get_label_input(self, index):
        if self.labels_input is None:
            # If the labels are not defined, generate generic ones...
             out = 'Input ' + str(index+1)
        else:
            out = self.labels_input[index]
        return out

    def get_labels_input(self, index=None):
        if index is None:
            index = range(self.dim_output)
        if self.labels_input is None:
            t = []
            for i in index:
                t.append('Input ' + str(i+1))
            return t
        else:
            return self.labels_input[index]

    def get_label_output(self, index):
        if self.labels_output is None:
            # If the labels are not defined, generate generic ones...
             out = 'Output ' + str(index+1)
        else:
            out = self.labels_output[index]
        return out

    def get_labels_output(self, index=None):
        if index is None:
            index = range(self.n_output)
        if isinstance(index, Number):
            index = [index]
        # TODO: is get_label_output
        if self.labels_output is None:
            # If the labels are not defined, generate generic ones...
            t = []
            for i in index:
                t.append('Output ' + str(i+1))
            return t
        else:
            return self.labels_output[index]

    def split_ndata(self, n_data, schema='random', labels=None):
        """
        Return a dataset subset of the original dataset, having only n_data datapoints.
        :param n_data:
        :param schema
        :param labels:
        :return: [new_dataset, complementary_dataset]
        """
        ratio = n_data/self.n_data
        return self.split(ratio, schema=schema, labels=labels)

    def split(self, ratio, schema='random', labels=None):
        """

        :param ratio: list or np.array of the ratios.
        :param schema: 'random', 'none'
        :param labels: strings with the name of the resulting datasets
        :return:
        """
        if np.sum(ratio) == 1:
            if ratio == 1:
                return [self]  # strange case. we are done!
            pass  # already in format [0,1]
        else:
            ratio = np.array([ratio, 1-np.sum(ratio)])
        assert np.sum(ratio) == 1, 'Sum of the split must sum up to 1'
        n_datasets = len(ratio)

        # {
        #     'random': self.shuffle(),
        #     'None': 2
        # }.get(schema, log.warning('Unknown schema: ') % schema)

        if schema == 'random':
            self.shuffle()  # TODO: Fix me! this line has the negative effect of reshuffling the original dataset!!!!
        else:
            if schema == 'none':
                pass
            else:
                log.warning('Not implemented yet')
        # TODO: implement interleaving

        log.cnd_msg(self.verbosity, 1, 'Split dataset')
        out = []
        idx = [0]
        for i in range(n_datasets):
            idx.append(int(np.ceil(self.get_n_data()*sum(ratio[:i+1]))))
            if labels is None:
                name = self.name
            else:
                name = labels[i]
            out.append(dataset(name=name,
                               data_input=self.input[:, idx[i]:idx[i+1]], data_output=self.output[:, idx[i]:idx[i+1]],
                               labels_input=self.labels_input, labels_output=self.labels_output))
        done = 0  # Fix me
        log.cnd_status(self.verbosity, 1, done)
        return out[0:]

    def subsample(self, ratio, schema='random'):
        """
        Subsample a dataset
        :param ratio:
        :param schema:
        :return:
        """
        out = self.split(ratio, schema)
        return out[0]

    def shuffle(self):
        """
        Shuffle a dataset
        :return: Success flag
        """
        log.cnd_msg(self.verbosity, 1, 'Shuffle dataset')
        try:
            idx = np.random.permutation(self.n_data)
            self.input = self.input[:, idx]
            self.output = self.output[:, idx]
            done = 0
        except:
            done = -1
            pass
        log.cnd_status(self.verbosity, 1, done)
        return done

    def get_bounds_input(self, idx=None):
        """

        :return: class R.utils.bounds
        """
        # TODO: add idx
        return rutils.bounds(min=np.array(self.input.min(axis=1)), max=np.array(self.input.max(axis=1)))

    def get_bounds_output(self, idx=None):
        """

        :return: class R.utils.bounds
        """
        # TODO: add idx
        return rutils.bounds(min=np.array(self.output.min(axis=1)), max=np.array(self.output.max(axis=1)))

    def remove_input_covariate(self, idx_cov):
        """
        Remove input covariates from the dataset
        :param idx_cov: numpy or list of the indexes of the covariates to be eliminated
        :return:
        """
        n_cov_to_remove = len(idx_cov)
        self.dim_input -= n_cov_to_remove
        self.input = np.delete(self.input, idx_cov, 0)
        if self.labels_input is not None:
            for index in sorted(idx_cov, reverse=True):
                del self.labels_input[index]
        return self

    def remove_output_covariate(self, idx_cov):
        """

        :param idx_cov:
        :return:
        """
        n_cov_to_remove = len(idx_cov)
        self.dim_output -= n_cov_to_remove
        self.output = np.delete(self.output, idx_cov, 0)
        if self.labels_output is not None:
            for index in sorted(idx_cov, reverse=True):
                del self.labels_output[index]
        return self

    def reshape_input(self, shape):
        # TODO: implement me, and change rest of the code to be compatible with reshaping

        return 0

    def plot_input_hist(self, dim=None, interactive=False):
        """
        """
        # TODO: implement interactive
        if dim is None:
            dim = np.array(range(self.get_dim_input()))
        for num in dim:
            plt.figure()
            plt.hist(self.input[num, :].T)
            # plt.title(self.get_label_input(num))
            plt.xlabel(self.get_label_input(num))
            plt.ylabel("")
            plt.show()
            #plt.interactive(False)
            #matplotlib.pylab.show(block=True)

    def plot_output_hist(self, dim=None, interactive=False):
        """
        """
        # TODO: implement interactive
        if dim is None:
            dim = np.array(range(self.get_dim_output()))
        for num in dim:
            # TODO: use subfigures
            plt.figure()
            plt.hist(self.output[num, :].T)
            plt.title(self.get_label_output(dim))
            plt.xlabel("Value")
            plt.ylabel("")
            plt.show()
            #plt.interactive(False)
            #matplotlib.pylab.show(block=True)

    def plot_input(self, dim=None, interactive=False):
        """
        """
        if dim is None:
            dim = range(self.get_dim_input())

        def plotFunc(idx):
            plt.plot(self.input[idx, :].T)
            plt.xlabel('Number of data')
            plt.ylabel(self.get_label_input(idx))

        if interactive:
            import scipyplot as rplot
            rplot.utils.interactivePlot(plotFunc, nplots=self.get_dim_input())
        else:
            for i in dim:
                plt.figure()
                plotFunc(i)
                plt.show()

    def plot_output(self, dim=None, interactive=False):
        """
        """
        if dim is None:
            dim = range(self.get_dim_output())

        def plotFunc(idx):
            plt.plot(self.output[idx, :].T)
            plt.xlabel('Number of data')
            plt.ylabel(self.get_label_output(idx))

        if interactive:
            import scipyplot as rplot
            rplot.utils.interactivePlot(plotFunc, nplots=self.get_dim_output())
        else:
            for i in dim:
                plt.figure()
                plotFunc(i)
                plt.show()

    def save2file(self, nameFile=None, compression=False, verbosity=1):
        if compression is True:
            pass
            # TODO: Implement me!
            #opt = [compression="gzip", compression_opts=9]
        with h5py.File(nameFile + '.h5', 'w') as hf:
            hf.create_dataset('input', data=self.input)
            hf.create_dataset('output', data=self.output)
            hf.attrs['name'] = self.name
            hf.attrs['n_data'] = self.n_data
            hf.attrs['dim_input'] = self.dim_input
            hf.attrs['dim_output'] = self.dim_output
            # hf.attrs['labels_input'] = self.labels_input
            # hf.attrs['labels_obj'] = self.labels_obj
            # TODO: save labels and such...
        done = 0
        return done

    def loadFromFile(self, nameFile, verbosity=1):
        # TODO: Implement me!
        with h5py.File(nameFile, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            self.input = np.array(hf.get('input'))
            self.output = np.array(hf.get('output'))
            self.name = hf.get('name')
            self.n_data = hf.get('n_data')
            self.dim_input = hf.get('dim_input')
            self.dim_output = hf.get('dim_output')
            print(self.n_data)
            # self.labels_input = hf.get('labels_input')
            # self.labels_obj = hf.get('labels_obj')
            # print('Shape of the array dataset_1: \n', np_data.shape)

    def hasLabel(self):
        return self._hasLabels

    def merge(self, other_dataset):
        self.check_compatibility(other_dataset)  # Do all the asserts to check that the two datasets are compatible

        out = copy.deepcopy(self)
        out.n_data += other_dataset.get_n_data()
        out.input = np.concatenate((out.input, other_dataset.input), axis=1)
        out.output = np.concatenate((out.output, other_dataset.output), axis=1)
        # TODO: merge labels (smartly checking for None)
        # self.labels_input =
        # self.labels_obj =

        return out

    def check_compatibility(self, other_dataset):
        """
        This function make sure that two dataset are compatible
        :param other_dataset: a dataset
        :return:
        """
        assert isinstance(other_dataset, dataset)
        assert self.get_dim_input() == other_dataset.get_dim_input(), \
            'Mismatch in the number of inputs: %r - %r' % (self.get_dim_input(), other_dataset.get_dim_input())
        assert self.get_dim_output() == other_dataset.get_dim_output(), \
            'Mismatch in the number of outputs: %r - %r' % (self.get_dim_output(), other_dataset.get_dim_output())
        done = 0
        return done

    def next_batch(self, size_batch=None, transpose=False, input_only=False):
        """
        Return a minibatch out of the full dataset
        :param size_batch:
        :param transpose:
        :param input_only:
        :return:
        """
        if size_batch is None:
            # Return full dataset
            x = self.input.transpose()
            if input_only is False:
                y = self.output.transpose()
        else:
            assert size_batch > 0
            if size_batch >= self.n_data:
                log.cnd_warning(self.verbosity, 1, 'Size minibatch larger then number of data')
                size_batch = self.n_data
            idx = np.random.permutation(self.n_data)
            if transpose is True:
                x = self.input[:, idx[0:size_batch]].transpose()
                if input_only is False:
                    y = self.output[:, idx[0:size_batch]].transpose()
            else:
                x = self.input[:, idx[0:size_batch]]
                if input_only is False:
                    y = self.output[:, idx[0:size_batch]]
        if input_only is True:
            out = x
        else:
            out = [x, y]
        return out

    def get_input(self, idx=None):
        """
        get_input
        :param idx:
        :return: Numpy matrix N_INPUT_DIMENSIONS x N_DATA
        """
        if idx is None:
            return self.input
        else:
            return self.input[:, idx]

    def get_output(self, idx_data=None, idx_covariate=None):
        """
        get_output
        :param idx_data:
        :param idx_covariate:
        :return: Numpy matrix N_OUTPUT_DIMENSIONS x N_DATA
        """
        if idx_data is None:
            if idx_covariate is None:
                return self.output
            else:
                return self.output[idx_covariate, :]
        else:
            if idx_covariate is None:
                return self.output[:, idx_data]
            else:
                return self.output[idx_covariate, idx_data]

    def kfold(self, K=1):
        if isinstance(K, basestring):
            if K in ('loo', 'LOO'):
                K = self.get_n_data()
        assert K <= self.get_n_data()
        idx_rand = np.random.permutation(self.get_n_data())
        idx = []
        index = 0
        for i in range(K):
            new_index = int(np.ceil(self.get_n_data()*(i+1)/K))
            # print(index, new_index)
            idx.append(idx_rand[index:new_index])
            index = new_index

        train_sets = []
        test_sets = []
        idx_train = []
        idx_test = []
        for i in range(K):
            # Compute the indexes for each K-fold
            idx_train.append(np.concatenate([x for j, x in enumerate(idx) if j != i]))
            idx_test.append(idx[i])
            # Create the dataset for each K-fold
            train_sets.append(dataset(data_input=self.input[:, idx_train[i]], data_output=self.output[:, idx_train[i]],
                                      labels_input=self.labels_input, labels_output=self.labels_output))
            test_sets.append(dataset(data_input=self.input[:, idx_test[i]], data_output=self.output[:, idx_test[i]],
                                     labels_input=self.labels_input, labels_output=self.labels_output))
        return train_sets, test_sets

    def normalize_input(self, type='mean0var1'):
        import R.data.normalize as normalize
        self.input, t = normalize(self.input, type=type)

    def normalize_output(self, type='mean0var1'):
        import R.data.normalize as normalize
        self.output, t = normalize(self.output, type=type)

    def append(self, input, output=None):
        # TODO: assert
        n_data = input.shape[1]
        self.n_data += n_data
        self.input = np.concatenate((self.input, input), axis=1)
        if output is None:
            if self._hasLabels:
                raise
        else:
            self.output = np.concatenate((self.output, output), axis=1)
        # NEVER TESTED!!!
