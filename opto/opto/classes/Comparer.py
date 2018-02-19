# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
import scipyplot as spp
from R.opto.utils.bestof import bestof
import R.data as rdata

import logging
logger = logging.getLogger(__name__)


class Comparer(object):
    """
    This is an automatic comparer for optimizers.
    """
    def __init__(self, experiments, tasks, parameters=DotMap()):
        """
        
        :param experiments: list of objects Optimizer
        """
        self.experiments = experiments
        self.tasks = tasks

        self.parallelize = parameters.get('parallelize', 0)  # if >0, it is the number of processes
        self.visualize = parameters.get('visualize', True)
        self.repetitions = []
        for i in experiments:
            self.repetitions.append(i.repetitions)
        self.repetitions = np.array(self.repetitions)
        # self.repetitions = np.array(repetitions)
        # self.n_experiments = np.sum(self.repetitions)
        # self.labels = labels

        self._logs = DotMap()

    def init_folder(self):
        # TODO: create folder based on running time
        pass

    def run_experiment(self, id, experiment, task):
        opt = experiment.f(parameters=experiment.p, task=task, stopCriteria=experiment.stopCriteria)
        opt.optimize()
        log = opt.get_logs()
        rdata.save(log, fileName='%08d' % id)
        return log

    def compare(self):
        """
        Start the experiments
        :return: 
        """
        if self.parallelize > 0:
            from multiprocessing import Pool, TimeoutError
            logging.info('Starting %d processes' % (self.parallelize))
            with Pool(processes=self.parallelize) as pool:
                pass
                # TODO: Implement me!!!
                # TODO: take care of saving logs to file
        else:
            # Fully serial, no parallelization
            tot_n_experiments = np.sum(self.repetitions)
            for idx, experiment in enumerate(self.experiments):
                for rep in np.arange(experiment.repetitions):
                    try:
                        id = self.get_id(idx, rep)
                        logging.info('Running experiment %d (of %d)' % (id+1, tot_n_experiments))
                        log = self.run_experiment(id=id, experiment=experiment, task=self.tasks)
                        # opt = experiment.f(parameters=experiment.p, task=self.tasks, stopCriteria=experiment.stopCriteria)
                        # opt.optimize()
                        # log = opt.get_logs()
                        # rdata.save(log, fileName='%08d' % id)
                        # TODO: save to file
                    except:
                        logging.warning('Experiment failed')
                    # self.load_log(idx, rep)
                    self._logs.logs[idx, rep] = log

    def _visualize(self):
        pass
        # TODO: implement me!

    def get_id(self, experiment, rep):
        """
        experiment first, and then repetition
        :param experiment: 
        :param rep: 
        :return: 
        """
        id = np.sum(self.repetitions[0:experiment]) + rep
        return id

    def save_log(self, log, idx, rep):
        """
        Save a log to file
        :param log: 
        :param idx: 
        :param rep: 
        :return: 
        """
        # TODO: generate nameFile
        # TODO: implement me!
        pass

    def load_log(self, idx, rep):
        """
        Load a log from file 
        :param idx: 
        :param rep: 
        :return: 
        """
        log = []  # TODO: read from file
        return log

    def get_logs(self):
        """
        Return the logs of all the experiments
        :return: 
        """
        return self._logs.logs

    def get_labels(self):
        labels = []
        for i in self.experiments:
            labels.append(i.name)
        return labels

    def plot_optCurve_nEvals(self, plot_best=True, scale='log', type='mean+65'):
        """
        assume that the experiment have the same lenght
        :return: 
        """
        data = []
        for idx, experiment in enumerate(self.experiments):
            max_lenght = 0
            for rep in np.arange(experiment.repetitions):
                max_lenght = max(max_lenght, self._logs.logs[idx, rep].get_n_evals())
            temp = np.nan * np.empty((experiment.repetitions, max_lenght))  # initialize matrix
            for rep in np.arange(experiment.repetitions):
                if plot_best:
                    temp[rep, 0:self._logs.logs[idx, rep].get_n_evals()] = bestof(np.matrix(self._logs.logs[idx, rep].get_objectives()).T).T
                else:
                    temp[rep, 0:self._logs.logs[idx, rep].get_n_evals()] = self._logs.logs[idx, rep].get_objectives()
            data.append(temp)

        #
        plt.figure()
        spp.rplot_data(data, legend=self.get_labels(), typeplot=type)
        plt.xlabel('Evaluations')
        plt.ylabel('Obj.Func.')
        if scale == 'log':
            try:
                ax = plt.gca()
                ax.set_yscale('log')
            except:
                print('log scale is not possible')
