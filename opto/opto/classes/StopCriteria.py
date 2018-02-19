# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from timeit import default_timer as timer


class StopCriteria(object):

    def __init__(self, maxTime=None, maxIter=1000, maxEvals=1000):

        self.maxTime = maxTime
        self.maxIter = maxIter
        self.maxEval = maxEvals
        self._timeStart = None

    def startTime(self):
        self._timeStart = timer()

    def eval(self, time, iter):
        """

        :param time:
        :param iter:
        :return: scalar. 0 if none of the criteria is satisfied. otherwise, the identifier of the stop criteria
            1 = time > maxTime
            2 = iter > maxIter
        """
        time = self.get_time()
        if self.maxTime is not None:
            if time > self.maxTime:
                return 1
                print('Out of Time')

        if self.maxIter is not None:
            if iter > self.maxIter:
                return 2
                print('Out of Iterations')

    def get_n_maxEvals(self):
        return self.maxEval

    def get_n_maxIters(self):
        return self.maxIter

    def get_time(self):
        return timer() - self._timeStart

    def get_max_evals_iters(self):
        return min(self.maxIter, self.maxEval)

    def get_exit_condition(self):
        # TODO:
        return  []
