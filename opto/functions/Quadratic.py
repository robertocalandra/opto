# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from opto.opto.classes.OptTask import OptTask
import opto.utils as rutils


class Quadratic(OptTask):

    def __init__(self, n_parameters=2):
        """
        Quadratic function
        """
        super(Quadratic, self).__init__(f=self._f,
                                        fprime=self._g,
                                        name='Quadratic',
                                        n_parameters=n_parameters,
                                        n_objectives=1,
                                        order=1,
                                        bounds=rutils.bounds(max=[1]*n_parameters, min=[-1]*n_parameters),
                                        task={'minimize'},
                                        labels_param=None,
                                        labels_obj=None,
                                        vectorized=False,
                                        info=None,
                                        opt_obj=0,
                                        opt_parameters=np.matrix([[0]*n_parameters]),
                                        )

    def _f(self, x):
        return np.matrix(np.sum(np.power(x, 2), 1))

    def _g(self, x):
        return np.matrix(2*x)

    # def _h(self, x):
    #     return np.matrix(2*x)
