# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from opto.opto.classes.OptTask import OptTask
import opto.utils as rutils


class MOP2(OptTask):
    def __init__(self):
        """
        MOP2 function
        """
        super(MOP2, self).__init__(f=self._f,
                                   name='MOP2',
                                   n_parameters=2,
                                   n_objectives=2,
                                   order=0,
                                   bounds=rutils.bounds(min=[-2, -2], max=[2, 2]),
                                   task={'minimize', 'minimize'},
                                   labels_param=None,
                                   labels_obj=None,
                                   vectorized=True,
                                   info=None,
                                   referencePoint=np.array([1, 1]))

    def _f(self, x):
        assert x.ndim == 2
        out = np.empty([self.n_objectives, x.shape[0]]) * np.nan
        out[0, :] = 1 - np.exp(-np.sum(np.power(x - (1 / np.sqrt(2)), 2), 1)).flatten()
        out[1, :] = 1 - np.exp(-np.sum(np.power(x + (1 / np.sqrt(2)), 2), 1)).flatten()
        return out
