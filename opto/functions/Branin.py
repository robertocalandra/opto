# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from opto.opto.classes.OptTask import OptTask
import opto.utils as rutils


class Branin(OptTask):

    def __init__(self):
        """
        Branin function
        """
        super(Branin, self).__init__(f=self._f,
                                     name='Branin',
                                     n_parameters=2,
                                     n_objectives=1,
                                     order=0,
                                     bounds=rutils.bounds(max=[10, 15], min=[-5, 0]),
                                     task={'minimize'},
                                     labels_param=None,
                                     labels_obj=None,
                                     vectorized=True,
                                     info=None,
                                     opt_obj=0.397887,
                                     opt_parameters=np.matrix([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]),
                                     codomain=rutils.bounds(min=0, max=250)
                                     )

    def _f(self, x):
        assert x.ndim == 2
        a = 1.
        b = (5.1 / (4. * np.pi ** 2))
        c = (5. / np.pi)
        t = (1. / (8. * np.pi))
        s = 10.
        return np.matrix(a * np.power(x[:, 1] - b * np.power(x[:, 0], 2) + c * x[:, 0] - 6., 2) + s * (1 - t) * np.cos(x[:, 0]) + s).T

    def _g(self, x):
        return 0  # TODO: implement me!

    # def eval_func(self, x):
    #     one = np.ones((x.shape[0], 1))
    #     x1 = x[:, 0][:, None]
    #     x2 = x[:, 1][:, None]
    #
    #     a = 1
    #     b = 5.1 / (4 * math.pi ** 2)
    #     c = 5 / math.pi
    #     r = 6
    #     s = 10
    #     t = 1 / (8 * math.pi)
    #     f = (x2 - b * x1 * x1 + c * x1 - r)
    #     y = one * a * f * f + s * (1 - t) * np.cos(x1) + s
    #
    #     return np.array(y)
