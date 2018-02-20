# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
from opto.opto.classes.AcquisitionFunction import AcquisitionFunction


class UCB(AcquisitionFunction):
    def __init__(self, model, logs, parameters={}):
        super(UCB, self).__init__(model=model, parameters=parameters)
        self.alpha = parameters.get('alpha', 0.1)

    def evaluate(self, parameters):
        mean, var = self._m.predict(parameters.T)
        assert np.all(var >= 0)
        f = mean + self.alpha * var
        assert np.all(np.invert(np.isnan(f)))
        assert np.all(np.invert(np.isinf(f)))
        return f
