# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
from opto.opto.classes.AcquisitionFunction import AcquisitionFunction


class mean(AcquisitionFunction):
    def __init__(self, model, parameters={}):
        super(mean, self).__init__(model=model, parameters=parameters)

    def evaluate(self, parameters):
        mean, var = self._m.predict(parameters.T)
        return mean
