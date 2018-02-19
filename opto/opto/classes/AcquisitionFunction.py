# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np


class AcquisitionFunction(object):
    def __init__(self, model=[], parameters={}):
        """
        
        :param model: model of the response surface
        :param parameters: 
        """
        self._m = model

    def evaluate(self, parameters):
        raise NotImplementedError('Implement in subclass')

    def update(self, model, logs):
        """
        
        :param model: model of the response surface
        :param logs: logs of the optimization process (if required)
        :return: 
        """
        self._m = model
