# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np


class linearTransformation():
    """
    Linear transformation 
    y = A x + B
    """
    def __init__(self, A, B):
        """
        
        :param A: np.array [M, N]
        :param B: np.array [M]
        """
        self.A = np.array(A)
        self.B = np.array(B)
        # TODO: assert size A and B
        assert A.ndim == 2
        assert A.shape[0] == B.shape[0], 'Inconsistent size'

    def transform(self, X):
        """
        
        :param X: np.array [N, D]
        :return: 
        """
        assert X.shape[0] == self.A.shape[1], 'Inconsistent size X %d - %d' % (X.shape[0], self.A.shape[1])
        return np.dot(self.A, X).T + self.B

    def invert(self, Y):
        """
        x = (y - B)/A
        :param Y: np.array [M, D]
        :return: X np.array []
        """
        assert Y.ndim == 2
        assert Y.shape[0] == self.A.shape[0], 'Inconsistent size Y %d - %d' % (Y.shape[0], self.A.shape[1])
        return np.linalg.solve(self.A, Y - np.matrix(self.B).T)
