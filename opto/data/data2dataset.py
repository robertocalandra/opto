# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from .dataset import dataset as rdataset


def data2dataset(dataset):
    """
    This function make sure that multiple common types of inputs are all translated into a dataset
    :param dataset: np vectors (1xn, nx1, n), lists, np matrices, and dataset
    :return:
    """
    # Is it already a valid dataset?
    if not isinstance(dataset, rdataset):
        # if not, is it a matrix or a vector or what?
        if not isinstance(dataset, np.matrix):
            # This look like a vector
            if isinstance(dataset, np.ndarray):
                if dataset.ndim is 1:
                    dataset = np.matrix(dataset)
                else:
                    dataset = np.matrix(dataset.T)
            else:
                dataset = np.matrix(dataset)

        dataset = rdataset(dataset)  # Store into a dataset
    return dataset
