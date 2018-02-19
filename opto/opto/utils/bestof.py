# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt


def bestof(sequence, minimize=True):
    """
    Given a sequence of objectives, it will return the best seen so far
    :param sequence: [LENGHT, N_COVARIATES]
    :param minimize: bool. Are we minimizing or maximizing?
    :return: 
    """
    out = np.copy(sequence)
    for i in range(1, sequence.shape[0]):
        if minimize:
            out[i, :] = np.min(out[i-1:i+1, :], 0)
        else:
            out[i, :] = np.max(out[i-1:i+1, :], 0)

    return out

    # SLOWER
    # out = np.zeros(sequence.shape)
    # for i in range(sequence.shape[0]):
    #     if minimize:
    #         out[i, :] = np.min(sequence[0:i+1, :])
    #     else:
    #         out[i, :] = np.max(sequence[0:i + 1, :])

    # MUCH SLOWER
    # for i in range(sequence.shape[0]):
    #     if i == 0:
    #         current_best = sequence[i, :]
    #     if minimize:
    #         current_best = np.min(np.vstack((current_best, sequence[i, :])), 0)
    #     else:
    #         current_best = np.max(np.vstack((current_best, sequence[i, :])), 0)
    #     out[i, :] = current_best

