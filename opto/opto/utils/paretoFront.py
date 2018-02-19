# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import R.log as log
from timeit import default_timer as timer

import logging
logger = logging.getLogger(__name__)


def is_dominated(x, y, minimize=True):
    """
    Compute if x is dominated by y
    :param x: np.matrix [N_OBJECTIVES] 
    :param y: np.matrix [N_POINTS, N_OBJECTIVES] 
    :param minimize: bool True=> compute PF that minimize, False=> compute PF that maximize
    :return: 
    """
    if minimize:
        return np.all(y <= x, axis=1)
    else:
        return np.all(y >= x, axis=1)


def dominates(x, y, minimize=True):
    """
    Compute if x dominates y
    :param x: np.matrix [N_OBJECTIVES] 
    :param y: np.matrix [N_POINTS, N_OBJECTIVES] 
    :param minimize: bool. True=> compute PF that minimize, False=> compute PF that maximize
    :return: 
    """
    if minimize:
        return np.all(x <= y, axis=1)
    else:
        return np.all(x >= y, axis=1)


def is_pareto_optimal_1(objectives, minimize=True):
    """
    :param costs: An [N_OBJECTIVES, N_POINTS] matrix
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto optimal
    """
    objectives = objectives.T
    is_PF = np.ones(objectives.shape[0], dtype=bool)
    for i, c in enumerate(objectives):
        is_PF[i] = (np.sum(is_dominated(x=c, y=objectives[is_PF], minimize=minimize)) <= 1)
        # (note that each point is dominated by each self)
    return is_PF


def is_pareto_optimal_1b(objectives, minimize=True):
    """
    :param objectives: An [N_OBJECTIVES, N_POINTS] matrix
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto optimal
    """

    objectives = objectives.T  # [N_POINTS, N_OBJECTIVES]
    is_PF = np.ones(objectives.shape[0], dtype=bool)
    for i, c in enumerate(objectives):
        if is_PF[i]:
            is_PF[is_PF] = np.array(np.invert(dominates(x=c, y=objectives[is_PF], minimize=minimize))).squeeze()
            is_PF[i] = True
            # Remove dominated points (note that each point is dominated by each self)
    return is_PF


def is_pareto_optimal_2(costs, minimize=True):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = costs.T
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Remove dominated points
    return is_efficient


def is_pareto_optimal_3(costs, minimize=True):
    # From http://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = costs.T
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs >= c, axis=1))
    return is_efficient

# def is_pareto_optimal_4(objectives, minimize=True):
#     nObj = objectives.shape[0]
#
#     if nObj == 2:
#         # TODO: stablesort
#         if func[0] == 'minimize':
#             idx = np.argsort(objectives[0])
#         if func[0] == 'maximize':
#             idx = np.argsort(objectives[0])[::-1]
#         idx_PF = [idx[0]]  # TODO: bug! first element is always in the PF!
#         cur = objectives[1, idx[0]]
#         if func[1] == 'minimize':
#             for i in idx:
#                 if objectives[1, i] < cur:
#                     idx_PF.append(i)
#                     cur = objectives[1, i]
#         if func[1] == 'maximize':
#             for i in idx:
#                 if objectives[1, i] > cur:
#                     idx_PF.append(i)
#                     cur = objectives[1, i]
#         PF = objectives[:, idx_PF]
#
#     if nObj > 2:
#         # Use simple_cull
#         # TODO: accept func with multiple value
#         if func[0] == 'maximize':
#             f = dominates_max
#         if func[0] == 'minimize':
#             f = dominates_min
#         dominated = []
#         cleared = []
#         remaining = np.transpose(objectives)
#         nPointsRemaning = remaining.shape[0]
#         while nPointsRemaning > 0:
#             # print(nPointsRemaning)
#             candidate = remaining[0]
#             new_remaining = []
#             for other in remaining[1:]:
#                 [new_remaining, dominated][f(candidate, other)].append(other)
#             if not any(f(other, candidate) for other in new_remaining):
#                 cleared.append(candidate)
#             else:
#                 dominated.append(candidate)
#             remaining = np.array(new_remaining)
#             nPointsRemaning = remaining.shape[0]
#         PF = np.transpose(np.array(cleared))
#         dom = np.transpose(np.array(dominated))


def paretoFront(objectives, parameters=None, func='minimize'):
    """ Compute the Pareto Front
    
    :param objectives: [N_OBJECTIVES, N_POINTS]
    :param parameters: [N_PARAMETERS, N_POINTS]
    :param func: if func is a single string, it will be considered as the same value for all objectives. 
    Alternatively, it is possible to present a list (currently not implemented).
    :return: 
        PF [N_OBJECTIVES, N_POINTS]
        PF_par [N_PARAMETERS, N_POINTS]
    """
    nObj = objectives.shape[0]
    if parameters is not None:
        nPars = parameters.shape[0]
        assert objectives.shape[1] == parameters.shape[1], 'Inconsistent size objectives - parameters'

    assert isinstance(func, basestring), 'currently only single values are accepted'  # TODO: allow multiple values
    if isinstance(func, basestring):
        func = [func] * nObj

    _startTime = timer()
    logging.info('Computing PF')

    idx_PF = is_pareto_optimal_1b(objectives, minimize=True)
    PF = objectives[:, idx_PF]
    if parameters is not None:
        PF_PAR = parameters[:, idx_PF]

    end = timer()
    logging.info('PF computed in %f[s]' % (end - _startTime))
    logging.info('Identified %d points in the PF' % (PF.shape[1]))

    assert PF.shape[0] == nObj, 'Inconsistent size PF'
    if parameters is None:
        return PF
    else:
        PF_PAR = parameters[:, idx_PF]
        assert PF_PAR.shape[0] == nPars, 'Inconsistent size PF'
        return PF, PF_PAR