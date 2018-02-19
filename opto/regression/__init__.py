from __future__ import absolute_import

# from R.regression.classes.comparer import comparer
# from R.regression.classes.tf_model import tf_model
# from R.regression.classes.model import model
# from R.regression.classes.tf_loader import tf_loader
# # from R.regression.utils.comparerGenerator import experimentGenerator, datasetGenerator
# # from R.regression.utils.longTermPred import longTermPred
# # from R.regression.utils.robo_hyperopt import robo_hyperopt
# from .BNN import BNN
from .GP import GP
# from .NN import NN
# # from .gpModel import GPModel
# from .linearModel_tf import linearModel_tf
# from .linearModel import linearModel
# from .stats import stats
# from .wide_n_deep import wide_n_deep

__all__ = ['BNN', 'VHGP', 'linearModel_tf', 'wide_n_deep']

