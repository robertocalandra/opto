# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from R import utils as rutils
import unittest
import numpy as np
import R.opto as opto


class bounds(unittest.TestCase):

    def test_cmaes(self):

        from R.opto.functions.Branin import Branin
        from dotmap import DotMap

        task = Branin()
        stopCriteria = opto.classes.StopCriteria(maxEvals=10000)

        p = DotMap()
        p.verbosity = 1
        opt = opto.CMAES(parameters=p, task=task, stopCriteria=stopCriteria)
        opt.optimize()
        print(opt.get_logs())

    def test_randomSearch(self):

        from R.opto.functions.Branin import Branin
        from dotmap import DotMap

        task = Branin()
        stopCriteria = opto.classes.StopCriteria(maxEvals=10000)

        p = DotMap()
        p.verbosity = 1
        opt = opto.RandomSearch(parameters=p, task=task, stopCriteria=stopCriteria)
        opt.optimize()
        print(opt.get_logs())

    def test_direct(self):

        from R.opto.functions.Branin import Branin
        from dotmap import DotMap

        task = Branin()
        stopCriteria = opto.classes.StopCriteria(maxEvals=10000)

        p = DotMap()
        p.verbosity = 1
        opt = opto.DIRECT(parameters=p, task=task, stopCriteria=stopCriteria)
        opt.optimize()
        print(opt.get_logs())

    def test_grid_search_1(self):

        from R.opto.functions.Branin import Branin
        from dotmap import DotMap

        task = Branin()
        stopCriteria = opto.classes.StopCriteria(maxEvals=10000)

        p = DotMap()
        p.verbosity = 1
        p.resolution = 100
        opt = opto.GridSearch(parameters=p, task=task, stopCriteria=stopCriteria)
        opt.optimize()
        print(opt.get_logs())

    def test_grid_search_2(self):

        from R.opto.functions.Branin import Branin
        from dotmap import DotMap

        task = Branin()
        stopCriteria = opto.classes.StopCriteria(maxEvals=10000)

        p = DotMap()
        p.verbosity = 1
        p.resolution = [10, 10]
        opt = opto.GridSearch(parameters=p, task=task, stopCriteria=stopCriteria)
        opt.optimize()
        print(opt.get_logs())

if __name__ == '__main__':
    unittest.main()
