# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import R.opto as opto
from R.opto.functions.Branin import Branin
from dotmap import DotMap

task = Branin()

print(task.get_labels_parameters())
print(task.get_labels_objectives())

import matplotlib.pyplot as plt
import matplotlib as mpl
with mpl.rc_context(rc={'interactive': False}):
    task.visualize()
    plt.show()
