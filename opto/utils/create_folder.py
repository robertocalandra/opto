# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

import os


def create_folder(nameFolder):
    if not os.path.exists(nameFolder):
        try:
            os.makedirs(nameFolder)
        except:
            print('Unable to create folder')
