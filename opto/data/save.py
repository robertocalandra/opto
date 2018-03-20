try:
    import cPickle as pickle  # Python 2
except ImportError:
    import pickle  # Python 3
# import numpy as np
import opto.log as rlog
# import re

__author__ = 'Roberto Calandra'
__version__ = '0.4'


def save(data, fileName, verbosity=1, indent=0):
    """
    Save class
    :param data:
    :param fileName: str
    :param verbosity:
    :param indent:
    :return:
    """
    # m = re.search('\w+(?<=.pkl)', fileName)  # Remove .pkl if existing
    # fileName = m.group(0) + '.pkl'  # Add .pkl
    rlog.cnd_msg(verbosity, 0, 'Saving pickle file: ' + fileName, indent_depth=indent)
    try:
        f = open(fileName, 'wb')
        f.write(pickle.dumps(data))
        f.close()
        status = 0
    except:
        rlog.warning('Unable to save file: ' + fileName)
        status = -1
    rlog.cnd_status(verbosity, 0, status)
    return status


def load(fileName, verbosity=1, indent=0):
    """
    Load class
    :param fileName:
    :param verbosity:
    :param indent:
    :return:
    """
    # fileName +=  '.pkl'
    rlog.cnd_msg(verbosity, 0, 'Load pickle file: ' + fileName, indent_depth=indent)
    try:
        f = open(fileName, 'r')
        dataPickle = f.read()
        f.close()
        out = pickle.loads(dataPickle)
        status = 0
    except:
        rlog.warning('Unable to load file: ' + fileName)
        status = -1
        out = None
    rlog.cnd_status(verbosity, 0, status)
    return out
