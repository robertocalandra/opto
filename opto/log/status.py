from __future__ import print_function

from colorama import Fore
import opto.utils as rutils
import sys


def status(f):
    """

    :param f:
    :return:
    """
    if f == 0:
        print('[' + Fore.GREEN + 'Done' + Fore.RESET + ']')  # Python 3: , flush=True)
    if f < 0:
        print('[' + Fore.RED + 'Error' + Fore.RESET + ']')  # Python 3: , flush=True)
    if f > 0:
        print('[' + Fore.MAGENTA + '???' + Fore.RESET + ']')  # Python 3: , flush=True)
    sys.stdout.flush()  # Force flush in Python 2


def cnd_status(current_verbosity, necessary_verbosity, f, cnt_verbosity=float('inf'), indent=0):
    if necessary_verbosity < current_verbosity:
        rutils.indent(indent)  # Indent
        status(f)
