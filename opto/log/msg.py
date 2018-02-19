from __future__ import print_function
from opto.utils import indent
from colorama import Fore
# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass


def msg(string, indent_depth=0, eol=True):
    """

    :param string:
    :param indent_depth:
    :param eol:
    :return:
    """
    assert indent_depth >= 0
    log_msg(string)
    indent(indent_depth)
    if eol:
        print(string)
    else:
        print(string, end="")
    # TODO: force flushing!!!


def log_msg(string):
    """

    :param string:
    :return:
    """
    logging.getLogger(__name__).addHandler(NullHandler())
    logging.info(string)


def cnd_msg(current_verbosity, necessary_verbosity, string, indent_depth=0, cnt_verbosity=float('inf'), eol=False):
    """

    :param current_verbosity:
    :param necessary_verbosity:
    :param string: String to be logged
    :param indent_depth: Indentation level
    :param cnt_verbosity:
    :param eol:
    :return:
    """
    if necessary_verbosity < current_verbosity:
        msg(string, indent_depth, eol=False)
        if cnt_verbosity is not float('inf'):
            if eol:
                print('\n', end="")
            else:
                if current_verbosity < cnt_verbosity:
                    print('... ', end="")
                else:
                    print(':\n', end="")
    else:
        log_msg(string)  # Just log, no print to screen
    return [current_verbosity, necessary_verbosity, indent_depth, cnt_verbosity]


def cnd_warning(current_verbosity, necessary_verbosity, string, indent_depth=0, cnt_verbosity=float('inf'), eol=False):
    """

    :param current_verbosity:
    :param necessary_verbosity:
    :param string:
    :param indent_depth:
    :param cnt_verbosity:
    :param eol:
    :return:
    """
    if necessary_verbosity < current_verbosity:
        msg(string, indent_depth, eol=False)
    else:
        log_warning(string)


def warning(string):
    log_warning(string)
    msg(color('Warning: ' + string, 'red'))


def log_warning(string):
    logging.getLogger(__name__).addHandler(NullHandler())
    logging.warning(string)


def color(string, color):
    # TODO: implement me !
    out = Fore.RED + string + Fore.RESET
    return out
