import copy
import os

from config.parse_console import parse_console
from config.defaults import _C


_C.__init__()


def update_args(cfg_file=None):
    _C.defrost()
    if cfg_file and os.path.exists(cfg_file):
        _C.merge_from_file(cfg_file)
    else:
        print('cfg_file %s not found, using default settings.' % cfg_file)
    return copy.deepcopy(_C)


__all__ = ['parse_console', 'update_args']
