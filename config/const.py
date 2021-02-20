from os import sep
from os.path import join, dirname, realpath

PROJECT_ROOT = join(sep, *dirname(realpath(__file__)).split(sep)[: -1])
