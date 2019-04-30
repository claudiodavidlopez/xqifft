import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(FILE_PATH, '..')
sys.path.insert(0, os.path.abspath(MODULE_PATH))

import xqifft
