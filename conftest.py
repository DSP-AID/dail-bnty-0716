"""Make the package importable when pytest runs from any directory."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
