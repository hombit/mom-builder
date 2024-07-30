"""Module for building multi-order healpix maps (MOMs).

The module provides multiple interfaces to build healpix MOMs merging
healpix tiles of some maximum order (healpix tile tree depth) into
tiles (leaves) of some lower order (top tree depth).
"""

from .mom_builder import *
from .mom_generator import gen_mom_from_fn
