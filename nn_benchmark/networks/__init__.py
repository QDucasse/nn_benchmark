# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

from __future__ import absolute_import

__all__ = ["lenet","lenet5","quant_lenet5",
           "quant_cnv",
           "mobilenetv1","quant_mobilenetv1",
           "vggnet", "quant_vggnet",
           "common"]

from .lenet             import *
from .lenet5            import *
from .mobilenetv1       import *
from .quant_mobilenetv1 import *
from .quant_lenet5      import *
from .quant_cnv         import *
from .vggnet            import *
from .quant_vggnet      import *
from .common            import *
