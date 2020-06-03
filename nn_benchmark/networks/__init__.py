# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

from __future__ import absolute_import

__all__ = ["lenet","lenet5","mobilenetv1","quant_lenet5","quant_cnv","vggnet"]

from .lenet        import *
from .lenet5       import *
from .mobilenetv1  import *
from .quant_lenet5 import *
from .quant_cnv    import *
from .vggnet       import *
