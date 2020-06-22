# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

from __future__ import absolute_import

__all__ = ["test_lenet5","test_quant_lenet5",
           "test_mobilenetv1","test_quant_mobilenetv1",
           "test_vggnet", "test_quant_vggnet"]

from .test_lenet5            import *
from .test_quant_lenet5      import *
from .test_mobilenetv1       import *
from .test_quant_mobilenetv1 import *
from .test_vggnet            import *
from .test_quant_vggnet      import *
from .test_common            import *
