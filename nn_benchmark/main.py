# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import sys
from nn_benchmark.core import CLI

if __name__ == "__main__":
    cli = CLI(sys.argv[1:])
    print(cli.args)
    cli.main()
