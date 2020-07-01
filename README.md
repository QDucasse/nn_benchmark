# Neural Network benchmark

This project consists of an utility wrapper around PyTorch and Brevitas to specify the network, dataset and parameters to train with.

### Usage

You can run the training of `LeNet` on the `MNIST` dataset as follows:
```bash
$ PYTORCH_JIT=1 python nn_benchmark/main.py --network LeNet --dataset MNIST --epochs 3
```

Or its quantized version `QuantLeNet` on the `CIFAR-10` dataset with bit-width of (4,4,8) (corresponding to activation, weight and input) :
```bash
$ PYTORCH_JIT=1 python nn_benchmark/main.py --network QuantLeNet --dataset CIFAR10 --epochs 3 \
$ --acq 4 --weq 4 --inq 8
```
The results can be observed under the experiments folder.

You can then evaluate your network with the following command:
```bash
$ python nn_benchmark/main.py --network LeNet --dataset MNIST --evaluate --resume ./experiments/<your_folder>/checkpoints/best.tar
```
### Available networks and datasets

The following networks are supported:
- LeNet5
- VGG (11/13/16/19)
- MobilenetV1

Their quantized counterparts are available as well:
- QuantLeNet5
- QuantVGG (11/13/16/19)
- QuantMobilenetV1

The following datasets can be used:
- MNIST
- FASHION-MNIST
- CIFAR10
- GTSRB

### Installation

I worked on the project through a virtual environment with `virtualenvwrapper`
and I highly recommend to do so as well. However, whether or not you are in a
virtual environment, the installation proceeds as follows:

* For downloading and installing the source code of the project:

  ```bash
    $ cd <directory you want to install to>
    $ git clone https://github.com/QDucasse/nn_benchmark
    $ python setup.py install
  ```
* For downloading and installing the source code of the project in a new virtual environment:  

  *Download of the source code & Creation of the virtual environment*
  ```bash
    $ cd <directory you want to install to>
    $ git clone https://github.com/QDucasse/nn_benchmark
    $ cd nn_benchmark
    $ mkvirtualenv -a . -r requirements.txt VIRTUALENV_NAME
  ```
  *Launch of the environment & installation of the project*
  ```bash
    $ workon VIRTUALENV_NAME
    $ pip install -e .
  ```

Finally, whether you chose the first or second option, you will need brevitas if you want to use quantized networks. The installation is better performed from source and can be done as follows (in your native or virtual environment):

```bash
    $ git clone https://github.com/Xilinx/brevitas.git
    $ cd brevitas
    $ pip install -e .
```
---
### Structure of the project

Quick presentation of the different modules of the project:
* [**Core:**][core] Core functionalities of the project such as the `CLI`, `Logger`, `Plotter` and `Trainer`.
* [**Extensions:**][extensions] Extended functionalities of the `PyTorch` modules. This package contains specific modules (e.g. `TensorNorm`), dataset (e.g. `GTSRB`) or loss functions (e.g. `SquaredHinge`) that can be used as drop-in replacements for their `PyTorch` homologues.
* [**Networks:**][networks] Network architectures implemented in the project that can be used with the CLI `--network` flag. If the suffix `Quant` is present, this means the network is quantized and the three following precisions can be specified: `weight_bit_width`, precision of the weights ; `act_bit_width` precision of the activation functions ; `in_bit_width`, input precision (this is useful to keep a higher precision at the beginning).
---
### Requirements

This project uses the following external libraries:
- [`numpy`](https://numpy.org/)
- [`torch`](https://pytorch.org/)
- [`torchvision`](https://pytorch.org/docs/stable/torchvision/index.html)
- [`matplotlib`](https://matplotlib.org/)
- [`brevitas`](https://xilinx.github.io/brevitas/)

If installed as specified above, the requirements are stated in the ``requirements.txt`` file
and therefore automatically installed.  
However, you can install each of them separately with the command (except for `brevitas`, please follow the installation from source provided at the end of the installation paragraph):

```bash
  $ pip install <library>
```

---
### Objectives and Milestones of the project

- [x] Basic project structure
- [x] `Trainer`/`Logger` logic
- [x] `LeNet` network training on `MNIST`
- [x] `LeNet5`, `MobilenetV1`, `VGG11`, `VGG13`, `VGG16`, `VGG19` network architectures
- [x] Quantized counterparts for the networks
- [x] `PyTorch` extensions with `GTSRB` dataset and another loss function (`SqrHinge`)
- [x] Extend trainer with `Plotter`
---

### Testing

All tests are written to work with `nose` and/or `pytest`. Just type `pytest` or
`nosetests` as a command line in the project. Every test file can still be launched
by executing the testfile itself.
```bash
  $ python nn_benchmark/tests/chosentest.py
  $ pytest
```

---

### References

[core]:https://github.com/QDucasse/nn_benchmark/tree/master/nn_benchmark/core	"core package"
[extensions]: https://github.com/QDucasse/nn_benchmark/tree/master/nn_benchmark/extensions	"extensions package"

[networks]: https://github.com/QDucasse/nn_benchmark/tree/master/nn_benchmark/networks	"networks package"
