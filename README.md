# Neural Network benchmark

This project consists of an utility wrapper around PyTorch and Brevitas to specify the network, dataset and parameters to train with.

### Usage

You can run the training of `LeNet` on the `MNIST` dataset as follows:
```bash
PYTORCH_JIT=1 python nn_benchmark/main.py --network LeNet --dataset MNIST --epochs 3
```
The results can be observed under the experiments folder.

You can then evaluate your network with the following command:
```bash
python nn_benchmark/main.py --evaluate --resume ./experiments/<your_folder>/checkpoints/best.tar
```
### Available networks and datasets

The following networks are supported:
- LeNet
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

### Installation

I worked on the project through a virtual environment with `virtualenvwrapper`
and I highly recommend to do so as well. However, whether or not you are in a
virtual environment, the installation proceeds as follows:

* For downloading and installing the source code of the project:

  ```bash
    $ cd <directory you want to install to>
    $ git clone https://github.com/QDucasse/pyquickstart
    $ python setup.py install
  ```
* For downloading and installing the source code of the project in a new virtual environment:  

  *Download of the source code & Creation of the virtual environment*
  ```bash
    $ cd <directory you want to install to>
    $ git clone https://github.com/QDucasse/pyquickstart
    $ cd pyquickstart
    $ mkvirtualenv -a . -r requirements.txt VIRTUALENV_NAME
  ```
  *Launch of the environment & installation of the project*
  ```bash
    $ workon VIRTUALENV_NAME
    $ pip install -e .
  ```

Finally, whether you choose the first or second option, you will need brevitas if you want to use quantized networks. The installation is better performed from source and can be done as follows (in your native or virtual environment):

```bash
    $ git clone https://github.com/Xilinx/brevitas.git
    $ cd brevitas
    $ pip install -e .
```
---
### Structure of the project

Quick presentation of the different modules of the project:
* [**Package1:**][package]
Dynamic systems models.
---
### Requirements

This project uses the following external libraries:
* [`Numpy`][dependency1]

If installed as specified above, the requirements are stated in the ``requirements.txt`` file
and therefore automatically installed.  
However, you can install each of them separately with the command:
```bash
  $ pip install <library>
```

---
### Objectives and Milestones of the project

- [X] Basic project structure
---

### Testing

All tests are written to work with `nose` and/or `pytest`. Just type `pytest` or
`nosetests` as a command line in the project. Every test file can still be launched
by executing the testfile itself.
```bash
  $ python pyquickstart/tests/chosentest.py
  $ pytest
```

---

### References

[package]:https://github.com/QDucasse/pyquickstart/tree/master/pyquickstart/package
[dependency1]: https://numpy.org/
