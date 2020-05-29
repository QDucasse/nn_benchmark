# pyquickstart

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
