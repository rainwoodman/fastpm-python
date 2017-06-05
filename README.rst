fastpm
======

This is the Python implementation of the FastPM numerical scheme for quasi-nbody simulations.

CI status / master

.. image:: https://travis-ci.org/rainwoodman/fastpm-python.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/fastpm-python

Install
-------

The best result is obtained by using anaconda.

Python 3 is the current development environment.

First set up the basic requirements,

.. code::

    conda install cython numpy scipy mpi4py nose
    conda install -c bccp nbodykit

    # update nbodykit to latest
    pip install -U --no-deps https://github.com/bccp/nbodykit/archive/master.zip


Finally, we can install fastpm, either from PYPI (the latest release)

.. code::

    pip install fastpm

or from the git clone :

.. code::

    pip install .

Command line interface
----------------------

There is a simple command line interface

.. code::

    python -m fastpm.main examples/run


Development
-----------

To run the tests

.. code::

    python runtests.py

.. code::

    python runtests.py --single

To run a single test (e.g. `test_fastpm.py:test_name`) :

.. code::

    python runtests.py fastpm/tests/test_fastpm:test_name



