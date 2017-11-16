fastpm
======

This is the Python implementation of the FastPM numerical scheme for quasi-nbody simulations.

CI status / master

.. image:: https://travis-ci.org/rainwoodman/fastpm-python.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/fastpm-python

DOI of fastpm-python.

.. image:: https://zenodo.org/badge/81290989.svg
   :target: https://zenodo.org/badge/latestdoi/81290989
   
   
.. image:: https://github.com/rainwoodman/fastpm-python/raw/artwork/artwork/10MpchTrajectories.png
    :align: center

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

There is a simple command line interface, which expects a config.py in the
first command line argument.

.. code::

    python -m fastpm.main examples/run

or with MPI

.. code::

    mpirun -n 4 python -m fastpm.main examples/run

The arguments are listed in `fastpm/main.py`

Development
-----------

To run the tests with MPI

.. code::

    python runtests.py

Run with a single rank and enable debugging

.. code::

    python runtests.py --single --pdb --capture=no

To run a single test (e.g. `test_fastpm.py:test_name`) :

.. code::

    python runtests.py fastpm/tests/test_fastpm:test_name


Profiling
---------

.. code::

    python -m cProfile -o profile.stats run.py run
    gprof2dot profile.stats -f pstats | dot -Tpng > profile.png

We can't use `-m fastpm.main` directory because there is no nested `-m` support.
