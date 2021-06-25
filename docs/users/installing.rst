Installation
============

Required dependencies
---------------------

- Python (3.7 or later)
- setuptools
- `xarray <http://xarray.pydata.org/>`_
- `numpy <http://www.numpy.org/>`_

Optional dependencies
---------------------

- `f90nml <https://f90nml.readthedocs.io/>`_

Instructions
------------

pyDOMCFG must be installed from source because
it has not been released on `PyPI <https://pypi.org/>`_ yet.
The best way to install all dependencies is to use `conda <http://conda.io/>`_.

.. code-block:: sh

    conda install -c conda-forge xarray f90nml pip
    pip install git+https://github.com/pyNEMO/pyDOMCFG.git
