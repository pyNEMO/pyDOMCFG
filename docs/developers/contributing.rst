Contributing
************

Documentation
=============

- The documentation consists of two parts: the docstrings in the code itself
  and the docs in this folder ``pyDOMCFG/docs/``.

- The documentation is written in `reStructuredText <http://sphinx-doc.org/>`_
  and built using `Sphinx <http://sphinx-doc.org/>`_.

- The docstrings follow the `Numpy Docstring Standard
  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

How to build the documentation:

.. code-block:: sh

    # Create and activate the docs environment
    conda env create -f pyDOMCFG/ci/docs.yml
    conda activate pydomcfg_docs

    # Navigate to the docs directory
    cd pyDOMCFG/docs/

    # If you want to do a full clean build
    make clean

    # Build the documentation
    make html
