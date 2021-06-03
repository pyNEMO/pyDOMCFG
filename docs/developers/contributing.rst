Contributing
************

Version control
===============

We use `Git <https://git-scm.com/>`_ for version control.
The code is hosted on `GitHub <https://github.com/>`_.

How to fork pyDOMCFG, create a branch, and commit your code:

1. Go to `pyDOMCFG <https://github.com/pyNEMO/pyDOMCFG.git>`_ and click the ``Fork`` button.

2. Clone your fork and connect the repository to the upstream repository:

.. code-block:: sh

    git clone https://github.com/your-user-name/pyDOMCFG.git
    cd pyDOMCFG
    git remote add upstream https://github.com/pyNEMO/pyDOMCFG.git

3. Create a branch:

.. code-block:: sh

    git checkout -b name-of-the-branch

.. note::

    Members of `@pyNEMO/pydomcfg <https://github.com/orgs/pyNEMO/teams/pydomcfg>`_
    can create a branch directly from main:

    .. code-block:: sh
        
        git clone https://github.com/pyNEMO/pyDOMCFG.git
        cd pyDOMCFG
        git checkout -b name-of-the-branch

    This is particularly useful for long-running branches (e.g., develop, stable, ...).

4. To update this branch retrieving changes from the main branch:

.. code-block:: sh

    git fetch upstream
    git merge upstream/main

5. Once you have made changes, commit your code:

.. code-block:: sh

    # To check changes:
    git status

    # If you have created new files:
    git add path/to/file-to-be-added

    # Commit changes:
    git commit -m "short message describing changes"

6. Push your commits:

.. code-block:: sh

    git push origin name-of-the-branch

7. Navigate to your repository on GitHub (`https://github.com/your-user-name/pyDOMCFG`),
   click on the ``Pull Request`` button, edit title and description, and click
   ``Send Pull Request``.



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


Tests
=====

- All tests go into this folder ``pyDOMCFG/pydomcfg/tests``.

- We are using `pytest <http://doc.pytest.org/en/latest/>`_ for testing.

Test functions should look like this:

.. code-block:: python

    def add_one(x):
        return x + 1


    def test_add_one():
        expected = 2
        actual = add_one(1)
        assert expected == actual

How to run the tests:

.. code-block:: sh

    # Create and activate the test environment
    conda env create -f pyDOMCFG/ci/environment.yml
    conda activate pydomcfg_test

    # Navigate to the root directory, install, and run pytest
    cd pyDOMCFG
    pip install -e .
    pytest


Pre-commit formatting
=====================

We are using several tools to ensure that code and docs are well formatted:

- `isort <https://github.com/timothycrosley/isort>`_
  for standardized order in imports.
- `Black <https://black.readthedocs.io/en/stable/>`_
  for standardized code formatting.
- `blackdoc <https://blackdoc.readthedocs.io/en/stable/>`_
  for standardized code formatting in documentation.
- `Flake8 <http://flake8.pycqa.org/en/latest/>`_ for general code quality.
- `Darglint <https://github.com/terrencepreilly/darglint>`_ for docstring quality.
- `mypy <http://mypy-lang.org/>`_ for static type checking on
  `type hints <https://docs.python.org/3/library/typing.html>`_.
- `doc8 <https://github.com/PyCQA/doc8>`_ for reStructuredText documentation quality.

Setup `pre-commit <https://pre-commit.com/>`_ hooks to automatically run all
the above tools every time you make a git commit:

.. code-block:: sh

    # Install the pre-commit package manager.
    conda install -c conda-forge pre-commit

    # Set up the git hook scripts.
    cd pyDOMCFG
    pre-commit install

    # Now pre-commit will run automatically on the changed files on ``git commit``
    # Alternatively, you can manually run all the hooks with:
    pre-commit run --all

    # You can skip the pre-commit checks with:
    git commit --no-verify
