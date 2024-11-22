Usage
=====

Installation
------------

To use pySPADS, first install it using pip - it is recommended you do this in a `virtual environment <https://docs.python.org/3/library/venv.html>`_.:

.. code-block:: console

    (.venv) $ pip install pySPADS

Usage
-----

There are three main ways to use pySPADS:
- As a command line interface (CLI) tool (work in progress, not yet recommended)
- As a Python package, by importing the relevant modules and functions into your own script (recommended)
- By using `Snakemake <https://snakemake.readthedocs.io/en/stable/>`_ to create and run a pipeline

Scripting
---------

Example scripts, on which to base your own, can be found in the `scripts directory <https://github.com/UoA-eResearch/pySPADS/tree/main/scripts>`_ of the `pySPADS repository <https://github.com/UoA-eResearch/pySPADS>`_.

Data
----

Example data can be found in the `data directory <https://github.com/UoA-eResearch/pySPADS/tree/main/data/example_run>`_ of the pySPADS repository.

pySPADS expects data to be provided as a set of CSV files, each having a datetime column (with a consistent label across input files), and any number of timeseries columns.
Timeseries with matching indices can be combined into a single CSV file with multiple columns, or provided as separate files with duplicate datetime columns.

pySPADS expects data to be sampled daily. Data sampled more frequently will be averaged within each day, data sampled at a lower frequency will be linearly interpolated to daily values.

The *config.yaml* file within the example data directory contains configuration details used by the Snakemake pipeline. This file can be modified to suit your own data and requirements, but is not needed by either the CLI or example scripts.