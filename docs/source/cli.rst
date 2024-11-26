===
CLI
===

Installation
============
To use pySPADS, first install it using pip - it is recommended you do this in a `virtual environment <https://docs.python.org/3/library/venv.html>`_.:

.. code-block:: console

    (.venv) $ pip install pySPADS

Once installed, the CLI will be available as the command `pySPADS`. If installed under a virtual envionment, you will first need to activate the environment before running the command. If installed globally, the command should be available from any terminal.

Run `pySPADS --help` to see the available commands and options.

Steps
=====

**pySPADS** analysis occurs in several steps, each of which is implemented as a separate command:

1. `pySPADS decompose` - Decomposition of signal and driver timeseries into IMFs
2. `pySPADS match` - Matching of component frequencies between signal and driver IMFs
3. `pySPADS fit` - Fitting of a linear model to predict each signal component from the driver components
4. `pySPADS predict` - Prediction of future signal components using the fitted model

For more information on each command, run `pySPADS <command> --help`.

CLI reference
=============

.. click :: pySPADS.cli:cli
   :prog: pySPADS
   :nested: full
