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

The CLI assumes the location and file name format of intermediate files, so that for a standard analysis you only need to specify minimal options, e.g.:

.. code-block:: console

    (.venv) $ pySPADS decompose ./input --signal shore --timecol date --noise 0.1 0.2 0.3 0.4 0.5
    (.venv) $ pySPADS match --signal shore
    (.venv) $ pySPADS fit --signal shore --model mreg2 --fit-intecept --normalize
    (.venv) $ pySPADS predict --signal shore

The commands above should perform a complete analysis, generating the following files, relative to the current directory:

.. list-table::
    :widths: 50 50
    :header-rows: 0

    * - ./imfs/<column_name>_imf_<noise>.csv
      - Decomposed IMFs for each input signal and noise level
    * - ./frequencies/frequencies_<noise>.csv
      - The indices of the matched frequencies between signal and driver IMFs
    * - ./coefficients/coefficients_<noise>.csv
      - The coefficients of the linear model expressing each signal component as a function of the driver components
    * - ./predictions_<noise>.csv
      - The prediction of the target signal from the fitted model, for each noise level
    * - ./reconstructed_total.csv
      - The total signal reconstructed from the predicted components, averaged over all noise levels


CLI reference
=============

.. click :: pySPADS.cli:cli
   :prog: pySPADS
   :nested: full
