tsauditor
=========

A data quality auditing library for time-series tabular data, with a focus
on financial and sensor domains. ``tsauditor`` scans a ``DataFrame`` and
returns a structured report of structural problems, anomalies, and
data-leakage between features and the prediction target.

.. code-block:: python

   import tsauditor as tsa

   report = tsa.scan(df, target="Direction", domain="finance")
   report.summary()
   report.critical
   report.to_json("report.json")

   # one-shot scan-and-repair, keeping the audit trail:
   clean, report = tsa.fix(df, target="Direction", domain="finance")

For a from-zero, runnable walkthrough of the whole scan -> read -> repair
workflow, see :doc:`quickstart`. For everything added in the most recent
release, see the `0.4.0 tutorial
<https://github.com/imann128/tsauditor/tree/main/examples/whats_new_0_4_0>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/scanner
   api/remediate
   api/report
   api/leakage
   api/profiler
   api/anomaly
   api/validity
   api/panel
   api/adapters

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
