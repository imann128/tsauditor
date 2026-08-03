Quickstart
==========

.. code-block:: python

   import tsauditor as tsa

   report = tsa.scan(df, target="Direction", domain="finance")

   report.summary()                 # rich-formatted CLI table
   report.critical                  # list[Issue] that block modeling
   report.filter(module="leakage")  # programmatic filtering
   report.leaky_columns()           # the shortlist of features to review/remove
   report.to_json("report.json")    # structured export

   # Repair on a copy and keep the audit trail (original is never modified):
   clean, report = tsa.fix(df, target="Direction", domain="finance")
   print(report.last_fixes)         # exactly what changed

``scan()`` returns a :class:`~tsauditor.GuardReport` holding
:class:`~tsauditor.Issue` dataclasses bucketed by severity (``critical``,
``warnings``, ``info``) plus dataset metadata.

Panel (long-format, multi-entity) data
----------------------------------------

If your frame stacks many entities under a repeated timestamp column,
pass ``group_col=`` and each entity is audited as its own independent
time series:

.. code-block:: python

   report = tsa.scan(panel, target="Direction", group_col="ticker", domain="finance")
   report.summary()          # prevalence view: which findings are systemic vs isolated

For a panel with many entities, pass ``n_jobs=-1`` to audit entities in
parallel across all available cores instead of sequentially.

New to tsauditor?
-------------------

Start with `examples/getting_started
<https://github.com/imann128/tsauditor/tree/main/examples/getting_started>`_,
a from-zero, runnable notebook covering the full scan -> read -> repair
workflow on a small synthetic dataset.

For everything added in the most recent release, see `examples/whats_new_0_4_0
<https://github.com/imann128/tsauditor/tree/main/examples/whats_new_0_4_0>`_.
