Installation
============

.. code-block:: bash

   pip install tsauditor

Requires Python >= 3.9. Core dependencies: ``pandas``, ``numpy``, ``scipy``,
``statsmodels``, ``rich``.

Optional extras (install only what you need):

.. code-block:: bash

   pip install 'tsauditor[pdf]'      # PDF report export (matplotlib)
   pip install 'tsauditor[polars]'   # polars DataFrame input
   pip install 'tsauditor[dev]'      # test + lint toolchain (contributors)

Development setup
------------------

.. code-block:: bash

   git clone https://github.com/imann128/tsauditor.git
   cd tsauditor
   pip install -e ".[dev]"

.. note::

   For domain-agnostic usage, either omit ``domain=`` entirely or pass
   ``domain=None`` (Python's ``None``, not the string ``"None"``: passing
   the string raises ``ValueError``, since only ``"finance"``, ``"sensor"``,
   and ``None`` are accepted). Omitting it is equivalent; ``None`` is
   already the default.
