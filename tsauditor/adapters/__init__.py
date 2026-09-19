"""
tsauditor.adapters
------------------
Boundary adapters that turn an audited-and-repaired DataFrame into the exact
input another library expects. Adapters live at the edge of tsauditor: they call
the core audit/fix engine, then reshape the result. They never add heavy or
model-specific dependencies to the library itself.

Currently provided
------------------
to_timesfm  Audit, repair, and format a single series into a 1-D float32 array
            for Google TimesFM inference.
"""

from tsauditor.adapters.timesfm import to_timesfm

__all__ = ["to_timesfm"]
