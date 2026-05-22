"""
tsauditor.report.summary
------------------------
Defines the GuardReport and Issue dataclasses that form the
structured output of every tsauditor scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich import box


# ── Severity constants ────────────────────────────────────────────────────────
CRITICAL = "critical"
WARNING  = "warning"
INFO     = "info"

_SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}
_SEVERITY_COLOR = {CRITICAL: "bold red", WARNING: "yellow", INFO: "cyan"}


@dataclass
class Issue:
    """
    A single quality issue detected by tsauditor.

    Attributes
    ----------
    module : str
        Which module raised the issue: "profiler", "anomaly", or "leakage".
    code : str
        Short issue code (e.g. "LEK001"). Use for programmatic filtering.
    severity : str
        One of "critical", "warning", "info".
    description : str
        Human-readable explanation of the issue.
    column : Optional[str]
        The affected column, or None if dataset-level.
    evidence : Dict[str, Any]
        Supporting statistics (e.g. {"lag0_corr": 0.99, "threshold": 0.95}).
    """

    module: str
    code: str
    severity: str
    description: str
    column: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module":      self.module,
            "code":        self.code,
            "severity":    self.severity,
            "description": self.description,
            "column":      self.column,
            "evidence":    self.evidence,
        }


@dataclass
class GuardReport:
    """
    The structured output of a tsauditor.scan() call.

    Attributes
    ----------
    critical : List[Issue]
        Issues that must be fixed before modeling.
    warnings : List[Issue]
        Issues worth reviewing but not necessarily blockers.
    info : List[Issue]
        Informational findings.
    metadata : Dict[str, Any]
        Dataset-level metadata: rows, columns, time range, inferred frequency.
    """

    critical: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)
    info:     List[Issue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def all_issues(self) -> List[Issue]:
        """All issues sorted by severity then module."""
        return sorted(
            self.critical + self.warnings + self.info,
            key=lambda i: (_SEVERITY_ORDER.get(i.severity, 9), i.module),
        )

    def filter(self, code: Optional[str] = None,
               module: Optional[str] = None,
               severity: Optional[str] = None) -> List[Issue]:
        """
        Return issues matching all supplied filters.

        Examples
        --------
        >>> report.filter(code="LEK001")
        >>> report.filter(module="leakage", severity="critical")
        """
        issues = self.all_issues
        if code is not None:
            issues = [i for i in issues if i.code == code]
        if module is not None:
            issues = [i for i in issues if i.module == module]
        if severity is not None:
            issues = [i for i in issues if i.severity == severity]
        return issues

    # ── Output methods ────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a formatted CLI summary using rich."""
        console = Console()

        # Header
        console.rule("[bold]tsauditor Report[/bold]")

        # Metadata block
        meta = self.metadata
        console.print(f"\n[bold]Dataset[/bold]")
        console.print(f"  Rows       : {meta.get('rows', 'N/A')}")
        console.print(f"  Columns    : {meta.get('columns', 'N/A')}")
        console.print(f"  Time range : {meta.get('time_start', '?')} → {meta.get('time_end', '?')}")
        console.print(f"  Frequency  : {meta.get('frequency', 'unknown')}")

        # Issue counts
        console.print(
            f"\n[bold red]Critical[/bold red]: {len(self.critical)}  "
            f"[yellow]Warnings[/yellow]: {len(self.warnings)}  "
            f"[cyan]Info[/cyan]: {len(self.info)}\n"
        )

        if not self.all_issues:
            console.print("[green]No issues detected.[/green]\n")
            return

        # Issues table
        table = Table(box=box.SIMPLE_HEAVY, show_lines=False, expand=True)
        table.add_column("Severity", style="bold", width=10)
        table.add_column("Code",     width=8)
        table.add_column("Module",   width=10)
        table.add_column("Column",   width=16)
        table.add_column("Description")

        for issue in self.all_issues:
            color = _SEVERITY_COLOR.get(issue.severity, "white")
            table.add_row(
                f"[{color}]{issue.severity.upper()}[/{color}]",
                issue.code,
                issue.module,
                issue.column or "—",
                issue.description,
            )

        console.print(table)

    def to_json(self, path: str) -> None:
        """
        Export the full report to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "report.json").
        """
        payload = {
            "metadata": self.metadata,
            "issues":   [i.to_dict() for i in self.all_issues],
            "counts": {
                "critical": len(self.critical),
                "warnings": len(self.warnings),
                "info":     len(self.info),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Return the full report as a plain Python dict."""
        return {
            "metadata": self.metadata,
            "issues":   [i.to_dict() for i in self.all_issues],
            "counts": {
                "critical": len(self.critical),
                "warnings": len(self.warnings),
                "info":     len(self.info),
            },
        }
