"""
Hand-built ``Issue`` fixtures must use the severity their detector actually
emits.

Why this exists
---------------
Three fixtures constructed PRF003 as WARNING. The detector emits it as INFO, and
the README and wiki both document it as INFO. Nothing failed, because ``Issue``
is a plain dataclass that accepts any string.

The cost was not a broken test. It was that contributors copy the nearest
existing fixture, so the wrong severity propagated into pull requests, and the
same correction had to be made in review more than once. A fixture is
documentation whether or not it is written as documentation.

This test reads the detectors to find what each code's severity really is, then
checks every ``Issue(...)`` constructed anywhere in ``tests/`` against it.
"""

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "tsauditor"
_TESTS = _ROOT / "tests"

_CONSTANT_TO_VALUE = {"CRITICAL": "critical", "WARNING": "warning", "INFO": "info"}


def _severity_of(node):
    """Resolve a severity argument written as a constant name or a literal."""
    if isinstance(node, ast.Name):
        return _CONSTANT_TO_VALUE.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _issue_calls(path):
    """Yield (lineno, code, severity) for every Issue(...) built in a file."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Issue"
        ):
            continue
        kwargs = {k.arg: k.value for k in node.keywords}
        # Issue(module, code, severity, description, ...) positionally, or by keyword.
        code = node.args[1] if len(node.args) > 1 else kwargs.get("code")
        sev = node.args[2] if len(node.args) > 2 else kwargs.get("severity")
        if not (isinstance(code, ast.Constant) and isinstance(code.value, str)):
            continue
        resolved = _severity_of(sev)
        if resolved is not None:
            yield node.lineno, code.value, resolved


def _detector_severities():
    """code -> the severity the library itself emits for it."""
    truth = {}
    for path in _PACKAGE.rglob("*.py"):
        for _, code, severity in _issue_calls(path):
            truth.setdefault(code, set()).add(severity)
    return truth


def test_every_code_has_exactly_one_severity_in_the_library():
    """A code that is CRITICAL in one detector and WARNING in another would make
    the severity meaningless to anyone filtering on it."""
    ambiguous = {c: sorted(s) for c, s in _detector_severities().items() if len(s) > 1}
    assert ambiguous == {}, f"codes emitted with more than one severity: {ambiguous}"


def test_test_fixtures_use_the_real_severity():
    truth = _detector_severities()
    wrong = []
    for path in sorted(_TESTS.rglob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        for lineno, code, severity in _issue_calls(path):
            expected = truth.get(code)
            if expected and severity not in expected:
                wrong.append(
                    f"{path.relative_to(_ROOT)}:{lineno} builds {code} as "
                    f"{severity!r}, but the detector emits {sorted(expected)[0]!r}"
                )
    assert not wrong, "\n" + "\n".join(wrong)


@pytest.mark.parametrize("code, expected", [("PRF003", "info"), ("PRF007", "critical")])
def test_known_severities_pinned(code, expected):
    """Explicit pins for the two that have caused confusion: PRF003 reads like a
    warning but is INFO, and PRF007 reads like a warning but is CRITICAL."""
    assert _detector_severities()[code] == {expected}
