"""Turn the Gaussian routes of a log into an AutoStorage model.

A model holds only what changes the result. The decoder gives the method and basis as
the archive names them (``UM062X``, ``def2TZVP``) and every route of the log
(``# opt freq M062X/Def2TZVP scf=(xqc,...) nosymm``). From these:

- the reference (``R``, ``U`` or ``RO`` prefix of the method) goes in
  ``keywords["reference"]`` as ``rks``, ``uks`` or ``roks``, so the closed-shell and
  open-shell models of one functional have the same ``minimal@1`` hash;
- the method and basis names are normalized (``M06-2X``, ``def2-TZVP``);
- keywords that change the result are kept, lower-cased (``empiricaldispersion``,
  ``integral`` unless it is Gaussian 16's default UltraFine, ``scf`` convergence,
  ``stable``);
- optimizer and job keywords (``opt``, ``freq``, ``geom``, ...) are dropped, and so
  are keywords that only steer convergence;
- every other token is returned as unclassified, for the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ModelSpec", "model_from_routes"]

# Archive and route spellings -> the names we store.
_METHODS = {"m062x": "M06-2X", "b3lyp": "B3LYP"}
_BASES = {"def2tzvp": "def2-TZVP", "tzvp": "TZVP"}
_REFERENCES = {"RO": "roks", "R": "rks", "U": "uks"}
_PREFIXES = ("RO", "R", "U")  # RO before R

# Route keywords with no effect on the model: the job, the optimizer, and the printing.
_JOB = frozenset(
    {
        "opt", "optimize", "freq", "frequency", "geom", "nosymm", "guess", "scrf",
        "genchk", "chkbasis", "p", "n", "t", "output", "pop", "density", "symmetry",
        "test", "units", "iop",
    }
)  # fmt: skip
# Keys that are aliases of one another.
_ALIASES = {"int": "integral"}
# SCF options that change the answer; the others (xqc, maxconventionalcycles, ...) only
# steer convergence.
_SCF_RESULT = re.compile(r"^(conver=\d+|tight|vtight|verytight|sleazy|fermi|nofermi)$")
_SCF_STEERING = re.compile(
    r"^(xqc|qc|yqc|maxcycle=\d+|maxconventionalcycles=\d+|novaracc|noincfock|"
    r"incfock|direct|conventional|damp|diis|nodiis|ndamp|vshift=\d+)$"
)
# Keywords kept as they are written (lower-cased).
_KEPT = frozenset({"empiricaldispersion", "integral", "stable", "scf"})
_PRINT_LEVELS = frozenset({"n", "p", "t", "l", "s"})  # ``#N``, ``#P``, ...


@dataclass(frozen=True)
class ModelSpec:
    """What a log says about its model (a ``ModelRow`` without a database)."""

    program: str
    program_version: str | None
    method: str
    basis: str | None
    keywords: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str | None, str, str | None, str]:
        """A hashable identity of the model, for the importer's cache."""
        items = ";".join(f"{k}={v}" for k, v in sorted(self.keywords.items()))
        return (self.program, self.program_version, self.method, self.basis, items)


def split_route(route: str) -> list[str]:
    """Split a route into top-level tokens, keeping ``(a, b)`` groups whole.

    Gaussian also accepts spaces and commas inside parentheses, and ``keyword=(...)`` or
    ``keyword(...)``, so this splits at whitespace outside parentheses only.
    """
    tokens: list[str] = []
    depth = 0
    current = ""
    for char in route.strip().lstrip("#"):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char.isspace() and depth <= 0:
            if current:
                tokens.append(current)
            current = ""
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


def _split_keyword(token: str) -> tuple[str, str]:
    """Split ``Int=UltraFine`` or ``scf=(xqc,tight)`` into (name, value)."""
    match = re.match(r"^([A-Za-z0-9_-]+)\s*(?:=|(?=\())\s*(.*)$", token)
    if match is None:
        return token.lower(), ""
    return match.group(1).lower(), match.group(2).strip().lower()


def _split_reference(method: str) -> tuple[str | None, str, bool]:
    """Split an archive method into (reference, bare method, recognized).

    ``UM062X`` gives ``("uks", "M062X", True)``. A method of an unknown name (no known
    bare name after a prefix, or without one) is returned whole and unrecognized.
    """
    for prefix in _PREFIXES:
        if method.startswith(prefix) and method[len(prefix) :].lower() in _METHODS:
            return _REFERENCES[prefix], method[len(prefix) :], True
    if method.lower() in _METHODS:
        return None, method, True
    return None, method, False


def _scf_options(value: str) -> tuple[list[str], list[str]]:
    """Split an ``scf=(...)`` value into (result-changing, unclassified) options."""
    inner = value.strip("()")
    kept: list[str] = []
    unknown: list[str] = []
    for option in (o.strip() for o in re.split(r"[,\s]+", inner) if o.strip()):
        if _SCF_RESULT.match(option):
            kept.append(option)
        elif not _SCF_STEERING.match(option):
            unknown.append(option)
    return kept, unknown


def _classify(token: str, keywords: dict[str, str]) -> list[str]:
    """Add a token's model keyword to ``keywords``; return what is unclassified."""
    name, value = _split_keyword(token)
    if "/" in token or name in _JOB or (name in _PRINT_LEVELS and not value):
        return []  # the level of theory comes from the archive; the rest is the job
    if name == "scf":
        kept, unknown = _scf_options(value)
        if kept:
            keywords["scf"] = ",".join(kept)
        return [f"scf:{option}" for option in unknown]
    if name in _ALIASES or name in _KEPT:
        keywords[_ALIASES.get(name, name)] = value.strip("()")
        return []
    return [token.lower()]


def model_from_routes(
    *,
    program_version: str | None,
    method: str,
    basis: str,
    routes: list[str],
) -> tuple[ModelSpec, list[str]]:
    """Build the model of a Gaussian log.

    Args:
        program_version: As the archive spells it (``G16RevB.01``).
        method: The Freq step's method with its reference (``UM062X``).
        basis: The archive's basis name (``def2TZVP``).
        routes: Every route of the log, in order.

    Returns:
        The model and the route tokens that were not classified.
    """
    reference, bare, known = _split_reference(method)
    keywords: dict[str, str] = {}
    if reference is not None:
        keywords["reference"] = reference
    unclassified: list[str] = [] if known else [f"method:{method}"]
    for route in routes:
        for token in split_route(route):
            unclassified.extend(_classify(token, keywords))
    # UltraFine is Gaussian 16's default grid, so writing it changes nothing (the RGD1
    # minima spell it out, the TSs do not).
    is_g16 = program_version is not None and program_version.startswith("G16")
    if is_g16 and keywords.get("integral") == "ultrafine":
        del keywords["integral"]
    name = _METHODS.get(bare.lower(), bare)
    basis_name = _BASES.get(basis.lower(), basis)
    spec = ModelSpec(
        program="gaussian",
        program_version=program_version,
        method=name,
        basis=basis_name,
        keywords=keywords,
    )
    return spec, unclassified
