"""Parse one Gaussian log into a plain record, or the reason to skip it.

This runs in worker processes: its results are picklable and hold no database objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from qccodec import get_codec
from qcdata import CalcType, OptimizationData
from qcref import constants

from . import gaussian  # noqa: F401  # registers the codec when run without its entry point
from .routes import ModelSpec, model_from_routes

__all__ = ["MAX_FREQUENCY", "ParsedLog", "Skip", "in_subset", "parse_log"]

MAX_FREQUENCY = 5000.0  # cm^-1; a larger |frequency| means a broken Hessian
SUBSET_ELEMENTS = frozenset({"C", "H", "O"})
SUBSET_HEAVY_ATOMS = 4


@dataclass(frozen=True)
class Skip:
    """A log that is not imported, and why."""

    dataset: str
    member: str
    reason: str  # parse_error, broken_hessian, outside_subset
    detail: str = ""


@dataclass(frozen=True)
class ParsedLog:
    """What the importer needs from one log."""

    dataset: str
    member: str
    archive: str
    md5: str
    calc_type: str  # the decoder's calctype value
    symbols: list[str]
    coordinates: np.ndarray  # Å
    charge: int
    multiplicity: int
    energy: float  # Hartree
    hessian: np.ndarray  # Hartree/bohr², (3N, 3N)
    model: ModelSpec
    unclassified: list[str]
    nimag: int
    routes: list[str]  # every route of the log, in order
    output: dict[str, Any]  # NImag, frequencies, ZPE, thermal corrections, T, P
    graph: Any = None  # a protomol graph for the stationary point (see graphs.py)
    graph_status: str = ""  # how it was made, or why not; counted by the store

    @property
    def spin(self) -> int:
        """The number of unpaired electrons."""
        return self.multiplicity - 1


def in_subset(symbols: list[str]) -> bool:
    """C/H/O with at most four heavy atoms."""
    heavy = sum(1 for s in symbols if s != "H")
    return set(symbols) <= SUBSET_ELEMENTS and heavy <= SUBSET_HEAVY_ATOMS


def _decode(kind: CalcType, raw: bytes) -> OptimizationData:
    """The decoder's result; raises on anything it cannot read."""
    data = get_codec("gaussian").parse(kind, stdout=raw.decode(errors="replace"))
    if not isinstance(data, OptimizationData):  # the codec's only result type
        raise TypeError(type(data).__name__)
    return data


def parse_log(
    dataset: str,
    archive: str,
    member: str,
    raw: bytes,
    *,
    subset: bool = False,
) -> ParsedLog | Skip:
    """Decode one log.

    Args:
        dataset: ``rgd1`` or ``stjohn``.
        archive: File name of the archive that holds the member.
        member: Path of the member inside the archive.
        raw: The log's bytes (already gunzipped for stjohn).
        subset: Skip logs outside the C/H/O, four-heavy-atom subset.
    """
    kind = (
        CalcType.transition_state
        if member.endswith("-TS.out")
        else CalcType.optimization
    )
    try:
        data = _decode(kind, raw)
    except Exception as exc:  # noqa: BLE001 - every failure is a row of the failures CSV
        message = " ".join(f"{type(exc).__name__}: {exc}".split())
        return Skip(dataset, member, "parse_error", message[:300])
    frame = data.frames[0]
    structure = frame.structure
    energy = float(frame.results.energy or 0.0)
    hessian = np.asarray(frame.results.hessian)
    extras = dict(data.extras["gaussian"])
    symbols = list(structure.symbols)
    if subset and not in_subset(symbols):
        return Skip(dataset, member, "outside_subset")
    frequencies = extras["frequencies"]
    if frequencies and max(abs(f) for f in frequencies) > MAX_FREQUENCY:
        worst = max(frequencies, key=abs)
        return Skip(dataset, member, "broken_hessian", f"{worst:.0f} cm-1")
    model, unclassified = model_from_routes(
        program_version=data.provenance.program_version,
        method=extras["method"],
        basis=extras["basis"],
        routes=extras["routes"],
    )
    routes = list(extras["routes"])
    output = {
        key: value
        for key, value in extras.items()
        if key not in {"routes", "route", "method", "basis", "jobtype"}
    }
    output["program_version"] = data.provenance.program_version
    return ParsedLog(
        dataset=dataset,
        member=member,
        archive=archive,
        md5=hashlib.md5(raw, usedforsecurity=False).hexdigest(),
        calc_type=str(kind.value),
        symbols=symbols,
        coordinates=np.asarray(structure.geometry) * constants.BOHR_TO_ANGSTROM,
        charge=int(structure.charge),
        multiplicity=int(structure.multiplicity),
        energy=energy,
        hessian=hessian,
        model=model,
        unclassified=unclassified,
        nimag=int(extras["nimag"]),
        routes=routes,
        output=output,
    )
