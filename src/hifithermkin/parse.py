"""Parse one Gaussian log into a plain record, or the reason to skip it.

This runs in worker processes: its results are picklable and hold no database objects.
"""

from __future__ import annotations

import hashlib
import re
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
    echoed_input: list[dict[str, Any]]  # the input of each job, as the log echoes it
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


_JOB_START = re.compile(r"^ Entering Link 1 = ", re.MULTILINE)
_ROUTE = re.compile(r"^ -{20,}\n( #.*?)\n -{20,}\n", re.MULTILINE | re.DOTALL)
_TITLE = re.compile(r"^ -{5,}\n (.*)\n -{5,}\n Symbolic Z-matrix:", re.MULTILINE)
_CHARGE = re.compile(
    r"^ Charge =\s*(-?\d+) Multiplicity = (\d+)\n((?: [A-Z][a-z]?\s+[-+\d].*\n)*)",
    re.MULTILINE,
)


def _unwrap(block: str) -> str:
    """Join the 70-column lines of an echoed section, dropping each margin space."""
    return "".join(line[1:] for line in block.split("\n")).strip()


def echoed_inputs(text: str) -> list[dict[str, Any]]:
    """The input of each Gaussian job in a log, rebuilt from what the log echoes.

    Each entry has ``link0`` (the ``%`` lines), ``route``, ``title``, ``charge``,
    ``multiplicity``, ``geometry`` (the echoed atom lines, as written) and ``text``: the
    same in Gaussian input-file form (a job that reads its geometry from a checkpoint
    echoes none, so its geometry is empty).
    """
    starts = [m.start() for m in _JOB_START.finditer(text)] or [0]
    jobs: list[dict[str, Any]] = []
    for begin, end in zip(starts, [*starts[1:], len(text)], strict=True):
        job = text[begin:end]
        route = _ROUTE.search(job)
        link0 = (
            [
                line.strip()
                for line in job[: route.start()].splitlines()
                if line.startswith(" %")
            ]
            if route
            else []
        )
        title = _TITLE.search(job)
        molecule = _CHARGE.search(job)
        geometry = (
            [line.rstrip() for line in molecule.group(3).splitlines()]
            if molecule
            else []
        )
        route_text = _unwrap(route.group(1)) if route else ""
        parts = [*link0, route_text, "", title.group(1).strip() if title else "", ""]
        if molecule:
            parts.append(f"{molecule.group(1)} {molecule.group(2)}")
            parts += [" ".join(line.split()) for line in geometry]
        jobs.append(
            {
                "link0": link0,
                "route": route_text,
                "title": title.group(1).strip() if title else None,
                "charge": int(molecule.group(1)) if molecule else None,
                "multiplicity": int(molecule.group(2)) if molecule else None,
                "geometry": [" ".join(line.split()) for line in geometry],
                "text": "\n".join(parts) + "\n",
            }
        )
    return jobs


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
    echoed = echoed_inputs(raw.decode(errors="replace"))
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
        echoed_input=echoed,
        output=output,
    )
