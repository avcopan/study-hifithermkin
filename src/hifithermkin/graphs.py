"""Graphs for the stationary points, so that AutoStorage can store graph identities.

Minima pass nothing: AutoStorage perceives their graph from the geometry. The graphs
built here are for the points that cannot be perceived that way:

- a transition state of RGD1 gets its reaction graph from the atom-mapped SMILES in
  ``RGD1CHNO_AMsmiles.csv``, provided its log geometry has the atom order of the h5
  group ``MR_X_Y`` (map number k is atom k - 1 of the h5);
- a "minimum" at order > 0 (a false minimum) gets the graph of its geometry, in its own
  atom order.

Autostorage turns these graphs into its graph identities: ``smg_conn@1`` (stereo-free,
minima and TSs), ``smg_low_t@1`` (minima) and the interim ``protomol_graph@0`` (saddle
points), the TS ones in a canonical direction. pm P29 (atom-mapped reaction SMILES to a
reaction graph) formalizes this glue later.

A TS graph is also checked against the h5 geometries, and a disagreement is a *flag* on
the point (it is still imported): :func:`ts_flags`. A TS whose graph cannot be built for
lack of a source, or because its log has another atom order, is imported without one,
with the status in `NO_GRAPH` and the reason in ``graph_detail``.
"""

from __future__ import annotations

import csv
import dataclasses
from typing import TYPE_CHECKING, Any

import autogeom
import h5py
import numpy as np
import protomol
from rdkit import Chem
from rdkit.rdBase import DisableLog

from . import sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autogeom import Geometry

    from .parse import ParsedLog

__all__ = [
    "NO_GRAPH",
    "attach_graph",
    "mapped_reaction_graph",
    "ts_flags",
    "tsg_mismatch",
]

MAX_ALIGNED_DISTANCE = 0.1  # Å, between the log's TS geometry and the h5's TSG
NO_GRAPH = frozenset({"ts_no_source", "ts_atom_order_mismatch"})

_smiles: dict[str, tuple[str, str]] | None = None  # per worker, loaded on first use
_h5: h5py.File | None = None


def _mapped_graph(smiles: str, spin: int | None = None) -> protomol.StereoMolGraph:
    """The graph of an atom-mapped SMILES, atoms in map-number order."""
    DisableLog("rdApp.*")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    molecule = Chem.MolFromSmiles(smiles, params)
    graph = protomol.from_smiles(smiles, spin=spin)
    if molecule is None or molecule.GetNumAtoms() != len(graph.symbols):
        msg = "RDKit and protomol disagree on the atoms of the SMILES"
        raise ValueError(msg)
    numbers = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if sorted(numbers) != list(range(1, len(numbers) + 1)):
        msg = "the map numbers are not 1..n"
        raise ValueError(msg)
    # atom k of the result is the atom with map number k + 1
    atom_map = [numbers.index(k + 1) for k in range(len(numbers))]
    return protomol.relabel(graph, atom_map)


def mapped_reaction_graph(
    reactant: str, product: str, spin: int | None = None
) -> protomol.StereoReactionGraph:
    """The reaction graph of an atom-mapped reaction SMILES, in map-number order.

    ``spin`` overrides the spin that the SMILES implies on both sides (RGD1 is all
    closed shell: its singlet carbenes are written as radicals, which would read as
    triplets).
    """
    return protomol.reaction_graph(
        _mapped_graph(reactant, spin), _mapped_graph(product, spin)
    )


def _aligned_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The largest atom distance (Å) between two structures after a rigid alignment."""
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return float(np.max(np.linalg.norm(a @ rotation - b, axis=1)))


def tsg_mismatch(log: ParsedLog, tsg: np.ndarray, elements: np.ndarray) -> str | None:
    """How the log's geometry differs from the atom order of the h5's ``TSG``, if so."""
    symbols = [autogeom.element.from_key(int(z)).symbol for z in elements]
    if symbols != log.symbols:
        return "the elements are not those of the h5, in its order"
    distance = _aligned_distance(log.coordinates, tsg)
    if distance < MAX_ALIGNED_DISTANCE:
        return None
    return f"an atom is {distance:.2f} Å from the h5's TSG after alignment"


def _channels() -> dict[str, tuple[str, str]]:
    global _smiles  # noqa: PLW0603
    if _smiles is None:
        path = sources.data_dir() / "rgd1/RGD1CHNO_AMsmiles.csv"
        with path.open(newline="") as handle:
            _smiles = {
                row["reaction"]: (row["reactant"], row["product"])
                for row in csv.DictReader(handle)
            }
    return _smiles


def _h5_file() -> h5py.File:
    global _h5  # noqa: PLW0603
    if _h5 is None:
        _h5 = h5py.File(sources.data_dir() / "rgd1/RGD1_allrxns.h5")
    return _h5


def _geometry(log: ParsedLog) -> autogeom.Geometry:
    return autogeom.Geometry(
        symbols=log.symbols,
        coordinates=log.coordinates,
        charge=log.charge,
        spin=log.spin,
    )


def _bonds(symbols: list[str], bonds: Any) -> str:  # noqa: ANN401
    """Bonds as ``C2-N4`` (atom numbers from 1, the map numbers of the SMILES)."""
    return (
        " ".join(f"{symbols[i]}{i + 1}-{symbols[j]}{j + 1}" for i, j in sorted(bonds))
        or "none"
    )


def ts_flags(
    graph: protomol.StereoReactionGraph,
    symbols: list[str],
    group: Mapping[str, Any],
    ts_geometry: Geometry,
) -> list[tuple[str, str]]:
    """Where a TS graph disagrees with the geometries, as ``(flag, detail)`` pairs.

    - ``bond_changes``: the bonds it breaks and forms differ from those between the
      graphs perceived from the h5's reactant and product geometries (``RG``, ``PG``).
      The detail names the bonds that only one of them has.
    - ``unchanged_bond_missing``: a bond that neither breaks nor forms is missing from
      the graph perceived from the TS geometry.
    """
    flags: list[tuple[str, str]] = []
    ends = [
        protomol.from_geometry(
            autogeom.Geometry(
                symbols=symbols, coordinates=group[key][()], charge=0, spin=0
            )
        )
        for key in ("RG", "PG")
    ]
    perceived = protomol.reaction_graph(*ends)
    ours = {"breaking": set(graph.breaking), "forming": set(graph.forming)}
    theirs = {"breaking": set(perceived.breaking), "forming": set(perceived.forming)}
    if ours != theirs:
        parts = [
            f"{kind}: SMILES only {_bonds(symbols, ours[kind] - theirs[kind])}, "
            f"geometry only {_bonds(symbols, theirs[kind] - ours[kind])}"
            for kind in ours
            if ours[kind] != theirs[kind]
        ]
        flags.append(("bond_changes", "; ".join(parts)))
    unchanged = set(graph.bonds) - ours["breaking"] - ours["forming"]
    missing = unchanged - set(protomol.from_geometry(ts_geometry).bonds)
    if missing:
        flags.append(("unchanged_bond_missing", _bonds(symbols, missing)))
    return flags


def _ts_graph(
    log: ParsedLog,
) -> tuple[Any, str, tuple[tuple[str, str], ...], str]:
    """The graph, status, flags and detail (why there is no graph) of a TS log."""
    channel = log.member.rsplit("/", 1)[-1].removesuffix("-TS.out")
    smiles = _channels().get(channel)
    group = _h5_file().get(channel)
    if smiles is None or group is None:
        missing = [
            source
            for source, absent in (
                ("RGD1CHNO_AMsmiles.csv", smiles is None),
                ("RGD1_allrxns.h5", group is None),
            )
            if absent
        ]
        return None, "ts_no_source", (), f"not in {' or '.join(missing)}"
    mismatch = tsg_mismatch(log, group["TSG"][()], group["elements"][()])
    if mismatch is not None:
        return None, "ts_atom_order_mismatch", (), mismatch
    graph = mapped_reaction_graph(*smiles, spin=0)
    try:
        flags = tuple(ts_flags(graph, log.symbols, group, _geometry(log)))
    except Exception as exc:  # noqa: BLE001 - the graph is kept; the check is not
        flags = (("flag_check_error", type(exc).__name__),)
    return graph, "ts_from_mapped_smiles", flags, ""


def attach_graph(log: ParsedLog) -> ParsedLog:
    """The log with the graph to pass to its point, and a status.

    Minima get no graph (it is perceived). Failures give no graph and a
    ``graph_error:<type>`` status instead of an exception.
    """
    if log.nimag == 0:
        return dataclasses.replace(log, graph_status="perceived")
    flags: tuple[tuple[str, str], ...] = ()
    detail = ""
    try:
        if log.member.endswith("-TS.out"):
            graph, status, flags, detail = _ts_graph(log)
        else:
            graph, status = protomol.from_geometry(_geometry(log)), "false_minimum"
    except Exception as exc:  # noqa: BLE001 - counted, and the point is stored without
        graph, status = None, f"graph_error:{type(exc).__name__}"
    return dataclasses.replace(
        log,
        graph=graph,
        graph_status=status,
        graph_flags=flags,
        graph_detail=detail,
    )
