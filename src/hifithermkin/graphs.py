"""Graphs for the stationary points, so that AutoStorage can store graph identities.

Minima pass nothing: AutoStorage perceives their graph from the geometry. The graphs
built here are for the points that cannot be perceived that way:

- a transition state of RGD1 gets its reaction graph from the atom-mapped SMILES in
  ``RGD1CHNO_AMsmiles.csv``, provided its log geometry has the atom order of the h5
  group ``MR_X_Y`` (map number k is atom k - 1 of the h5);
- a "minimum" at order > 0 (a false minimum) gets the graph of its geometry, in its own
  atom order.

Autostorage turns these graphs into its graph identities (its built-in
``protomol_graph@0`` today); pm P29 (atom-mapped reaction SMILES to a reaction graph)
formalizes this glue later.
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

    from .parse import ParsedLog

__all__ = [
    "attach_graph",
    "mapped_reaction_graph",
    "tsg_matches",
]

MAX_ALIGNED_DISTANCE = 0.1  # Å, between the log's TS geometry and the h5's TSG

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


def tsg_matches(log: ParsedLog, tsg: np.ndarray, elements: np.ndarray) -> bool:
    """Whether the log's geometry has the atom order of the h5's ``TSG``."""
    symbols = [autogeom.element.from_key(int(z)).symbol for z in elements]
    if symbols != log.symbols:
        return False
    return _aligned_distance(log.coordinates, tsg) < MAX_ALIGNED_DISTANCE


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


def _ts_graph(log: ParsedLog) -> tuple[Any, str]:
    channel = log.member.rsplit("/", 1)[-1].removesuffix("-TS.out")
    smiles = _channels().get(channel)
    group = _h5_file().get(channel)
    if smiles is None or group is None:
        return None, "ts_no_source"
    if not tsg_matches(log, group["TSG"][()], group["elements"][()]):
        return None, "ts_atom_order_mismatch"
    return mapped_reaction_graph(*smiles, spin=0), "ts_from_mapped_smiles"


def attach_graph(log: ParsedLog) -> ParsedLog:
    """The log with the graph to pass to its point, and a status.

    Minima get no graph (it is perceived). Failures give no graph and a
    ``graph_error:<type>`` status instead of an exception.
    """
    if log.nimag == 0:
        return dataclasses.replace(log, graph_status="perceived")
    try:
        if log.member.endswith("-TS.out"):
            graph, status = _ts_graph(log)
        else:
            graph, status = protomol.from_geometry(_geometry(log)), "false_minimum"
    except Exception as exc:  # noqa: BLE001 - counted, and the point is stored without
        graph, status = None, f"graph_error:{type(exc).__name__}"
    return dataclasses.replace(log, graph=graph, graph_status=status)
