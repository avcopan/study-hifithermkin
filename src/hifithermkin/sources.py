"""Read the RGD1 zip and the stjohn tarball without unpacking them."""

from __future__ import annotations

import ast
import csv
import gzip
import io
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.rdBase import DisableLog

from .parse import SUBSET_ELEMENTS, SUBSET_HEAVY_ATOMS, Skip, parse_log

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .parse import ParsedLog

__all__ = [
    "MAX_SUBSET_ATOMS",
    "data_dir",
    "read_rgd1",
    "read_stjohn",
    "rgd1_mapping",
    "rgd1_members",
    "rgd1_subset_species",
    "rgd1_zip",
    "rgd1_zip_read",
    "stjohn_members",
    "stjohn_tar",
]

MAX_SUBSET_ATOMS = 3 * SUBSET_HEAVY_ATOMS + 2  # C4H10, as many H as heavy atoms allow
RP_PREFIX = "raw_output/RP/"
TS_PREFIX = "raw_output/final-TS/"


def data_dir() -> Path:
    """The datasets' folder: ``$DATA_DIR`` or ``~/code/am/data``."""
    return Path(os.environ.get("DATA_DIR", Path.home() / "code/am/data"))


def rgd1_zip() -> Path:
    """The RGD1 raw-output zip."""
    return data_dir() / "rgd1/RGD1_rawoutput.zip"


def stjohn_tar() -> Path:
    """The stjohn-radicals log tarball."""
    return data_dir() / "stjohn-radicals/20200411_scidata_logs.tar.gz"


_zips: dict[str, zipfile.ZipFile] = {}  # one handle per worker process


def _zip(path: Path) -> zipfile.ZipFile:
    handle = _zips.get(str(path))
    if handle is None:
        handle = _zips[str(path)] = zipfile.ZipFile(path)
    return handle


def rgd1_zip_read(member: str) -> bytes:
    """The bytes of one member of the RGD1 zip."""
    return _zip(rgd1_zip()).read(member)


def rgd1_mapping(path: Path | None = None) -> dict[str, tuple[list[str], list[str]]]:
    """Each reaction channel (``MR_X_Y``) with its reactant and product species."""
    text = _zip(path or rgd1_zip()).read("raw_output/RP_mapping.csv").decode()
    return {
        row["channel"]: (ast.literal_eval(row["Rmols"]), ast.literal_eval(row["Pmols"]))
        for row in csv.DictReader(io.StringIO(text))
    }


def rgd1_subset_species(path: Path | None = None) -> set[str]:
    """Species (``molN``) whose source SMILES is C/H/O with at most four heavy atoms."""
    DisableLog("rdApp.*")
    text = _zip(path or rgd1_zip()).read("raw_output/RandP_smiles.txt").decode()
    species: set[str] = set()
    for line in text.splitlines()[1:]:
        smiles, _, mol = line.rpartition("   ")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        if set(symbols) <= SUBSET_ELEMENTS and len(symbols) <= SUBSET_HEAVY_ATOMS:
            species.add(mol)
    return species


def rgd1_members(
    mapping: dict[str, tuple[list[str], list[str]]],
    *,
    subset: set[str] | None = None,
    path: Path | None = None,
) -> Iterator[str]:
    """Raw-output members to import: species logs, then TS logs.

    With ``subset`` (the species to keep), only those species, and the TSs of reactions
    whose species are all in it.
    """
    names = set(_zip(path or rgd1_zip()).namelist())
    species = sorted(
        {
            m
            for r, p in mapping.values()
            for m in (*r, *p)
            if subset is None or m in subset
        }
    )
    for mol in species:
        member = f"{RP_PREFIX}{mol}.out"
        if member in names:
            yield member
    for channel, (reactants, products) in mapping.items():
        member = f"{TS_PREFIX}{channel}-TS.out"
        if member in names and (
            subset is None or all(m in subset for m in (*reactants, *products))
        ):
            yield member


def read_rgd1(member: str, *, subset: bool = False) -> ParsedLog | Skip:
    """Read one RGD1 member from the zip and parse it (runs in a worker)."""
    raw = _zip(rgd1_zip()).read(member)
    return parse_log("rgd1", rgd1_zip().name, member, raw, subset=subset)


def stjohn_members(
    done: set[str], limit: int | None = None
) -> Iterator[tuple[str, bytes]]:
    """Stream the tarball once: (member, gzipped bytes) of members not in ``done``."""
    count = 0
    with tarfile.open(stjohn_tar(), "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if limit is not None and count >= limit:
                return
            count += 1
            if member.name in done:
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                yield member.name, handle.read()


def read_stjohn(
    member: str, packed: bytes, *, subset: bool = False
) -> ParsedLog | Skip:
    """Gunzip and parse one stjohn member (runs in a worker)."""
    raw = gzip.decompress(packed)
    if subset:
        match = re.search(rb"^ NAtoms=\s+(\d+)", raw, re.MULTILINE)
        if match and int(match.group(1)) > MAX_SUBSET_ATOMS:  # cheap: no decoding
            return Skip("stjohn", member, "outside_subset")
    return parse_log("stjohn", stjohn_tar().name, member, raw, subset=subset)
