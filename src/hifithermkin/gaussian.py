r"""Gaussian 16: decode the output of an Opt+Freq job (no encoder).

This codec, an external qccodec plugin, reads finished Gaussian 16 logs; it cannot
write an input or run the program. It is for importing existing calculations, such as
the Opt+Freq logs of the RGD1 and St. John radical datasets. Use it through
``qccodec.get_codec("gaussian").parse(CalcType.optimization, stdout=text)`` (or
``CalcType.transition_state`` for an ``Opt=TS`` job).

Everything but the frequencies comes from the *archive block* of the last job step,
the ``1\1\GINC-...\@`` summary Gaussian prints at the end of each step, because it
carries full precision values in one machine-readable place. The last archive block
must be a frequency job, since it holds the Hessian. The result is an
``OptimizationData`` with one frame, the structure the Freq job ran at. The
optimization's earlier steps are not parsed yet, so read the result through
``final_structure``, ``final_energy`` or ``frames[-1]``, which stay the same once they
are:

- ``frames[-1].structure``: the archive geometry (bohr, like all qcdata structures; the
  archive's angstrom values are converted), symbols, charge and multiplicity. It is in
  the frame of the archive, which is the frame of the Hessian; with ``nosymm`` it is
  the input frame, otherwise Gaussian may have reoriented it.
- ``frames[-1].results.energy``: the SCF energy (Hartree) of the Freq job, from its last
  ``SCF Done`` line (ten decimals; the archive's ``HF=`` has seven and is used when
  the log has no such line, and checked against it otherwise).
- ``frames[-1].results.hessian``: the Cartesian Hessian (Hartree/bohr^2), the archive's
  lower triangle returned as the full symmetric ``3N x 3N`` matrix.
- ``extras["gaussian"]``: ``nimag``, ``frequencies`` (cm^-1 from the log body,
  negative for imaginary modes), ``method`` and ``basis`` as the archive names them
  (``UM062X``, ``def2TZVP``), ``jobtype``, ``route`` (of the Freq step, which with
  ``Geom=AllCheck`` carries no keywords of the original route), ``routes`` (every
  route of the log, in order, which have the keywords: ``EmpiricalDispersion``,
  ``Int=UltraFine``, ``stable=opt``, ...), ``zero_point_energy`` and
  ``thermal_correction_energy`` (archive, Hartree) and, when printed, the thermal
  corrections to enthalpy and Gibbs energy, ``temperature`` and ``pressure``.
- ``provenance.program_version``: as the archive spells it, e.g. ``G16RevB.01``.

The archive's other contents (dipole, polarizability, gradient, ...) are not parsed.
No input field carries atom indices.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
from qccodec import Codec
from qccodec.exceptions import MatchNotFoundError, ParserError
from qcdata import CalcType, Frame, Provenance, SinglePointData, Structure
from qcref import constants

codec = Codec("gaussian", calctypes={CalcType.optimization, CalcType.transition_state})

# An archive block is a run of lines that start with a space (it is followed by a blank
# line). Its ``\\@`` terminator can be split across two lines, so it is not searched
# for.
_ARCHIVE = re.compile(r"^ 1\\1\\.*(?:\n .*)*", re.MULTILINE)
_GEOMETRY = re.compile(r"^\\?(-?\d+),(\d+)\\")
_FLOAT = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"
_NOT_FOUND_CONTEXT = 300  # characters of the log kept in a MatchNotFoundError
_MIN_HEADER_FIELDS = 6  # in an archive header: ..., jobtype, method, basis
_FROZEN_ATOM_FIELDS = 5  # symbol, freeze code, x, y, z
_ATOM_FIELDS = 4  # symbol, x, y, z
_HF_TOLERANCE = 1e-6  # Hartree, between the SCF Done line and the archive's HF=


def re_search(regex: str, contents: str, flags: int = 0) -> re.Match[str]:
    """Return the first match of ``regex`` in ``contents``.

    Raises:
        MatchNotFoundError: No match is found.
    """
    match = re.search(regex, contents, flags)
    if not match:
        raise MatchNotFoundError(regex, contents)
    return match


def _not_found(regex: str, stdout: str) -> MatchNotFoundError:
    """A match error that keeps only the end of a (large) log."""
    return MatchNotFoundError(regex, stdout[-_NOT_FOUND_CONTEXT:])


def _unwrap(block: str) -> str:
    """Join the 70-column lines of an archive block, dropping each first space."""
    return "".join(line[1:] for line in block.split("\n"))


@lru_cache(maxsize=4)
def _final_archive(stdout: str) -> tuple[dict[str, Any], int]:
    """Split the last archive block into its sections.

    Returns:
        The sections of the block and the offset in ``stdout`` where it ends.

    Raises:
        MatchNotFoundError: There is no archive block, or it has no Hessian.
        ParserError: The block is malformed.
    """
    blocks = list(_ARCHIVE.finditer(stdout))
    if not blocks:
        raise _not_found(_ARCHIVE.pattern, stdout)
    block = blocks[-1]
    parts = _unwrap(block.group()).split("\\\\")
    header = parts[0].split("\\")
    start = next(
        (k for k in range(2, len(parts)) if _GEOMETRY.match(parts[k])),
        None,
    )
    if start is None or len(header) < _MIN_HEADER_FIELDS:
        msg = "The last archive block has no charge, multiplicity and atoms."
        raise ParserError(msg)
    geometry = _GEOMETRY.match(parts[start])
    if geometry is None:  # unreachable: ``start`` is the first part that matches
        msg = "The last archive block has no charge and multiplicity."
        raise ParserError(msg)
    charge, multiplicity = geometry.groups()
    atoms = parts[start].split("\\")[1:]
    properties: dict[str, str] = {}
    for item in parts[start + 1].split("\\"):
        key, _, value = item.partition("=")
        properties[key] = value
    return (
        {
            "jobtype": header[3],
            "method": header[4],
            "basis": header[5],
            "route": parts[1],
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "atoms": atoms,
            "properties": properties,
            "hessian": parts[start + 2] if len(parts) > start + 3 else None,
        },
        block.end(),
    )


def _final_step(stdout: str) -> str:
    """The text of the job step that wrote the last archive block."""
    end = _final_archive(stdout)[1]
    start = stdout.rfind("\n Link1:", 0, end)
    return stdout[max(start, 0) : end]


def _structure(stdout: str) -> Structure:
    """The archive's atoms, in bohr."""
    archive = _final_archive(stdout)[0]
    symbols: list[str] = []
    geometry: list[list[float]] = []
    for atom in archive["atoms"]:
        fields = atom.split(",")
        if (
            len(fields) == _FROZEN_ATOM_FIELDS
        ):  # a freeze code between the symbol and the coordinates
            del fields[1]
        if len(fields) != _ATOM_FIELDS or not re.fullmatch(r"[A-Z][a-z]?", fields[0]):
            msg = f"Unsupported archive atom: {atom!r}."
            raise ParserError(msg)
        symbols.append(fields[0])
        geometry.append([float(v) / constants.BOHR_TO_ANGSTROM for v in fields[1:]])
    return Structure(
        symbols=symbols,
        geometry=np.array(geometry),
        charge=archive["charge"],
        multiplicity=archive["multiplicity"],
    )


def _hessian(stdout: str) -> np.ndarray:
    """The full symmetric Hessian from the archive's lower triangle."""
    archive = _final_archive(stdout)[0]
    natoms = len(archive["atoms"])
    n = 3 * natoms
    text = archive["hessian"]
    try:
        values = [float(v) for v in text.split(",")] if text else []
    except ValueError:  # an opt-only block: the gradient, then the terminator
        values = []
    if len(values) != n * (n + 1) // 2:
        msg = "a lower-triangle Hessian in the last archive block"
        raise _not_found(msg, stdout)
    hessian = np.zeros((n, n))
    hessian[np.tril_indices(n)] = values
    return hessian + hessian.T - np.diag(np.diag(hessian))


def _energy(stdout: str) -> float:
    """The SCF energy of the Freq job; the archive's ``HF=`` backs it up."""
    archive = _final_archive(stdout)[0]
    scf = re.findall(rf"SCF Done:\s+E\([^)]*\)\s*=\s*({_FLOAT})", _final_step(stdout))
    hf = archive["properties"].get("HF")
    if not scf:
        if hf is None:
            msg = "SCF Done or HF= in the last archive block"
            raise _not_found(msg, stdout)
        return float(hf)
    energy = float(scf[-1])
    if hf is not None and abs(energy - float(hf)) > _HF_TOLERANCE:
        msg = f"SCF Done energy {energy} differs from archive HF={hf}."
        raise ParserError(msg)
    return energy


@codec.parser(target="frames")
def parse_frames(stdout: str) -> list[Frame]:
    """The one frame of the Freq job: archive geometry, SCF energy and Hessian."""
    structure = _structure(stdout)
    results = SinglePointData(
        provenance=Provenance(
            program="gaussian", program_version=parse_version(stdout)
        ),
        energy=_energy(stdout),
        hessian=_hessian(stdout),
        calcinfo_natoms=len(structure.symbols),
    )
    return [Frame(success=True, structure=structure, results=results)]


@codec.parser(target="calcinfo_natoms")
def parse_natoms(stdout: str) -> int:
    """The number of atoms of the last archive block."""
    return len(_final_archive(stdout)[0]["atoms"])


def parse_routes(stdout: str) -> list[str]:
    """Every route section of the log (``# B3LYP/TZVP ... Opt``), in order."""
    return [
        _unwrap(match.group(1)).strip()
        for match in re.finditer(
            r"^ -{20,}\n( #.*?)\n -{20,}\n", stdout, re.MULTILINE | re.DOTALL
        )
    ]


def parse_frequencies(stdout: str) -> list[float]:
    """The frequencies (cm^-1, negative if imaginary) printed by the Freq step."""
    lines = re.findall(r"^ Frequencies --(.*)$", _final_step(stdout), re.MULTILINE)
    return [float(v) for line in lines for v in re.findall(_FLOAT, line)]


@codec.parser(target=("extras", "gaussian"))
def parse_extras(stdout: str) -> dict[str, Any]:
    """Level of theory, route, frequencies, NImag and thermal corrections."""
    archive = _final_archive(stdout)[0]
    properties = archive["properties"]
    step = _final_step(stdout)
    extras: dict[str, Any] = {
        "jobtype": archive["jobtype"],
        "method": archive["method"],
        "basis": archive["basis"],
        "route": archive["route"],
        "routes": parse_routes(stdout),
        "nimag": int(re_search(r"\d+", properties.get("NImag", "")).group()),
        "frequencies": parse_frequencies(stdout),
    }
    if "ZeroPoint" in properties:
        extras["zero_point_energy"] = float(properties["ZeroPoint"])
    if "Thermal" in properties:
        extras["thermal_correction_energy"] = float(properties["Thermal"])
    for key, label in {
        "thermal_correction_enthalpy": "Thermal correction to Enthalpy",
        "thermal_correction_gibbs": "Thermal correction to Gibbs Free Energy",
    }.items():
        match = re.search(rf"^ {label}=\s*({_FLOAT})", step, re.MULTILINE)
        if match:
            extras[key] = float(match.group(1))
    conditions = re.search(
        rf"^ Temperature\s+({_FLOAT}) Kelvin\.\s+Pressure\s+({_FLOAT}) Atm\.",
        step,
        re.MULTILINE,
    )
    if conditions:
        extras["temperature"] = float(conditions.group(1))
        extras["pressure"] = float(conditions.group(2))
    return extras


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """The version as the archive spells it, ``G16RevB.01``."""
    version = _final_archive(stdout)[0]["properties"].get("Version", "")
    return re_search(r"(G\d+Rev\w+\.\d+)", version).group(1)
