"""Checks of the Gaussian codec that do not depend on its own recorded output.

The fixture cases under ``tests/data/gaussian`` (run by ``test_cases.py``) pin the whole
result; these tests compare it with the log text and with physics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from qccodec import get_codec
from qccodec.exceptions import EncoderError, MatchNotFoundError
from qccodec.testing import cases as load_cases
from qcdata import CalcType, OptimizationData
from qcref import constants

from hifithermkin import gaussian

DATA = Path(__file__).parent / "data"
CASES = sorted(p for p in (DATA / "gaussian").iterdir() if p.is_dir())
AMU = 1822.888486209  # electron masses per amu
HARTREE_TO_CM = 219474.6313632


def _decode(case: Path):  # noqa: ANN202
    """The decoded result and log text of a fixture case."""
    text = (case / "stdout").read_text()
    kind = CalcType.transition_state if "_ts_" in case.name else CalcType.optimization
    return get_codec("gaussian").parse(kind, stdout=text), text


def _frequencies_from_hessian(
    geometry: np.ndarray, hessian: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Vibrational wavenumbers (cm^-1, negative if imaginary) of a Cartesian Hessian.

    Mass-weights it and projects out translations and rotations about the center of
    mass.
    """
    n = len(masses)
    weights = np.repeat(masses, 3)
    mass_weighted = hessian / np.sqrt(np.outer(weights, weights))
    centered = geometry - (masses[:, None] * geometry).sum(0) / masses.sum()
    external = []
    for k in range(3):
        translation = np.zeros((n, 3))
        translation[:, k] = np.sqrt(masses)
        external.append(translation.ravel())
        axis = np.zeros(3)
        axis[k] = 1.0
        external.append((np.cross(axis, centered) * np.sqrt(masses)[:, None]).ravel())
    _, singular, vt = np.linalg.svd(np.array(external))
    internal = vt[int((singular > 1e-8).sum()) :].T
    eigenvalues = np.linalg.eigvalsh(internal.T @ mass_weighted @ internal)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) / AMU) * HARTREE_TO_CM


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_values_match_the_log_text(case: Path) -> None:
    """Energy, geometry, charge, NImag and thermal values agree with the printed log."""
    data, text = _decode(case)
    frame = data.frames[-1]
    extras = data.extras["gaussian"]
    step = text[text.rfind("\n Link1:") :]
    scf = re.findall(r"SCF Done:\s+E\(\S+\) =\s+(-?\d+\.\d+)", step)
    assert frame.results.energy == pytest.approx(float(scf[-1]), abs=1e-10)
    charge, multiplicity = re.findall(r"Charge =\s+(-?\d+) Multiplicity = (\d+)", step)[
        -1
    ]
    assert frame.structure.charge == int(charge)
    assert frame.structure.multiplicity == int(multiplicity)
    # The last "Input orientation" table is the geometry of the Freq job (all of the
    # fixtures run with its input orientation equal to the archive's frame, up to
    # the printed precision of 6 decimals).
    table = re.findall(
        r"Input orientation:.*?-{20,}\n.*?-{20,}\n(.*?)\n -{20,}", step, re.DOTALL
    )
    if table:
        rows = [line.split() for line in table[-1].splitlines()]
        numbers = [int(r[1]) for r in rows]
        assert numbers == [int(z) for z in frame.structure.atomic_numbers]
    assert extras["nimag"] == sum(f < 0 for f in extras["frequencies"])
    zpe = re.findall(r"Zero-point correction=\s+(-?\d+\.\d+)", step)[-1]
    assert extras["zero_point_energy"] == pytest.approx(float(zpe), abs=1e-6)
    gibbs = re.findall(r"Thermal correction to Gibbs Free Energy=\s+(-?\d+\.\d+)", step)
    assert extras["thermal_correction_gibbs"] == pytest.approx(
        float(gibbs[-1]), abs=1e-6
    )
    assert extras["temperature"] == pytest.approx(298.15)
    n = len(frame.structure.symbols)
    assert frame.results.hessian.shape == (3 * n, 3 * n)
    assert len(extras["frequencies"]) in (3 * n - 6, 3 * n - 5)
    assert data.calcinfo_natoms == n


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_hessian_reproduces_the_logged_frequencies(case: Path) -> None:
    """Mass-weighting and projecting the Hessian gives the logged frequencies."""
    data, text = _decode(case)
    frame = data.frames[-1]
    n = len(frame.structure.symbols)
    # Gaussian's isotope masses, printed per atom (the Freq job's are the last n).
    masses = np.array(
        [
            float(m)
            for m in re.findall(r"has atomic number\s+\d+ and mass\s+([\d.]+)", text)[
                -n:
            ]
        ]
    )
    hessian = frame.results.hessian
    assert np.array_equal(hessian, hessian.T)
    computed = np.sort(
        _frequencies_from_hessian(frame.structure.geometry, hessian, masses)
    )
    logged = np.sort(data.extras["gaussian"]["frequencies"])
    assert np.abs(computed[-len(logged) :] - logged).max() < 1.0


def test_geometry_is_the_archive_geometry_in_bohr() -> None:
    """NH3's first atom is the archive's N atom, converted from angstrom."""
    data, _ = _decode(next(c for c in CASES if c.name == "rgd1_nh3_minimum"))
    structure = data.frames[-1].structure
    assert structure.symbols == ["N", "H", "H", "H"]
    assert structure.geometry[0] == pytest.approx(
        np.array([1.0627657434, 0.0489738059, -0.0629474685])
        / constants.BOHR_TO_ANGSTROM
    )


def test_level_of_theory_and_routes() -> None:
    """The archive's method and basis, and the keywords of every route of the log."""
    data, _ = _decode(next(c for c in CASES if c.name == "stjohn_radical_stable_opt"))
    extras = data.extras["gaussian"]
    assert (extras["method"], extras["basis"], extras["jobtype"]) == (
        "UM062X",
        "def2TZVP",
        "Freq",
    )
    assert data.provenance.program_version == "G16RevB.01"
    assert extras["routes"][0].startswith("# stable=opt M062X/Def2TZVP")
    assert "nosymm" in extras["routes"][0]
    assert extras["routes"][-1].endswith("UM062X/def2TZVP Freq")


def test_transition_state_has_one_imaginary_mode() -> None:
    """NImag=1 for the TS, with the imaginary frequency negative."""
    data, _ = _decode(next(c for c in CASES if "_ts_" in c.name))
    extras = data.extras["gaussian"]
    assert extras["nimag"] == 1
    assert extras["frequencies"][0] < 0 < extras["frequencies"][1]


def test_log_without_an_archive_block_raises_a_short_error() -> None:
    """A truncated log fails with MatchNotFoundError, not a message holding the log."""
    text = (CASES[0] / "stdout").read_text()
    truncated = text[: text.find("\n 1\\1\\")]
    with pytest.raises(MatchNotFoundError) as caught:
        get_codec("gaussian").parse(CalcType.optimization, stdout=truncated)
    assert len(str(caught.value)) < 1000


def test_archive_without_a_hessian_is_not_a_result() -> None:
    """An optimization-only archive block (no Hessian) is refused."""
    text = (CASES[0] / "stdout").read_text()
    start = text.rfind("\n 1\\1\\")
    end = text.find("\\@", start)
    block = text[start:end]
    # Drop the Hessian section, which follows the properties' NImag.
    cut = block.replace("\n ", "").find("NImag=")
    flat = block.replace("\n ", "")
    stripped = flat[: flat.find("\\\\", cut)] + "\\\\@"
    with pytest.raises(MatchNotFoundError):
        get_codec("gaussian").parse(
            CalcType.optimization, stdout=text[:start] + "\n " + stripped
        )


def test_has_no_encoder() -> None:
    """The Gaussian codec only reads output."""
    input_data = load_cases(DATA)[0].input_data
    with pytest.raises(EncoderError):
        gaussian.codec.encode(input_data)


def test_terminator_split_across_lines_ends_the_block() -> None:
    r"""A ``\@`` broken over two lines must not merge a block into the next one.

    In RGD1 and stjohn logs an Opt-only block (no Hessian) can come first, with its
    terminator split like this; searching for ``\@`` ran it on into the Freq block
    that follows, and the Hessian was read from the wrong place.
    """
    text = (CASES[0] / "stdout").read_text()
    start = text.rfind("\n 1\\1\\")
    end = text.find("\\@", start) + 2
    block = text[start:end]
    parts = "".join(line[1:] for line in block.split("\n")[1:]).split("\\\\")
    assert len(parts) == 8  # header, route, title, atoms, properties, Hessian, ...
    flat = (
        "\\\\".join([*parts[:5], parts[6]]) + "\\\\\\"
    )  # no Hessian; ends in 3 backslashes
    lines = [flat[i : i + 69] for i in range(0, len(flat), 69)]
    assert lines[-1].endswith("\\")  # the terminator will wrap before its @
    opt_only = "\n 1\\1\\" + "\n ".join(lines) + "\n @"
    doubled = text[:start] + opt_only + "\n\n A QUOTATION\n" + block + text[end:]
    codec = get_codec("gaussian")
    data = codec.parse(CalcType.optimization, stdout=doubled)
    expected = codec.parse(CalcType.optimization, stdout=text)
    assert isinstance(data, OptimizationData)
    assert isinstance(expected, OptimizationData)
    hessian = data.frames[-1].results.hessian
    assert hessian is not None
    reference = expected.frames[-1].results.hessian
    assert reference is not None
    assert np.array_equal(hessian, reference)
