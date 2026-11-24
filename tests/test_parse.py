"""Parsing a log into a record, and the reasons to skip one."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hifithermkin import parse
from hifithermkin.parse import ParsedLog, Skip, in_subset, parse_log

if TYPE_CHECKING:
    import pytest

CASES = Path(__file__).parent / "data" / "gaussian"


def _raw(name: str) -> bytes:
    return (CASES / name / "stdout").read_bytes()


def test_minimum_record() -> None:
    log = parse_log(
        "rgd1", "rgd1.zip", "raw_output/RP/mol83138.out", _raw("rgd1_nh3_minimum")
    )
    assert isinstance(log, ParsedLog)
    assert log.calc_type == "optimization"
    assert log.symbols == ["N", "H", "H", "H"] or sorted(log.symbols) == [
        "H",
        "H",
        "H",
        "N",
    ]
    assert log.nimag == 0
    assert log.hessian.shape == (12, 12)
    assert log.spin == 0
    assert len(log.md5) == 32
    assert log.output["program_version"] == "G16RevB.01"
    assert "routes" not in log.output
    assert len(log.routes) == 2


def test_coordinates_are_angstrom() -> None:
    log = parse_log("rgd1", "a", "RP/mol1.out", _raw("rgd1_nh3_minimum"))
    assert isinstance(log, ParsedLog)
    bonds = np.linalg.norm(log.coordinates[1:] - log.coordinates[0], axis=1)
    assert np.all((bonds > 0.9) & (bonds < 1.2))  # N-H, in Å (not 1.9 bohr)


def test_ts_record() -> None:
    log = parse_log(
        "rgd1", "a", "final-TS/MR_299320_1-TS.out", _raw("rgd1_ts_MR_299320_1")
    )
    assert isinstance(log, ParsedLog)
    assert (log.calc_type, log.nimag) == ("transition_state", 1)


def test_radical_spin() -> None:
    log = parse_log("stjohn", "a", "4321_0.log", _raw("stjohn_radical_um062x"))
    assert isinstance(log, ParsedLog)
    assert log.spin == 1
    assert log.model.keywords["reference"] == "uks"


def test_truncated_log_is_a_parse_error() -> None:
    raw = _raw("rgd1_nh3_minimum")
    skip = parse_log("rgd1", "a", "RP/mol1.out", raw[: len(raw) // 2])
    assert isinstance(skip, Skip)
    assert skip.reason == "parse_error"
    assert skip.detail


def test_broken_hessian_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parse, "MAX_FREQUENCY", 10.0)
    skip = parse_log("rgd1", "a", "RP/mol1.out", _raw("rgd1_nh3_minimum"))
    assert isinstance(skip, Skip)
    assert skip.reason == "broken_hessian"


def test_subset_filter() -> None:
    assert in_subset(["C", "H", "H", "H", "O", "H"])
    assert not in_subset(["C", "N", "H"])  # nitrogen
    assert not in_subset(["C"] * 5 + ["H"])  # five heavy atoms
    skip = parse_log("rgd1", "a", "RP/mol1.out", _raw("rgd1_nh3_minimum"), subset=True)
    assert isinstance(skip, Skip)
    assert skip.reason == "outside_subset"


def test_echoed_input_of_a_single_job(logs: dict[str, ParsedLog]) -> None:
    (job,) = logs["rgd1_nh3_minimum"].echoed_input
    assert job["link0"] == ["%NProcShared=8", "%Mem=8000MB"]
    assert job["route"].endswith("Int=UltraFine SCF=QC Freq")
    assert (job["title"], job["charge"], job["multiplicity"]) == ("mol83138 Opt", 0, 1)
    assert len(job["geometry"]) == 4
    assert job["geometry"][0] == "N 1.05522 0.0701 -0.06024"
    assert job["text"].splitlines()[:3] == [
        "%NProcShared=8",
        "%Mem=8000MB",
        job["route"],
    ]
    assert job["text"].endswith("H 2.07255 0.01222 -0.06767\n")


def test_echoed_input_of_a_two_job_log(logs: dict[str, ParsedLog]) -> None:
    first, second = logs["stjohn_radical_stable_opt"].echoed_input
    assert first["route"].startswith("# stable=opt")
    assert (first["charge"], first["multiplicity"], len(first["geometry"])) == (
        0,
        2,
        14,
    )
    # the second job reads its geometry from the checkpoint: nothing is echoed
    assert second["route"].endswith("guess=read geom=check")
    assert second["geometry"] == []
    assert second["link0"][0].startswith("%chk=")
