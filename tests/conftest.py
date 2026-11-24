"""Shared fixtures: the decoder's Gaussian logs as parsed records."""

from __future__ import annotations

from pathlib import Path

import pytest

from hifithermkin.parse import ParsedLog, parse_log

CASES = Path(__file__).parent / "data" / "gaussian"


@pytest.fixture(scope="session")
def logs() -> dict[str, ParsedLog]:
    """Every fixture log, parsed, by case name."""
    out: dict[str, ParsedLog] = {}
    for case in sorted(p for p in CASES.iterdir() if p.is_dir()):
        suffix = "-TS.out" if "_ts_" in case.name else ".out"
        log = parse_log(
            "rgd1" if case.name.startswith("rgd1") else "stjohn",
            "archive",
            f"member/{case.name}{suffix}",
            (case / "stdout").read_bytes(),
        )
        assert isinstance(log, ParsedLog)
        out[case.name] = log
    return out
