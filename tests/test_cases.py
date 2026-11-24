"""Every fixture case under tests/data runs through qccodec's shared harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from qccodec import get_codec
from qccodec.testing import Case, cases, check_codec, run_case

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("case", cases(DATA), ids=str)
def test_case(case: Case) -> None:
    """Decode and per-parser checks for one recorded Gaussian run."""
    run_case(get_codec(case.input_data.program), case)


def test_codec_satisfies_the_contract() -> None:
    """The plugin passes qccodec's structural checks."""
    check_codec(get_codec("gaussian"))
