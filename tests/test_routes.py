"""Route normalization: the models the Gaussian logs give."""

from __future__ import annotations

import pytest

from hifithermkin.routes import model_from_routes, split_route

RGD1 = [
    "# B3LYP/TZVP EmpiricalDispersion=GD3 Opt=(maxcycles=100) Int=UltraFine",
    "#N Geom=AllCheck Guess=TCheck SCRF=Check GenChk RB3LYP/TZVP Freq",
]
RGD1_TS = [
    (
        "# B3LYP/TZVP EmpiricalDispersion=GD3 OPT=(TS, CALCALL, NOEIGEN, maxcyc=100) "
        "Int=UltraFine"
    ),
    "#N Geom=AllCheck Guess=TCheck SCRF=Check GenChk RB3LYP/TZVP Freq",
]
STJOHN_R = [
    "# opt freq M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400) nosymm",
    "#N Geom=AllCheck Guess=TCheck SCRF=Check GenChk RM062X/def2TZVP Freq",
]
STJOHN_U = [
    "# stable=opt M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400) nosymm guess=mix",
    (
        "# opt freq M062X/Def2TZVP scf=(xqc,maxconventionalcycles=400) nosymm "
        "guess=read geom=check"
    ),
    "#N Geom=AllCheck Guess=TCheck SCRF=Check GenChk UM062X/def2TZVP Freq",
]


def _model(method: str, basis: str, routes: list[str]):  # noqa: ANN202
    return model_from_routes(
        program_version="G16RevB.01", method=method, basis=basis, routes=routes
    )


def test_split_route_keeps_groups_whole() -> None:
    tokens = split_route("# B3LYP/TZVP OPT=(TS, CALCALL, NOEIGEN) Int=UltraFine")
    assert tokens == ["B3LYP/TZVP", "OPT=(TS, CALCALL, NOEIGEN)", "Int=UltraFine"]


@pytest.mark.parametrize("routes", [RGD1, RGD1_TS])
def test_rgd1_model(routes: list[str]) -> None:
    spec, unclassified = _model("RB3LYP", "TZVP", routes)
    assert (spec.method, spec.basis) == ("B3LYP", "TZVP")
    assert spec.keywords == {"reference": "rks", "empiricaldispersion": "gd3"}
    assert unclassified == []


def test_ts_and_minimum_share_a_model() -> None:
    assert (
        _model("RB3LYP", "TZVP", RGD1)[0].key
        == _model("RB3LYP", "TZVP", RGD1_TS)[0].key
    )


def test_stjohn_models_differ_only_in_reference_and_stable() -> None:
    closed, left1 = _model("RM062X", "def2TZVP", STJOHN_R)
    radical, left2 = _model("UM062X", "def2TZVP", STJOHN_U)
    assert (closed.method, closed.basis) == ("M06-2X", "def2-TZVP")
    assert closed.keywords == {"reference": "rks"}
    assert radical.keywords == {"reference": "uks", "stable": "opt"}
    assert left1 == left2 == []


def test_unknown_tokens_are_reported() -> None:
    _, unclassified = _model("RM062X", "def2TZVP", [*STJOHN_R, "# cphf=rdfreq"])
    assert unclassified == ["cphf=rdfreq"]


def test_unknown_method_is_kept_and_reported() -> None:
    spec, unclassified = _model("UWB97XD", "6-31G*", ["# opt"])
    assert spec.method == "UWB97XD"
    assert unclassified == ["method:UWB97XD"]


def test_scf_convergence_is_a_model_keyword() -> None:
    spec, _ = _model("RM062X", "def2TZVP", ["# scf=(xqc,conver=10) M062X/def2TZVP"])
    assert spec.keywords["scf"] == "conver=10"


def test_only_the_g16_default_grid_is_dropped() -> None:
    routes = ["# Int=UltraFine M062X/def2TZVP", "# Int=SuperFine"]
    g16, _ = _model("RM062X", "def2TZVP", routes[:1])
    g09, _ = model_from_routes(
        program_version="G09RevD.01",
        method="RM062X",
        basis="def2TZVP",
        routes=routes[:1],
    )
    fine, _ = _model("RM062X", "def2TZVP", routes[1:])
    assert "integral" not in g16.keywords
    assert g09.keywords["integral"] == "ultrafine"
    assert fine.keywords["integral"] == "superfine"
