"""Graphs for the points AutoStorage cannot perceive: TSs and false minima."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
from autostorage import Database, StationaryPointRow
from sqlmodel import select

from hifithermkin import graphs, sources
from hifithermkin.graphs import attach_graph, mapped_reaction_graph, tsg_mismatch
from hifithermkin.parse import ParsedLog
from hifithermkin.reactions import member_of_ts
from hifithermkin.store import Store

if TYPE_CHECKING:
    from pathlib import Path

# MR_299320_1 of RGD1: the SMILES of CSV row, atoms in map-number order
REACTANT = r"[C:1](\[N:3][C:2]([H:7])[H:8])(=[N:4]\[H:5])[H:6]"
PRODUCT = "[C:1]1([H:6])[N:3][C:2]([H:7])([H:8])[N:4]1[H:5]"


def test_reaction_graph_is_in_map_number_order() -> None:
    graph = mapped_reaction_graph(REACTANT, PRODUCT)
    assert graph.symbols == ("C", "C", "N", "N", "H", "H", "H", "H")
    assert graph.forming == ((1, 3),)  # C2-N4 closes the ring
    assert graph.breaking == ()


def test_map_numbers_decide_the_order_not_the_smiles() -> None:
    shuffled_r = "[H:8][C:2]([H:7])([N:3]\\[C:1](=[N:4]\\[H:5])[H:6])"
    shuffled_p = "[H:6][C:1]1[N:3][C:2]([H:7])([H:8])[N:4]1[H:5]"
    shuffled = mapped_reaction_graph(shuffled_r, shuffled_p)
    original = mapped_reaction_graph(REACTANT, PRODUCT)
    # (the C=N stereo marks differ between the two spellings, so compare connectivity)
    assert shuffled.symbols == original.symbols
    assert shuffled.bonds == original.bonds
    assert (shuffled.breaking, shuffled.forming) == (
        original.breaking,
        original.forming,
    )


def test_tsg_must_have_the_log_atom_order(logs: dict[str, ParsedLog]) -> None:
    log = logs["rgd1_ts_MR_299320_1"]
    elements = np.array([{"C": 6, "N": 7, "H": 1}[s] for s in log.symbols])
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = log.coordinates @ rotation + 3.0
    assert tsg_mismatch(log, moved, elements) is None
    swapped = moved.copy()
    swapped[[0, 1]] = swapped[[1, 0]]  # two atoms in the other order
    assert "from the h5's TSG" in str(tsg_mismatch(log, swapped, elements))
    assert "elements" in str(tsg_mismatch(log, moved, elements[::-1].copy()))


@pytest.fixture
def rgd1_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The CSV and h5 of one reaction, as the importer reads them."""
    root = tmp_path / "data"
    (root / "rgd1").mkdir(parents=True)
    (root / "rgd1/RGD1CHNO_AMsmiles.csv").write_text(
        f"reaction,reactant,product,DE_F,DE_B,DG_F,DG_B,DH\n"
        f'MR_299320_1,"{REACTANT}","{PRODUCT}",0,0,0,0,0\n'
    )
    monkeypatch.setenv("DATA_DIR", str(root))
    monkeypatch.setattr(graphs, "_smiles", None)
    monkeypatch.setattr(graphs, "_h5", None)
    return root


def write_h5(root: Path, log: ParsedLog, *, shuffle: bool) -> None:
    z = {"C": 6, "N": 7, "H": 1}
    coordinates = log.coordinates + 1.0
    elements = np.array([z[s] for s in log.symbols])
    if shuffle:
        coordinates = coordinates[::-1].copy()
    with h5py.File(root / "rgd1/RGD1_allrxns.h5", "w") as h5:
        group = h5.create_group("MR_299320_1")
        group["TSG"] = coordinates
        group["elements"] = elements


def test_ts_gets_its_graph_and_is_stored_with_an_identity(
    rgd1_sources: Path, logs: dict[str, ParsedLog], tmp_path: Path
) -> None:
    log = logs["rgd1_ts_MR_299320_1"]
    ts = dataclasses.replace(log, member="raw_output/final-TS/MR_299320_1-TS.out")
    write_h5(rgd1_sources, log, shuffle=False)
    with_graph = attach_graph(ts)
    assert with_graph.graph_status == "ts_from_mapped_smiles"
    with Database(tmp_path / "g.db") as db, db.session() as session:
        Store(session).add(with_graph)
        session.commit()
        point = session.exec(select(StationaryPointRow)).one()
        names = {link.identity.algorithm_name for link in point.identity_links}
    assert "protomol_graph@0" in names


def test_ts_with_another_atom_order_gets_no_graph(
    rgd1_sources: Path, logs: dict[str, ParsedLog]
) -> None:
    log = logs["rgd1_ts_MR_299320_1"]
    ts = dataclasses.replace(log, member="raw_output/final-TS/MR_299320_1-TS.out")
    write_h5(rgd1_sources, log, shuffle=True)
    result = attach_graph(ts)
    assert result.graph is None
    assert result.graph_status == "ts_atom_order_mismatch"
    assert result.graph_detail


def test_unknown_reaction_and_minimum(
    rgd1_sources: Path, logs: dict[str, ParsedLog]
) -> None:
    log = logs["rgd1_ts_MR_299320_1"]
    write_h5(rgd1_sources, log, shuffle=False)
    other = dataclasses.replace(log, member="raw_output/final-TS/MR_1_0-TS.out")
    no_source = attach_graph(other)
    assert no_source.graph_status == "ts_no_source"
    assert no_source.graph_detail == ("not in RGD1CHNO_AMsmiles.csv or RGD1_allrxns.h5")
    minimum = attach_graph(logs["rgd1_nh3_minimum"])
    assert (minimum.graph, minimum.graph_status) == (None, "perceived")


def test_false_minimum_gets_the_graph_of_its_geometry(
    logs: dict[str, ParsedLog],
) -> None:
    false_minimum = dataclasses.replace(logs["rgd1_nh3_minimum"], nimag=1)
    result = attach_graph(false_minimum)
    assert result.graph_status == "false_minimum"
    assert result.graph.symbols == tuple(false_minimum.symbols)


BOND_CHANGES = ["MR_98994_0", "MR_2933_0", "MR_678186_0", "MR_491795_1", "MR_66887_0",
                "MR_66883_0", "MR_66883_1", "MR_20835_1"]  # fmt: skip
UNCHANGED_MISSING = ["MR_274058_1", "MR_66887_0", "MR_66883_0", "MR_66883_1"]


def _real_flags(channel: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    if (
        not sources.rgd1_zip().exists()
        or not (sources.data_dir() / "rgd1/RGD1_allrxns.h5").exists()
    ):
        pytest.skip("the RGD1 data is not on disk")
    monkeypatch.setattr(graphs, "_smiles", None)  # not another test's data
    monkeypatch.setattr(graphs, "_h5", None)
    log = sources.read_rgd1(member_of_ts(channel))
    assert isinstance(log, ParsedLog)
    result = attach_graph(log)
    assert result.graph_status == "ts_from_mapped_smiles"
    return dict(result.graph_flags)


@pytest.mark.parametrize("channel", BOND_CHANGES)
def test_flag_bond_changes_of_real_ts(
    channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = _real_flags(channel, monkeypatch)["bond_changes"]
    assert "SMILES only" in detail
    assert "-" in detail  # names the atoms, e.g. C2-N4


@pytest.mark.parametrize("channel", UNCHANGED_MISSING)
def test_flag_unchanged_bond_missing_of_real_ts(
    channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "-" in _real_flags(channel, monkeypatch)["unchanged_bond_missing"]
