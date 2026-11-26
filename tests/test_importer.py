"""The writer: batches, checkpoint lines and failures, on the fixture logs."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
from autostorage import CalculationRow, Database
from sqlmodel import select

from hifithermkin.check import flags_section
from hifithermkin.importer import _read_checkpoint, _Writer
from hifithermkin.parse import Skip

if TYPE_CHECKING:
    from pathlib import Path

    from hifithermkin.parse import ParsedLog


def test_writer_checkpoints_committed_batches(
    tmp_path: Path, logs: dict[str, ParsedLog]
) -> None:
    out = tmp_path / "x.db"
    with Database(out) as db, db.session() as session:
        writer = _Writer("rgd1", session, out, batch=2)
        writer.take(logs["rgd1_nh3_minimum"])
        assert not (tmp_path / "x.db.checkpoint.tsv").read_text()  # not committed yet
        writer.take(logs["rgd1_ts_MR_299320_1"])  # the batch of two commits
        writer.take(Skip("rgd1", "raw_output/RP/mol5093.out", "parse_error", "garbage"))
        writer.take(Skip("rgd1", "raw_output/RP/mol7.out", "outside_subset"))
        writer.commit()
        writer.close()
        assert len(session.exec(select(CalculationRow)).all()) == 2
    index = _read_checkpoint(tmp_path / "x.db.checkpoint.tsv")
    assert len(index) == 4
    ts = index[logs["rgd1_ts_MR_299320_1"].member]
    assert (ts.status, ts.order) == ("ok", 1)
    assert ts.point_id
    assert index["raw_output/RP/mol5093.out"].status == "parse_error"
    failures = (tmp_path / "x.db.failures.csv").read_text().splitlines()
    assert failures == [
        "member,reason,detail",
        "raw_output/RP/mol5093.out,parse_error,garbage",
    ]
    assert writer.counts["skipped_outside_subset"] == 1


@pytest.mark.parametrize("status", ["ts_no_source", "ts_atom_order_mismatch"])
def test_writer_lists_a_ts_without_a_graph(
    status: str, tmp_path: Path, logs: dict[str, ParsedLog]
) -> None:
    log = dataclasses.replace(
        logs["rgd1_ts_MR_299320_1"], graph_status=status, graph_detail="why"
    )
    out = tmp_path / "x.db"
    with Database(out) as db, db.session() as session:
        writer = _Writer("rgd1", session, out, batch=2)
        writer.take(log)
        writer.commit()
        writer.close()
        assert len(session.exec(select(CalculationRow)).all()) == 1  # imported
    failures = (tmp_path / "x.db.failures.csv").read_text().splitlines()
    assert failures == ["member,reason,detail", f"{log.member},graph_{status},why"]
    assert writer.store.counts[f"graph_{status}"] == 1
    report = flags_section(out)
    assert f"1 TSs: {{'graph_{status}': 1}}" in report
    assert f"- {log.member}: graph_{status}: why" in report
