"""A small import of the fixture logs, through a zip and a tarball, into a database."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import zipfile
from typing import TYPE_CHECKING

import pytest
from autostorage import CalculationRow, Database, StationaryPointRow
from sqlmodel import select

from hifithermkin import importer, sources

if TYPE_CHECKING:
    from pathlib import Path

CASES = importer.Path(__file__).parent / "data" / "gaussian"


def _log(name: str) -> bytes:
    return (CASES / name / "stdout").read_bytes()


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data folder with a tiny RGD1 zip and a tiny stjohn tarball."""
    root = tmp_path / "data"
    (root / "rgd1").mkdir(parents=True)
    (root / "stjohn-radicals").mkdir()
    mapping = (
        "channel,reactant,product,Rmols,Pmols\n"
        "MR_299320_1,N,C=C,['mol83138'],['mol12021']\n"
        "MR_5_0,N.N,C=C,\"['mol83138', 'mol83138']\",['mol12021']\n"
        "MR_6_0,C,C=C,['mol5093'],['mol12021']\n"
    )
    with zipfile.ZipFile(root / "rgd1/RGD1_rawoutput.zip", "w") as archive:
        archive.writestr("raw_output/RP/mol83138.out", _log("rgd1_nh3_minimum"))
        archive.writestr("raw_output/RP/mol12021.out", _log("rgd1_minimum_mol12021"))
        archive.writestr("raw_output/RP/mol5093.out", b"Error termination")
        archive.writestr(
            "raw_output/final-TS/MR_299320_1-TS.out", _log("rgd1_ts_MR_299320_1")
        )
        archive.writestr("raw_output/RP_mapping.csv", mapping)
        archive.writestr(
            "raw_output/RandP_smiles.txt",
            "smiles   ind\nN   mol83138\nC=C   mol12021\n",
        )
    with tarfile.open(
        root / "stjohn-radicals/20200411_scidata_logs.tar.gz", "w:gz"
    ) as tar:
        for name in ("stjohn_radical_um062x", "stjohn_closed_shell_rm062x"):
            packed = gzip.compress(_log(name))
            info = tarfile.TarInfo(f"20200411_scidata_logs/{name}.log.gz")
            info.size = len(packed)
            tar.addfile(info, io.BytesIO(packed))
    monkeypatch.setenv("DATA_DIR", str(root))
    return root


def test_rgd1_import_and_reactions(data: Path, tmp_path: Path) -> None:
    assert sources.rgd1_zip().parent.parent == data
    out = tmp_path / "rgd1.db"
    counts = importer.run_logs("rgd1", out, subset=False, workers=1, batch=2)
    assert counts["logs"] == 3
    assert counts["skipped_parse_error"] == 1  # mol5093
    assert counts["points_order_1"] == 1  # the TS
    reactions = importer.run_reactions(out, subset=False)
    assert reactions["skipped_unbalanced"] == 1  # NH3 -> C2H4
    assert reactions["skipped_ts_not_imported"] == 2  # MR_5_0 and MR_6_0 have no TS log
    summary = json.loads((tmp_path / "rgd1.db.summary.json").read_text())
    assert summary["logs"]["counts"]["models_created"] == 1
    with Database(out) as db, db.session() as session:
        assert len(session.exec(select(StationaryPointRow)).all()) == 3
    # a rerun finds everything done
    again = importer.run_logs("rgd1", out, subset=False, workers=1)
    assert again["logs_seen"] == 0


def test_rgd1_subset_keeps_c_h_o_species(data: Path, tmp_path: Path) -> None:
    assert data.exists()
    out = tmp_path / "subset.db"
    counts = importer.run_logs("rgd1", out, subset=True, workers=1)
    assert counts["logs"] == 1  # ethene only: NH3 is outside, and so are the reactions
    assert importer.run_reactions(out, subset=True)["reactions_seen"] == 0


def test_stjohn_import(data: Path, tmp_path: Path) -> None:
    assert data.exists()
    out = tmp_path / "stjohn.db"
    counts = importer.run_logs("stjohn", out, subset=False, workers=2)
    assert counts["logs"] == 2
    with Database(out) as db, db.session() as session:
        calculations = session.exec(select(CalculationRow)).all()
        members = {(c.input_provenance or {})["member"] for c in calculations}
        assert "20200411_scidata_logs/stjohn_radical_um062x.log.gz" in members
    subset = importer.run_logs("stjohn", tmp_path / "s2.db", subset=True, workers=1)
    assert subset["logs"] == 0  # these molecules have nitrogen or too many atoms
    assert subset["skipped_outside_subset"] == 2
