"""A small import of the fixture logs, through a zip and a tarball, into a database."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import zipfile
from typing import TYPE_CHECKING

import h5py
import pytest
from autostorage import (
    CalculationRow,
    Database,
    IdentityStationaryLink,
    StationaryPointRow,
)
from sqlmodel import select

from hifithermkin import check, importer, sources
from hifithermkin.parse import ParsedLog, parse_log

from .test_graphs import PRODUCT, REACTANT, write_h5

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
    (root / "rgd1/RGD1CHNO_AMsmiles.csv").write_text(
        "reaction,reactant,product,DE_F,DE_B,DG_F,DG_B,DH\nMR_299320_1,a,b,0,0,0,0,0\n"
    )
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


def test_check_report_on_the_small_import(data: Path, tmp_path: Path) -> None:
    assert data.exists()
    out = tmp_path / "rgd1.db"
    importer.run_logs("rgd1", out, subset=False, workers=1)
    smiles = {
        "raw_output/RP/mol83138.out": "N",
        "raw_output/RP/mol12021.out": "C=C",
    }
    with Database(out) as db, db.session() as session:
        counts = check.counts_section(session, out)
        assert "order 1" in counts
        assert "b3lyp/tzvp" in counts
        assert "worst |Δν|" in check.frequencies_section(session)
        round_trip = check.round_trip_section(session, "rgd1", 10)
        assert "3 sampled points" in round_trip
        inchi = check.inchi_section(session, smiles.get)
        assert "'match': 2" in inchi
        assert "MB per point" in check.size_section(out, session)


def test_check_energies_against_h5_and_sdf(data: Path, tmp_path: Path) -> None:
    out = tmp_path / "rgd1.db"
    importer.run_logs("rgd1", out, subset=False, workers=1)
    ts = parse_log(
        "rgd1", "a", "final-TS/MR_299320_1-TS.out", _log("rgd1_ts_MR_299320_1")
    )
    assert isinstance(ts, ParsedLog)
    with h5py.File(tmp_path / "x.h5", "w") as h5:
        group = h5.create_group("MR_299320_1")
        group["TS_E"] = ts.energy + 1e-3
        group["R_E"] = 0.0
        group["P_E"] = 0.0
    with Database(out) as db, db.session() as session:
        text = check.rgd1_energies_section(session, tmp_path / "x.h5")
    assert "TS: 1 compared, largest |E(log) - TS_E| = 1.00e-03" in text
    assert "barriers against the CSV" in text

    sdf = data / "stjohn-radicals/20200415_radical_database.sdf.gz"
    block = "\n  RDKit          3D\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\n"
    block += "M  V30 BEGIN CTAB\nM  V30 COUNTS 0 0 0 0 0\nM  V30 END CTAB\nM  END\n"
    radical = parse_log(
        "stjohn", "a", "m/stjohn_radical_um062x.log.gz", _log("stjohn_radical_um062x")
    )
    assert isinstance(radical, ParsedLog)
    record = (
        f"stjohn_radical_um062x{block}"
        f">  <SMILES>\n[CH2]C\n\n>  <SCFEnergy>\n{radical.energy + 2e-3}\n\n$$$$\n"
    )
    with gzip.open(sdf, "wt") as handle:
        handle.write(record)
    records = check.stjohn_sdf()
    assert records["stjohn_radical_um062x"][0] == "[CH2]C"
    stjohn_out = tmp_path / "stjohn.db"
    importer.run_logs("stjohn", stjohn_out, subset=False, workers=1)
    with Database(stjohn_out) as db, db.session() as session:
        text = check.stjohn_energies_section(session, records)
    assert "1 compared (1 without a record)" in text
    assert "2.00e-03" in text


def test_import_with_graph_identities(
    data: Path, tmp_path: Path, logs: dict[str, ParsedLog]
) -> None:
    (data / "rgd1/RGD1CHNO_AMsmiles.csv").write_text(
        "reaction,reactant,product,DE_F,DE_B,DG_F,DG_B,DH\n"
        f'MR_299320_1,"{REACTANT}","{PRODUCT}",0,0,0,0,0\n'
    )
    write_h5(data, logs["rgd1_ts_MR_299320_1"], shuffle=False)
    out = tmp_path / "graphs.db"
    counts = importer.run_logs("rgd1", out, subset=False, workers=1, graphs=True)
    assert counts["graph_ts_from_mapped_smiles"] == 1, dict(counts)
    assert counts["graph_perceived"] == 2
    with Database(out) as db, db.session() as session:
        links = session.exec(select(IdentityStationaryLink)).all()
        names = [link.identity.algorithm_name for link in links]
    assert names.count("protomol_graph@0") == 3  # the TS, and the two perceived
