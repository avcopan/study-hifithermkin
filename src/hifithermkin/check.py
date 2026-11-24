"""Check an imported database against its sources (the subset gate, and the review).

    python -m hifithermkin.check rgd1 --db DB [--sample 200]

Prints a Markdown report: counts, round trips against the decoder, frequencies from the
stored Hessians against the logs', RDKit InChI against the source SMILES, energies
against the h5 and SDF, and time and size per point.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import autogeom
import h5py
import numpy as np
from autostorage import (
    CalculationRow,
    Database,
    GeometryRow,
    IdentityAlgorithmRow,
    IdentityRow,
    IdentityStationaryLink,
    ModelRow,
    StageRow,
    StationaryPointRow,
    StepRow,
)
from rdkit import Chem
from rdkit.rdBase import DisableLog
from sqlalchemy import func
from sqlmodel import col, select

from . import sources
from .parse import ParsedLog, parse_log

if TYPE_CHECKING:
    from collections.abc import Callable

    from autostorage import AutostorageSession

FREQUENCY_TOLERANCE = 1.0  # cm^-1
LINE = "\n"
CONNECTIVITY = 3  # InChI layers: version, formula, connectivity
MAX_EXAMPLES = 5


def _table(rows: list[tuple[Any, ...]], header: tuple[str, ...]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return LINE.join(out)


def counts_section(session: AutostorageSession, out: Path) -> str:
    """Rows in the database, and what happened to the logs."""
    lines = ["## Counts"]
    points = Counter(
        {
            f"order {order}": n
            for order, n in session.exec(
                select(StationaryPointRow.order, func.count()).group_by(
                    col(StationaryPointRow.order)
                )
            )
        }
    )
    for name, table in {
        "geometries": GeometryRow,
        "calculations": CalculationRow,
        "models": ModelRow,
        "stages": StageRow,
        "steps": StepRow,
    }.items():
        lines.append(
            f"- {name}: {session.exec(select(func.count()).select_from(table)).one()}"
        )
    ts_stages = session.exec(
        select(func.count()).select_from(StageRow).where(col(StageRow.is_ts))
    ).one()
    lines.append(f"- TS stages: {ts_stages}")
    lines.append(f"- stationary points: {dict(points)}")
    validated = session.exec(
        select(func.count())
        .select_from(StationaryPointRow)
        .where(col(StationaryPointRow.is_validated))
    ).one()
    lines.append(f"- validated points: {validated}")
    identities = session.exec(
        select(
            IdentityAlgorithmRow.name,
            func.count(col(IdentityStationaryLink.identity_id)),
        )
        .join(
            IdentityRow,
            col(IdentityRow.algorithm_name) == col(IdentityAlgorithmRow.name),
            isouter=True,
        )
        .join(
            IdentityStationaryLink,
            col(IdentityStationaryLink.identity_id) == col(IdentityRow.id),
            isouter=True,
        )
        .group_by(IdentityAlgorithmRow.name)
    ).all()
    lines.append(f"- identity links by algorithm: {dict(identities)}")
    lines.append("")
    lines.append("Models:")
    rows = []
    for model in session.exec(select(ModelRow)).all():
        n = session.exec(
            select(func.count())
            .select_from(CalculationRow)
            .where(CalculationRow.model_id == model.id)
        ).one()
        hashes = sorted(
            link.hash.value
            for link in model.hash_links
            if link.hash.algorithm_name == "minimal@1"
        )
        rows.append(
            (
                model.program_version,
                model.method,
                model.basis,
                json.dumps(model.keywords),
                hashes,
                n,
            )
        )
    lines.append(
        _table(
            rows,
            ("version", "method", "basis", "keywords", "minimal@1", "calculations"),
        )
    )
    summary = Path(f"{out}.summary.json")
    if summary.exists():
        data = json.loads(summary.read_text())
        for phase, body in data.items():
            lines.append(f"\n{phase}: `{json.dumps(body['counts'])}`")
            if body["unclassified_tokens"]:
                lines.append(
                    "unclassified route tokens: "
                    f"`{json.dumps(body['unclassified_tokens'])}`"
                )
    return LINE.join(lines)


def _geometry(row: GeometryRow) -> autogeom.Geometry:
    return autogeom.Geometry(
        symbols=row.symbols,
        coordinates=row.coordinates,
        charge=row.charge,
        spin=row.spin,
    )


def frequencies_section(session: AutostorageSession) -> str:
    """Frequencies from the stored Hessians against the logs' (every point)."""
    worst = 0.0
    bad: list[tuple[str, float]] = []
    n = 0
    for point in session.exec(select(StationaryPointRow)).all():
        hessian = next(
            (
                p.value
                for p in point.geometry.properties
                if p.property_kind_name == "hessian"
            ),
            None,
        )
        member = (point.calculation.input_provenance or {})["member"]
        logged = np.array((point.calculation.output_provenance or {})["frequencies"])
        try:
            vib = autogeom.vibrational_analysis(
                autogeom.PESPoint(
                    geometry=_geometry(point.geometry),
                    hessian=np.asarray(hessian, dtype=float),
                )
            )
        except Exception as exc:  # noqa: BLE001
            bad.append((f"{member}: {type(exc).__name__}", float("nan")))
            continue
        freqs = np.sort(vib.frequencies)
        if len(freqs) != len(logged):
            bad.append((f"{member}: {len(freqs)} vs {len(logged)} modes", float("nan")))
            continue
        diff = float(np.max(np.abs(freqs - np.sort(logged)))) if len(freqs) else 0.0
        worst = max(worst, diff)
        n += 1
        if diff > FREQUENCY_TOLERANCE:
            bad.append((member, diff))
    lines = [
        "## Frequencies from the stored Hessian",
        (
            f"{n} points compared; worst |Δν| = {worst:.3f} cm⁻¹; "
            f"{len(bad)} beyond {FREQUENCY_TOLERANCE} cm⁻¹ or not comparable."
        ),
    ]
    lines += [
        f"- {member}: {diff:.2f}"
        for member, diff in sorted(bad, key=lambda b: -b[1] if b[1] == b[1] else 0)[:30]
    ]
    return LINE.join(lines)


def _read_log(
    member: str, tar_members: dict[str, bytes], dataset: str
) -> ParsedLog | None:
    if dataset == "rgd1":
        raw = sources.rgd1_zip_read(member)
        out = parse_log("rgd1", "", member, raw)
    else:
        raw = gzip.decompress(tar_members[member])
        out = parse_log("stjohn", "", member, raw)
    return out if isinstance(out, ParsedLog) else None


def _stjohn_members(wanted: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(sources.stjohn_tar(), "r|gz") as tar:
        for member in tar:
            if member.name in wanted:
                handle = tar.extractfile(member)
                if handle is not None:
                    found[member.name] = handle.read()
                if len(found) == len(wanted):
                    break
    return found


def round_trip_section(session: AutostorageSession, dataset: str, sample: int) -> str:
    """Geometry, energy and Hessian out of the database against the decoder."""
    points = list(session.exec(select(StationaryPointRow)).all())
    random.Random(0).shuffle(points)  # noqa: S311 - a reproducible sample
    points = points[:sample]
    members = {(p.calculation.input_provenance or {})["member"] for p in points}
    packed = _stjohn_members(members) if dataset == "stjohn" else {}
    worst = {"coordinates": 0.0, "energy": 0.0, "hessian": 0.0}
    n = 0
    for point in points:
        member = (point.calculation.input_provenance or {})["member"]
        log = _read_log(member, packed, dataset)
        if log is None:
            continue
        values = {p.property_kind_name: p.value for p in point.geometry.properties}
        worst["coordinates"] = max(
            worst["coordinates"],
            float(np.max(np.abs(point.geometry.coordinates - log.coordinates))),
        )
        worst["energy"] = max(
            worst["energy"], abs(float(values["energy"]) - log.energy)
        )
        worst["hessian"] = max(
            worst["hessian"], float(np.max(np.abs(values["hessian"] - log.hessian)))
        )
        n += 1
    return (
        "## Round trips\n"
        f"{n} sampled points re-read from the archive and decoded again; "
        "largest differences: "
        f"coordinates {worst['coordinates']:.2e} Å, energy {worst['energy']:.2e} Eh, "
        f"Hessian {worst['hessian']:.2e} Eh/bohr² (stored as float32)."
    )


def _kind(source_smiles: str, stored: str, expected: str) -> str:
    """Classify an InChI mismatch."""
    layers = lambda s: s.split("/")  # noqa: E731
    if (
        layers(stored)[:CONNECTIVITY] == layers(expected)[:CONNECTIVITY]
        and len(layers(stored)) > CONNECTIVITY
    ):
        return "stereo or hydrogen layers"
    if layers(stored)[1:2] != layers(expected)[1:2]:
        return "formula"
    if "+" in source_smiles or "-]" in source_smiles:
        return "charge separation"
    return "connectivity (resonance or other)"


def inchi_section(
    session: AutostorageSession, source_smiles: Callable[[str], str | None]
) -> str:
    """RDKit InChI of each order-0 point against the source SMILES."""
    DisableLog("rdApp.*")
    kinds: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    n = 0
    stmt = (
        select(StationaryPointRow, IdentityRow.value)
        .join(
            IdentityStationaryLink,
            col(IdentityStationaryLink.stationary_id) == col(StationaryPointRow.id),
        )
        .join(
            IdentityRow, col(IdentityRow.id) == col(IdentityStationaryLink.identity_id)
        )
        .where(
            IdentityRow.algorithm_name == "rdkit_inchi", StationaryPointRow.order == 0
        )
    )
    for point, stored in session.exec(stmt).all():
        member = (point.calculation.input_provenance or {})["member"]
        smiles = source_smiles(member)
        if smiles is None:
            kinds["no source smiles"] += 1
            continue
        molecule = Chem.MolFromSmiles(smiles)
        expected = Chem.MolToInchi(molecule) if molecule is not None else ""
        n += 1
        if stored == expected:
            kinds["match"] += 1
            continue
        kind = _kind(smiles, stored, expected)
        kinds[kind] += 1
        if len(examples[kind]) < MAX_EXAMPLES:
            examples[kind].append(f"{member}: {smiles} -> {stored} vs {expected}")
    lines = [
        "## InChI of the geometry against the source SMILES",
        f"{n} order-0 points: {dict(kinds)}",
    ]
    for kind, items in examples.items():
        lines += [f"- {kind}: {item}" for item in items]
    return LINE.join(lines)


def rgd1_energies_section(session: AutostorageSession, h5_path: Path) -> str:
    """TS and reaction energies against the h5 (R_E, P_E, TS_E)."""
    lines = ["## Energies against the h5"]
    worst_ts = 0.0
    worst_r = 0.0
    n = 0
    energy: dict[str, float] = {}
    for point in session.exec(select(StationaryPointRow)).all():
        member = (point.calculation.input_provenance or {})["member"]
        value = next(
            p.value
            for p in point.geometry.properties
            if p.property_kind_name == "energy"
        )
        energy[member] = float(value)
    with h5py.File(h5_path) as h5:
        for member, e_ts in energy.items():
            if not member.endswith("-TS.out"):
                continue
            channel = member.rsplit("/", 1)[-1][: -len("-TS.out")]
            if channel in h5:
                worst_ts = max(worst_ts, abs(e_ts - float(h5[channel]["TS_E"][()])))
                n += 1
        mapping = sources.rgd1_mapping()
        m = 0
        for channel, (reactants, products) in mapping.items():
            parts = [
                energy.get(f"raw_output/RP/{mol}.out")
                for mol in (*reactants, *products)
            ]
            if channel not in h5 or None in parts:
                continue
            e_r = sum(energy[f"raw_output/RP/{mol}.out"] for mol in reactants)
            e_p = sum(energy[f"raw_output/RP/{mol}.out"] for mol in products)
            worst_r = max(
                worst_r,
                abs(e_r - float(h5[channel]["R_E"][()])),
                abs(e_p - float(h5[channel]["P_E"][()])),
            )
            m += 1
    lines.append(f"- TS: {n} compared, largest |E(log) - TS_E| = {worst_ts:.2e} Eh")
    lines.append(
        f"- reactants/products: {m} reactions compared (sums of the species "
        f"energies), largest difference {worst_r:.2e} Eh"
    )
    return LINE.join(lines)


def stjohn_sdf(path: Path | None = None) -> dict[str, tuple[str, float]]:
    """Source SMILES and SCF energy by record name (``<id>_<suffix>``) of the SDF."""
    path = (
        path or sources.data_dir() / "stjohn-radicals/20200415_radical_database.sdf.gz"
    )
    DisableLog("rdApp.*")
    records: dict[str, tuple[str, float]] = {}
    with gzip.open(path, "rb") as handle:
        for molecule in Chem.ForwardSDMolSupplier(
            handle, removeHs=False, sanitize=False
        ):
            if molecule is not None:
                records[molecule.GetProp("_Name")] = (
                    molecule.GetProp("SMILES"),
                    float(molecule.GetProp("SCFEnergy")),
                )
    return records


def stjohn_energies_section(
    session: AutostorageSession, sdf: dict[str, tuple[str, float]]
) -> str:
    """Log energies against the SDF's ``SCFEnergy``."""
    worst = 0.0
    n = 0
    missing = 0
    for point in session.exec(select(StationaryPointRow)).all():
        member = (point.calculation.input_provenance or {})["member"]
        record = sdf.get(Path(member).name.removesuffix(".log.gz"))
        if record is None:
            missing += 1
            continue
        value = next(
            p.value
            for p in point.geometry.properties
            if p.property_kind_name == "energy"
        )
        worst = max(worst, abs(float(value) - record[1]))
        n += 1
    return (
        f"## Energies against the SDF\n{n} compared ({missing} without a record), "
        f"largest |E(log) - SCFEnergy| = {worst:.2e} Eh"
    )


def size_section(out: Path, session: AutostorageSession) -> str:
    """Time and size per point, extrapolated."""
    n = session.exec(select(func.count()).select_from(StationaryPointRow)).one()
    size = out.stat().st_size
    lines = [
        "## Time and size",
        (
            f"- database: {size / 1e6:.1f} MB for {n} points = "
            f"{size / max(n, 1) / 1e6:.3f} MB per point (INCITE estimate: 0.25 MB)"
        ),
    ]
    summary = Path(f"{out}.summary.json")
    if summary.exists():
        counts = json.loads(summary.read_text())["logs"]["counts"]
        seconds, logs = counts.get("seconds", 0), counts.get("logs", 0)
        if logs:
            lines.append(
                f"- the run: {seconds} s for {counts['logs_seen']} logs read, "
                f"{logs} imported ({seconds / logs:.3f} s per imported log, "
                "including the reads of skipped logs)"
            )
    return LINE.join(lines)


def main(argv: list[str] | None = None) -> None:
    """Print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["rgd1", "stjohn"])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args(argv)
    sections = []
    with Database(args.db) as db, db.session() as session:
        sections.append(counts_section(session, args.db))
        sections.append(round_trip_section(session, args.dataset, args.sample))
        sections.append(frequencies_section(session))
        if args.dataset == "rgd1":
            smiles = {}
            for line in (
                sources.rgd1_zip_read("raw_output/RandP_smiles.txt")
                .decode()
                .splitlines()[1:]
            ):
                text, _, mol = line.rpartition("   ")
                smiles[f"raw_output/RP/{mol}.out"] = text
            sections.append(inchi_section(session, smiles.get))
            sections.append(
                rgd1_energies_section(
                    session, sources.data_dir() / "rgd1/RGD1_allrxns.h5"
                )
            )
        else:
            sdf = stjohn_sdf()

            def by_member(member: str) -> str | None:
                record = sdf.get(Path(member).name.removesuffix(".log.gz"))
                return record[0] if record else None

            sections.append(inchi_section(session, by_member))
            sections.append(stjohn_energies_section(session, sdf))
        sections.append(size_section(args.db, session))
    print(LINE.join(f"{s}{LINE}" for s in sections))  # noqa: T201


if __name__ == "__main__":
    main()
