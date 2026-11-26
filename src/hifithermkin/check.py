"""Check an imported database against its sources (the subset gate, and the review).

    python -m hifithermkin.check rgd1 --db DB [--sample 200] [--graphs]

Prints a Markdown report, section by section: counts, round trips against the decoder,
frequencies from the stored Hessians against the logs', RDKit InChI against the source
SMILES, energies against the h5 and SDF, and time and size per point.

Frequencies, InChI and energies cover every point, read in chunks so that memory stays
bounded. Round trips and the two TS graph sections use ``--sample`` random points (the
import already flags every TS: see ``failures.csv``).
"""

from __future__ import annotations

import argparse
import csv
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
import protomol
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

from . import graphs, sources
from .parse import ParsedLog, parse_log

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Iterator, Sequence

    from autostorage import AutostorageSession

FREQUENCY_TOLERANCE = 1.0  # cm^-1
LINE = "\n"
CONNECTIVITY = 3  # InChI layers: version, formula, connectivity
MAX_EXAMPLES = 5
MAX_GRAPH_EXAMPLES = 10
MAX_BAD_FREQUENCIES = 30
CHUNK = 1000  # points in memory at a time, about 0.1 MB each with their Hessians
LOOSE = 1e-5  # Eh, agreement with the h5
HARTREE_TO_KCAL = 627.5094740631


def _table(rows: list[tuple[Any, ...]], header: tuple[str, ...]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return LINE.join(out)


def _point_ids(
    session: AutostorageSession, order: int | None = None
) -> list[uuid.UUID]:
    """The ids of every stationary point (of one order, if given), sorted."""
    stmt = select(col(StationaryPointRow.id)).order_by(col(StationaryPointRow.id))
    if order is not None:
        stmt = stmt.where(StationaryPointRow.order == order)
    return [i for i in session.exec(stmt).all() if i is not None]


def _points(
    session: AutostorageSession, ids: Sequence[uuid.UUID]
) -> Iterator[StationaryPointRow]:
    """The points of ``ids``, in that order, holding `CHUNK` of them at a time.

    The session is emptied before each chunk, so it must have no pending changes, and
    rows read from it earlier are detached.
    """
    for start in range(0, len(ids), CHUNK):
        session.expunge_all()
        part = ids[start : start + CHUNK]
        stmt = select(StationaryPointRow).where(col(StationaryPointRow.id).in_(part))
        found = {point.id: point for point in session.exec(stmt).all()}
        yield from (found[i] for i in part)


def _member(point: StationaryPointRow) -> str:
    """The archive member (the log) that a point was imported from."""
    return (point.calculation.input_provenance or {})["member"]


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


def near_linear_modes(freqs: np.ndarray, logged: np.ndarray) -> tuple[np.ndarray, bool]:
    """Autogeom's sorted modes, without the extra one of a near-linear molecule.

    Gaussian counts 3N-6 modes for a geometry that is linear to the log's precision,
    while autogeom counts 3N-5. If autogeom has one mode more and dropping one of its
    modes matches the log within `FREQUENCY_TOLERANCE`, that mode is dropped.
    """
    if len(freqs) != len(logged) + 1:
        return freqs, False
    dropped = [np.delete(freqs, k) for k in range(len(freqs))]
    diffs = [float(np.max(np.abs(d - logged))) if len(d) else 0.0 for d in dropped]
    if min(diffs) > FREQUENCY_TOLERANCE:
        return freqs, False
    return dropped[int(np.argmin(diffs))], True


def shifted_modes(freqs: np.ndarray, logged: np.ndarray) -> tuple[int, float]:
    """How far a near-linear molecule is off when no single extra mode explains it.

    ``freqs`` (sorted) has one mode more than ``logged``. Of the ways to drop one, this
    takes the one that leaves the fewest modes beyond `FREQUENCY_TOLERANCE` (then the
    smallest largest difference), and returns that number and difference.
    """
    diffs = [np.abs(np.delete(freqs, k) - logged) for k in range(len(freqs))]
    return min(
        (int(np.sum(d > FREQUENCY_TOLERANCE)), float(np.max(d, initial=0.0)))
        for d in diffs
    )


def frequencies_section(session: AutostorageSession) -> str:
    """Frequencies from the stored Hessians against the logs' (every point)."""
    worst = 0.0
    bad: list[tuple[str, float]] = []
    near_linear: list[str] = []
    shifted: list[tuple[str, int, float]] = []
    n = 0
    for point in _points(session, _point_ids(session)):
        hessian = next(
            (
                p.value
                for p in point.geometry.properties
                if p.property_kind_name == "hessian"
            ),
            None,
        )
        member = _member(point)
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
        freqs, is_near_linear = near_linear_modes(
            np.sort(vib.frequencies), np.sort(logged)
        )
        logged = np.sort(logged)
        if is_near_linear:
            near_linear.append(member)
        if len(freqs) == len(logged) + 1:
            shifted.append((member, *shifted_modes(freqs, logged)))
            continue
        if len(freqs) != len(logged):
            bad.append((f"{member}: {len(freqs)} vs {len(logged)} modes", float("nan")))
            continue
        diff = float(np.max(np.abs(freqs - logged))) if len(freqs) else 0.0
        worst = max(worst, diff)
        n += 1
        if diff > FREQUENCY_TOLERANCE:
            bad.append((member, diff))
    lines = [
        "## Frequencies from the stored Hessian",
        (
            f"{n} points compared; worst |Δν| = {worst:.3f} cm⁻¹; "
            f"{len(bad)} beyond {FREQUENCY_TOLERANCE} cm⁻¹ or not comparable; "
            f"{len(near_linear)} near-linear (Gaussian logged 3N-6 modes, autogeom "
            "gives 3N-5; compared without autogeom's extra one); "
            f"{len(shifted)} near-linear with other modes shifted too (no one extra "
            "mode explains the difference; not compared)."
        ),
    ]
    lines += [f"- near-linear: {member}" for member in near_linear]
    lines += [
        f"- near-linear, modes shifted: {member}: {count} beyond "
        f"{FREQUENCY_TOLERANCE} cm⁻¹, largest {diff:.2f}"
        for member, count, diff in shifted
    ]
    lines += [
        f"- {member}: {diff:.2f}"
        for member, diff in sorted(bad, key=lambda b: -b[1] if b[1] == b[1] else 0)[
            :MAX_BAD_FREQUENCIES
        ]
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
    ids = _point_ids(session)
    random.Random(0).shuffle(ids)  # noqa: S311 - a reproducible sample
    ids = ids[:sample]
    members = {_member(p) for p in _points(session, ids)}
    packed = _stjohn_members(members) if dataset == "stjohn" else {}
    worst = {"coordinates": 0.0, "energy": 0.0, "hessian": 0.0}
    n = 0
    for point in _points(session, ids):
        member = _member(point)
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
    stored_layers, expected_layers = stored.split("/"), expected.split("/")
    if "+" in source_smiles or "-]" in source_smiles:
        return "charge separation"
    if stored_layers[1:CONNECTIVITY] == expected_layers[1:CONNECTIVITY]:
        return "stereo or hydrogen layers"
    if stored_layers[1:2] != expected_layers[1:2]:
        return "formula"
    return "connectivity (resonance or other)"


def inchi_section(
    session: AutostorageSession, source_smiles: Callable[[str], str | None]
) -> str:
    """RDKit InChI of each order-0 point against the source SMILES."""
    DisableLog("rdApp.*")
    kinds: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    n = 0
    # Every InChI in one pass: joined to the points of a chunk of ids instead, SQLite
    # picks a plan that is quadratic in the number of points.
    stmt = (
        select(col(IdentityStationaryLink.stationary_id), col(IdentityRow.value))
        .join(
            IdentityRow, col(IdentityRow.id) == col(IdentityStationaryLink.identity_id)
        )
        .where(IdentityRow.algorithm_name == "rdkit_inchi")
    )
    inchis = dict(session.exec(stmt).all())
    ids = [i for i in _point_ids(session, order=0) if i in inchis]
    for point in _points(session, ids):
        stored = inchis[point.id]
        member = _member(point)
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


def energy_channels(energy: dict[str, float]) -> list[str]:
    """The reaction channels whose TS log is among ``energy``'s members."""
    suffix = "-TS.out"
    return [m.rsplit("/", 1)[-1][: -len(suffix)] for m in energy if m.endswith(suffix)]


def _csv_barriers(channels: set[str]) -> dict[str, tuple[float, float]]:
    """DE_F and DE_B (kcal/mol) of the given channels from the RGD1 CSV."""
    path = sources.data_dir() / "rgd1/RGD1CHNO_AMsmiles.csv"
    with path.open(newline="") as handle:
        return {
            row["reaction"]: (float(row["DE_F"]), float(row["DE_B"]))
            for row in csv.DictReader(handle)
            if row["reaction"] in channels
        }


def rgd1_energies_section(session: AutostorageSession, h5_path: Path) -> str:
    """TS and reaction energies against the h5 (R_E, P_E, TS_E)."""
    energy: dict[str, float] = {}
    for point in _points(session, _point_ids(session)):
        value = next(
            p.value
            for p in point.geometry.properties
            if p.property_kind_name == "energy"
        )
        energy[_member(point)] = float(value)
    ts_diffs: list[float] = []
    channels: Counter[str] = Counter()
    other: list[tuple[float, str]] = []
    mapping = sources.rgd1_mapping()
    with h5py.File(h5_path) as h5:
        for member, e_ts in energy.items():
            if member.endswith("-TS.out"):
                channel = member.rsplit("/", 1)[-1][: -len("-TS.out")]
                if channel in h5:
                    ts_diffs.append(abs(e_ts - float(h5[channel]["TS_E"][()])))
        for channel, (reactants, products) in mapping.items():
            keys = [f"raw_output/RP/{mol}.out" for mol in (*reactants, *products)]
            if channel not in h5 or any(k not in energy for k in keys):
                continue
            ours_r = sum(energy[f"raw_output/RP/{mol}.out"] for mol in reactants)
            ours_p = sum(energy[f"raw_output/RP/{mol}.out"] for mol in products)
            h5_r, h5_p = float(h5[channel]["R_E"][()]), float(h5[channel]["P_E"][()])
            straight = max(abs(ours_r - h5_r), abs(ours_p - h5_p))
            swapped = max(abs(ours_r - h5_p), abs(ours_p - h5_r))
            if straight < LOOSE:
                channels["R_E and P_E agree"] += 1
            elif swapped < LOOSE:
                channels["h5 group stored in reverse direction (consistent)"] += 1
            else:
                channels["other"] += 1
                other.append((min(straight, swapped), channel))
    barriers = _csv_barriers(set(energy_channels(energy)))
    csv_diffs = []
    for channel, (de_f, de_b) in barriers.items():
        reactants, products = mapping[channel]
        e_ts = energy[f"raw_output/final-TS/{channel}-TS.out"]
        sides = [
            sum(energy.get(f"raw_output/RP/{mol}.out", np.nan) for mol in mols)
            for mols in (reactants, products)
        ]
        csv_diffs.append(
            max(
                abs((e_ts - sides[0]) * HARTREE_TO_KCAL - de_f),
                abs((e_ts - sides[1]) * HARTREE_TO_KCAL - de_b),
            )
        )
    lines = [
        "## Energies against the h5",
        (
            f"- TS: {len(ts_diffs)} compared, largest |E(log) - TS_E| = "
            f"{max(ts_diffs, default=0.0):.2e} Eh"
        ),
        (
            "- reactant and product sides (sums of the species energies against "
            f"R_E / P_E; agreement within {LOOSE:.0e} Eh), by reaction: "
            f"{dict(channels)}"
        ),
    ]
    lines += [f"  - {d:.2e}: {c}" for d, c in sorted(other, reverse=True)[:5]]
    ok = [d for d in csv_diffs if np.isfinite(d)]  # NaN: a species not imported
    lines.append(
        f"- barriers against the CSV (DE_F, DE_B in kcal/mol): {len(ok)} reactions, "
        f"largest difference {max(ok, default=0.0):.3f} kcal/mol"
    )
    return LINE.join(lines)


def graph_section(session: AutostorageSession, h5_path: Path, sample: int) -> str:
    """The TS graphs: breaking and forming bonds from the mapped SMILES and the h5."""
    ids = _point_ids(session, order=1)
    random.Random(0).shuffle(ids)  # noqa: S311 - a reproducible sample
    channels: list[str] = []
    for point in _points(session, ids):
        if len(channels) >= sample:
            break
        name = _member(point).rsplit("/", 1)[-1]
        if name.endswith("-TS.out"):  # not a false minimum
            channels.append(name.removesuffix("-TS.out"))
    smiles = graphs._channels()  # noqa: SLF001
    agree, disagree = 0, []
    skipped: Counter[str] = Counter()
    with h5py.File(h5_path) as h5:
        for channel in channels:
            group = h5.get(channel)
            if group is None or channel not in smiles:
                skipped["no source"] += 1
                continue
            symbols = [
                autogeom.element.from_key(int(z)).symbol for z in group["elements"][()]
            ]
            try:
                mapped = graphs.mapped_reaction_graph(*smiles[channel], spin=0)
                ends = [
                    protomol.from_geometry(
                        autogeom.Geometry(
                            symbols=symbols,
                            coordinates=group[key][()],
                            charge=0,
                            spin=0,
                        )
                    )
                    for key in ("RG", "PG")
                ]
                from_geometries = protomol.reaction_graph(*ends)
            except Exception as exc:  # noqa: BLE001
                skipped[type(exc).__name__] += 1
                continue
            ours = (set(mapped.breaking), set(mapped.forming))
            theirs = (set(from_geometries.breaking), set(from_geometries.forming))
            if ours == theirs:
                agree += 1
            else:
                disagree.append((channel, ours, theirs))
    lines = [
        "## TS graphs: breaking and forming bonds, mapped SMILES against the h5",
        (
            f"{len(channels)} sampled TSs: {agree} agree, {len(disagree)} disagree, "
            f"{dict(skipped)} not compared"
        ),
    ]
    for channel, ours, theirs in disagree[:MAX_GRAPH_EXAMPLES]:
        lines.append(
            f"- {channel}: SMILES (breaking, forming) {ours}; geometries {theirs}"
        )
    return LINE.join(lines)


def graph_order_section(session: AutostorageSession, sample: int) -> str:
    """Bonds that neither break nor form in a TS graph, against its geometry's bonds."""
    smiles = graphs._channels()  # noqa: SLF001
    ids = _point_ids(session, order=1)
    random.Random(1).shuffle(ids)  # noqa: S311 - a reproducible sample
    checked, bad, skipped = 0, [], 0
    for point in _points(session, ids):
        member = _member(point)
        channel = member.rsplit("/", 1)[-1].removesuffix("-TS.out")
        if channel not in smiles:
            skipped += 1
            continue
        try:
            graph = graphs.mapped_reaction_graph(*smiles[channel], spin=0)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        geometry_bonds = set(protomol.from_geometry(_geometry(point.geometry)).bonds)
        unchanged = set(graph.bonds) - set(graph.breaking) - set(graph.forming)
        checked += 1
        missing = unchanged - geometry_bonds
        if missing:
            bad.append((channel, len(missing), len(unchanged)))
        if checked >= sample:
            break
    lines = [
        "## TS graph atom order: unchanged bonds against the geometry's bonds",
        f"{checked} sampled TSs ({skipped} without a graph): {len(bad)} disagree",
    ]
    lines += [f"- {c}: {m} of {n} unchanged bonds missing" for c, m, n in bad[:10]]
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
    for point in _points(session, _point_ids(session)):
        member = _member(point)
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


def flags_section(out: Path) -> str:
    """The TSs flagged at import (``flag:*`` rows of the failures file)."""
    path = Path(f"{out}.failures.csv")
    if not path.exists():
        return "## TS graph flags\nno failures file"
    with path.open(newline="") as handle:
        flagged = [r for r in csv.DictReader(handle) if r["reason"].startswith("flag:")]
    by_flag = Counter(r["reason"] for r in flagged)
    lines = [
        "## TS graph flags (imported all the same)",
        f"{len(flagged)} flags: {dict(by_flag)}",
        f"(the first {MAX_GRAPH_EXAMPLES} of each; all are in the failures file)",
    ]
    shown: Counter[str] = Counter()
    for row in flagged:
        shown[row["reason"]] += 1
        if shown[row["reason"]] <= MAX_GRAPH_EXAMPLES:
            lines.append(f"- {row['member']}: {row['reason']}: {row['detail']}")
    return LINE.join(lines)


def _sections(session: AutostorageSession, args: argparse.Namespace) -> Iterator[str]:
    """The sections of the report, each made when it is asked for."""
    yield counts_section(session, args.db)
    yield round_trip_section(session, args.dataset, args.sample)
    yield frequencies_section(session)
    if args.dataset == "rgd1":
        smiles = {}
        for line in (
            sources.rgd1_zip_read("raw_output/RandP_smiles.txt")
            .decode()
            .splitlines()[1:]
        ):
            text, _, mol = line.rpartition("   ")
            smiles[f"raw_output/RP/{mol}.out"] = text
        yield inchi_section(session, smiles.get)
        h5_path = sources.data_dir() / "rgd1/RGD1_allrxns.h5"
        yield rgd1_energies_section(session, h5_path)
        if args.graphs:
            yield graph_section(session, h5_path, args.sample)
            yield graph_order_section(session, args.sample)
            yield flags_section(args.db)
    else:
        sdf = stjohn_sdf()

        def by_member(member: str) -> str | None:
            record = sdf.get(Path(member).name.removesuffix(".log.gz"))
            return record[0] if record else None

        yield inchi_section(session, by_member)
        yield stjohn_energies_section(session, sdf)
    yield size_section(args.db, session)


def main(argv: list[str] | None = None) -> None:
    """Print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["rgd1", "stjohn"])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--graphs", action="store_true", help="check the TS graphs")
    args = parser.parse_args(argv)
    with Database(args.db) as db, db.session() as session:
        for section in _sections(session, args):
            print(f"{section}{LINE}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
