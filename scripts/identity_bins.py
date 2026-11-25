# ruff: noqa: INP001  (a script, not a package module)
"""Compare the identity bins of an imported database, and an earlier one.

    pixi run -e test python scripts/identity_bins.py NEW.db [--old OLD.db]

Minima (order 0): points and distinct values per algorithm; for pairs of algorithms,
the bins of one split by the other (a finer scheme splits a coarser one) and merged by
it. A class that `smg_low_t@1` splits is put in a category by which kind of LS tells its
members apart (removing those LS makes their canonical JSON equal): a stereocentre (four
ligands), an amine or other lone-pair centre (N, P, S, ...), a radical centre, a bond
LS. With `--old`, the
minima are matched by geometry hash, and the old `protomol_graph@0` bins (the graph as
perceived before the low-T rules) are compared with the new `smg_low_t@1` bins.

Saddle points: TSs that share a graph identity though imported in opposite directions
(`algorithm_cache["reversed"]`, pm P58), and self-reaction steps. Read-only (SQLite).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import protomol

MINIMUM_PAIRS = (
    ("rdkit_inchi_conn", "smg_conn@1"),
    ("rdkit_inchi", "smg_low_t@1"),
    ("smg_conn@1", "smg_low_t@1"),
    ("rdkit_inchi", "protomol_graph@0"),
)


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load(path: Path) -> dict[str, Any]:
    """Points (id: order, geometry hash) and links (algorithm: point: value, cache)."""
    con = _connect(path)
    cols = {
        row[1] for row in con.execute("pragma table_info(identity_stationary_link)")
    }
    cache_col = "algorithm_cache" if "algorithm_cache" in cols else "identity_cache"
    points = {
        pid: (order, ghash)
        for pid, order, ghash in con.execute(
            'select s.id, s."order", g.hash from stationary_point s '
            "join geometry g on g.id = s.geometry_id"
        )
    }
    links: dict[str, dict[str, tuple[str, dict[str, Any] | None]]] = defaultdict(dict)
    for pid, alg, value, blob in con.execute(
        f"select l.stationary_id, i.algorithm_name, i.value, l.{cache_col} "  # noqa: S608
        "from identity_stationary_link l join identity i on i.id = l.identity_id"
    ):
        cache = json.loads(zlib.decompress(blob)) if blob else None
        links[alg][pid] = (value, cache)
    steps = con.execute("select stage_id1, stage_id2 from step").fetchall()
    return {"points": points, "links": links, "steps": steps}


def _bins(values: dict[str, str]) -> dict[str, set[str]]:
    bins: dict[str, set[str]] = defaultdict(set)
    for pid, value in values.items():
        bins[value].add(pid)
    return bins


def _values(db: dict[str, Any], alg: str, order: int) -> dict[str, str]:
    return {
        pid: value
        for pid, (value, _) in db["links"].get(alg, {}).items()
        if db["points"][pid][0] == order
    }


def split_counts(
    coarse: dict[str, str], fine: dict[str, str]
) -> tuple[list[set[str]], list[set[str]]]:
    """Coarse bins split by `fine`, and fine bins merged by `coarse` (as point sets)."""
    shared = coarse.keys() & fine.keys()
    split = [
        pids
        for pids in _bins({p: coarse[p] for p in shared}).values()
        if len({fine[p] for p in pids}) > 1
    ]
    merged = [
        pids
        for pids in _bins({p: fine[p] for p in shared}).values()
        if len({coarse[p] for p in pids}) > 1
    ]
    return split, merged


def _without(gra: protomol.StereoMolGraph, which: str) -> protomol.StereoMolGraph:
    """`gra` without the LS of one category."""
    for ls in (*gra.atom_stereo, *gra.bond_stereo):
        is_atom = isinstance(ls, protomol.Tetrahedral)
        lone = is_atom and None in ls.atoms
        symbol = gra.symbols[ls.atoms[0]] if is_atom else None
        drop = (
            (which == "stereocentre" and is_atom and not lone)
            or (which == "lone-pair centre" and lone and symbol != "C")
            or (which == "radical centre" and lone and symbol == "C")
            or (which == "bond LS" and not is_atom)
        )
        if drop:
            gra = protomol.remove_stereo(gra, protomol.site(ls))
    return gra


CATEGORIES = ("stereocentre", "lone-pair centre", "radical centre", "bond LS")


def split_category(values: list[str]) -> str:
    """Which LS kinds tell the canonical graphs apart (the smallest set that does)."""
    graphs = []
    for value in values:
        gra = protomol.from_json(value)
        assert isinstance(gra, protomol.StereoMolGraph)  # noqa: S101
        graphs.append(gra)
    for size in range(1, len(CATEGORIES) + 1):
        for combo in _combinations(CATEGORIES, size):
            reduced = graphs
            for which in combo:
                reduced = [_without(g, which) for g in reduced]
            if len({protomol.canonical_json(g) for g in reduced}) == 1:
                return " + ".join(combo)
    return "other"


def _combinations(items: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return list(itertools.combinations(items, size))


def minima_section(db: dict[str, Any], examples: int) -> list[str]:
    """The minima's bins per algorithm and per pair."""
    lines = ["## Minima"]
    n = sum(1 for order, _ in db["points"].values() if order == 0)
    lines.append(f"{n} minima")
    for alg in sorted(db["links"]):
        vals = _values(db, alg, 0)
        if vals:
            lines.append(
                f"- `{alg}`: {len(vals)} points, {len(set(vals.values()))} bins"
            )
    for coarse, fine in MINIMUM_PAIRS:
        a, b = _values(db, coarse, 0), _values(db, fine, 0)
        if not a or not b:
            continue
        split, merged = split_counts(a, b)
        lines.append(
            f"- `{coarse}` bins split by `{fine}`: {len(split)} "
            f"({sum(len(s) for s in split)} points); `{fine}` bins merged by "
            f"`{coarse}`: {len(merged)} ({sum(len(s) for s in merged)} points)"
        )
        if fine == "smg_low_t@1" and split:
            cats = Counter(split_category(sorted({b[p] for p in s})) for s in split)
            lines.append(f"  - splits by category: {dict(cats.most_common())}")
        lines.extend(
            f"  - merged: {sorted({a[p] for p in pids})}" for pids in merged[:examples]
        )
    return lines


def compare_section(
    new: dict[str, Any], old: dict[str, Any], examples: int
) -> list[str]:
    """Old `protomol_graph@0` minima bins against new `smg_low_t@1` ones."""
    lines = ["## Against the earlier database (minima matched by geometry hash)"]
    old_by_hash = {
        ghash: value
        for pid, (value, _) in old["links"].get("protomol_graph@0", {}).items()
        if old["points"][pid][0] == 0
        for ghash in [old["points"][pid][1]]
    }
    new_by_hash = {
        new["points"][pid][1]: value
        for pid, value in _values(new, "smg_low_t@1", 0).items()
    }
    shared = old_by_hash.keys() & new_by_hash.keys()
    lines.append(
        f"{len(shared)} minima in both ({len(old_by_hash)} old, {len(new_by_hash)} new)"
    )
    old_v = {h: old_by_hash[h] for h in shared}
    new_v = {h: new_by_hash[h] for h in shared}
    split, merged = split_counts(old_v, new_v)
    lines.append(
        f"- old `protomol_graph@0` bins split by new `smg_low_t@1`: {len(split)}; "
        f"new bins merged by the old: {len(merged)}"
    )
    changed = sum(1 for h in shared if old_v[h] != new_v[h])
    lines.append(f"- canonical graph changed: {changed} of {len(shared)}")
    lines.extend(
        f"  - e.g. {len(hashes)} points: {sorted(hashes)[:3]}"
        for hashes in (split + merged)[:examples]
    )
    return lines


def saddle_section(db: dict[str, Any]) -> list[str]:
    """TSs merged by canonical direction, and self-reaction steps."""
    lines = ["## Saddle points"]
    for alg in ("protomol_graph@0", "smg_conn@1"):
        links = {
            pid: (value, cache)
            for pid, (value, cache) in db["links"].get(alg, {}).items()
            if db["points"][pid][0] >= 1
        }
        if not links:
            continue
        bins: dict[str, list[bool | None]] = defaultdict(list)
        for value, cache in links.values():
            bins[value].append(None if cache is None else cache.get("reversed"))
        both = [flags for flags in bins.values() if {True, False} <= set(flags)]
        n_rev = sum(1 for _, c in links.values() if c and c.get("reversed"))
        lines.append(
            f"- `{alg}`: {len(links)} saddle points, {len(bins)} bins, "
            f"{n_rev} reversed; {len(both)} bins hold both directions "
            f"({sum(len(f) for f in both)} points); "
            f"{sum(1 for f in bins.values() if len(f) > 1)} bins with more than one "
            "point"
        )
    selfs = sum(1 for s1, s2 in db["steps"] if s1 == s2)
    lines.append(f"- steps: {len(db['steps'])}, self-reaction steps: {selfs}")
    return lines


def main() -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    new = load(args.db)
    lines = [f"# Identity bins: {args.db.name}", ""]
    lines += [*minima_section(new, args.examples), ""]
    if args.old:
        lines += [*compare_section(new, load(args.old), args.examples), ""]
    lines += saddle_section(new)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
