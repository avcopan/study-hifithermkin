"""Import RGD1 and stjohn-radicals into one AutoStorage database per dataset.

    python -m hifithermkin.importer rgd1 --subset --out ~/code/am/data/db/rgd1-subset.db

Logs are read and parsed in worker processes; this process writes, in batches. Beside
the database ``X.db`` it keeps, all appended to as the run goes:

- ``X.db.checkpoint.tsv``: one line per finished log (member, status, point id, order),
  written after the commit that holds it. A rerun with the same ``--out`` skips these
  members. (A kill between a commit and its checkpoint lines repeats at most one batch,
  which adds a second calculation on the same geometry.)
- ``X.db.failures.csv``: every skipped log and its reason (not ``outside_subset``).
- ``X.db.reactions.tsv`` (RGD1): one line per reaction, ``ok`` or the reason it was
  skipped.
- ``X.db.summary.json``: counts and timings, rewritten at the end of each phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autostorage import Database

from . import sources
from .parse import ParsedLog, Skip
from .reactions import Entry, add_reaction
from .store import GraphFn, Store

if TYPE_CHECKING:
    from collections.abc import Iterator

    from autostorage import AutostorageSession

log = logging.getLogger("hifithermkin")

PROGRESS_EVERY = 2000  # logs
DEFAULT_BATCH = 300  # logs per commit
MAX_PENDING_LINES = 5000  # checkpoint lines held back, skipped logs included
REACTION_BATCH = 1000  # reactions per commit


def default_out(dataset: str, *, subset: bool) -> Path:
    """The database path of a run."""
    tag = "-subset" if subset else ""
    return sources.data_dir() / "db" / f"{dataset}{tag}-{datetime.now(UTC).date()}.db"


def _read_checkpoint(path: Path) -> dict[str, Entry]:
    index: dict[str, Entry] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                member, status, point_id, order = line.rstrip("\n").split("\t")
                index[member] = Entry(
                    status, point_id or None, int(order) if order else None
                )
    return index


def _jobs(
    dataset: str,
    pool: ProcessPoolExecutor,
    done: set[str],
    *,
    subset: bool,
    limit: int | None,
) -> Iterator[Future[ParsedLog | Skip]]:
    """Submit every log not done to the pool."""
    if dataset == "stjohn":
        for member, packed in sources.stjohn_members(done, limit):
            yield pool.submit(sources.read_stjohn, member, packed, subset=subset)
        return
    mapping = sources.rgd1_mapping()
    species = sources.rgd1_subset_species() if subset else None
    members = sources.rgd1_members(mapping, subset=species)
    for count, member in enumerate(members):
        if limit is not None and count >= limit:
            return
        if member not in done:
            yield pool.submit(sources.read_rgd1, member, subset=False)


class _Writer:
    """Takes parsed logs in the main process: stores them, commits in batches."""

    def __init__(
        self,
        dataset: str,
        session: AutostorageSession,
        out: Path,
        *,
        batch: int,
        graph_fn: GraphFn | None,
    ) -> None:
        self.dataset = dataset
        self.session = session
        self.batch = batch
        self.store = Store(session, graph_fn=graph_fn)
        self.counts: Counter[str] = Counter()
        self.started = time.perf_counter()
        self.lines: list[str] = []
        self.in_batch = 0
        self.marks = Path(f"{out}.checkpoint.tsv").open("a")  # noqa: SIM115
        path = Path(f"{out}.failures.csv")
        new = not path.exists()
        self.failures_file = path.open("a", newline="")
        self.failures = csv.writer(self.failures_file)
        if new:
            self.failures.writerow(["member", "reason", "detail"])

    def close(self) -> None:
        self.marks.close()
        self.failures_file.close()

    def commit(self) -> None:
        self.session.commit()
        self.store.committed()
        self.marks.writelines(self.lines)
        self.marks.flush()
        self.failures_file.flush()
        self.lines = []
        self.in_batch = 0

    def take(self, result: ParsedLog | Skip) -> None:
        if isinstance(result, Skip):
            self.counts[f"skipped_{result.reason}"] += 1
            if result.reason != "outside_subset":
                self.failures.writerow([result.member, result.reason, result.detail])
            self.lines.append(f"{result.member}\t{result.reason}\t\t\n")
        else:
            point = self.store.add(result)
            self.lines.append(f"{result.member}\tok\t{point.id}\t{result.nimag}\n")
            self.in_batch += 1
        self.counts["logs_seen"] += 1
        if self.counts["logs_seen"] % PROGRESS_EVERY == 0:
            rate = self.counts["logs_seen"] / (time.perf_counter() - self.started)
            log.info(
                "%s: %d logs (%.1f/s), %s",
                self.dataset,
                self.counts["logs_seen"],
                rate,
                dict(self.store.counts),
            )
        if self.in_batch >= self.batch or len(self.lines) >= MAX_PENDING_LINES:
            self.commit()


def run_logs(  # noqa: PLR0913
    dataset: str,
    out: Path,
    *,
    subset: bool,
    workers: int,
    batch: int = DEFAULT_BATCH,
    limit: int | None = None,
    graph_fn: GraphFn | None = None,
) -> Counter[str]:
    """Import the logs of a dataset; returns the counts of this run."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(_read_checkpoint(Path(f"{out}.checkpoint.tsv")))
    log.info("%s: %d logs already done", dataset, len(done))
    with Database(out) as db, db.session() as session:
        writer = _Writer(dataset, session, out, batch=batch, graph_fn=graph_fn)
        try:
            with ProcessPoolExecutor(workers) as pool:
                pending: set[Future[ParsedLog | Skip]] = set()
                for future in _jobs(dataset, pool, done, subset=subset, limit=limit):
                    pending.add(future)
                    while len(pending) >= 2 * workers:
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for item in finished:
                            writer.take(item.result())
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for item in finished:
                        writer.take(item.result())
            writer.commit()
        finally:
            writer.close()
    counts = writer.counts + writer.store.counts
    counts["seconds"] = round(time.perf_counter() - writer.started)
    _write_summary(out, "logs", counts, writer.store.unclassified)
    return counts


def run_reactions(out: Path, *, subset: bool, limit: int | None = None) -> Counter[str]:
    """Add the stages and steps of the RGD1 reactions whose logs were imported."""
    index = _read_checkpoint(Path(f"{out}.checkpoint.tsv"))
    mapping = sources.rgd1_mapping()
    species = sources.rgd1_subset_species() if subset else None
    marks_path = Path(f"{out}.reactions.tsv")
    done: set[str] = set()
    if marks_path.exists():
        done = {line.split("\t")[0] for line in marks_path.read_text().splitlines()}
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    with Database(out) as db, db.session() as session, marks_path.open("a") as marks:
        lines: list[str] = []

        def commit() -> None:
            nonlocal lines
            session.commit()
            marks.writelines(lines)
            marks.flush()
            lines = []

        for channel, sides in mapping.items():
            if channel in done:
                continue
            if species is not None and not all(
                m in species for m in (*sides[0], *sides[1])
            ):
                continue  # outside the subset: not a reaction of this run
            if limit is not None and counts["reactions_seen"] >= limit:
                break
            result = add_reaction(session, channel, sides, index, counts)
            lines.append(f"{channel}\t{result}\n")
            counts["reactions_seen"] += 1
            if counts["reactions_seen"] % REACTION_BATCH == 0:
                commit()
                rate = counts["reactions_seen"] / (time.perf_counter() - started)
                log.info(
                    "reactions: %d (%.1f/s), %s",
                    counts["reactions_seen"],
                    rate,
                    dict(counts),
                )
        commit()
    counts["seconds"] = round(time.perf_counter() - started)
    _write_summary(out, "reactions", counts, Counter())
    return counts


def _write_summary(
    out: Path, phase: str, counts: Counter[str], tokens: Counter[str]
) -> None:
    path = Path(f"{out}.summary.json")
    summary: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    summary[phase] = {"counts": dict(counts), "unclassified_tokens": dict(tokens)}
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info("%s done: %s", phase, dict(counts))


def main(argv: list[str] | None = None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["rgd1", "stjohn"])
    parser.add_argument("--subset", action="store_true", help="C/H/O, <= 4 heavy atoms")
    parser.add_argument("--out", type=Path, help="database (default: data/db/...)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--limit", type=int, help="logs to read (testing)")
    parser.add_argument("--no-reactions", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr
    )
    out = (args.out or default_out(args.dataset, subset=args.subset)).expanduser()
    run_logs(
        args.dataset,
        out,
        subset=args.subset,
        workers=args.workers,
        batch=args.batch,
        limit=args.limit,
    )
    if args.dataset == "rgd1" and not args.no_reactions:
        run_reactions(out, subset=args.subset, limit=args.limit)


if __name__ == "__main__":
    main()
