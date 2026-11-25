"""Reactions: stages shared between steps, A + A, and each reason to skip."""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytest
from autostorage import Database, StageRow, StepRow
from sqlmodel import select

from hifithermkin.reactions import Entry, add_reaction, member_of_species, member_of_ts
from hifithermkin.store import Store

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from autostorage import AutostorageSession

    from hifithermkin.parse import ParsedLog


def _fake(base: ParsedLog, symbols: list[str], nimag: int = 0) -> ParsedLog:
    """A log of ``symbols`` on a line; ``nimag`` is only recorded."""
    n = len(symbols)
    coordinates = np.array([[1.5 * k, 0.1 * k * k, 0.0] for k in range(n)])
    # a distinct geometry for each call: the first atom moves a little
    coordinates[0, 2] = 0.01 * next(_COUNTER)
    diagonal = np.full(3 * n, 0.5)  # the Hessian itself is not under test
    return dataclasses.replace(
        base,
        symbols=symbols,
        coordinates=coordinates,
        hessian=np.diag(diagonal),
        nimag=nimag,
        charge=0,
        multiplicity=1,
    )


def _counter() -> Iterator[int]:
    n = 0
    while True:
        n += 1
        yield n


_COUNTER = _counter()


pytestmark = [
    pytest.mark.filterwarnings("ignore::autostorage.StationaryOrderWarning"),
    pytest.mark.filterwarnings("ignore::autostorage.IdentityGenerationWarning"),
]


class World:
    """A session with a few species and TSs of ammonia chemistry."""

    def __init__(self, session: AutostorageSession, base: ParsedLog) -> None:  # noqa: D107
        self.session = session
        self.store = Store(session)
        self.base = base
        self.index: dict[str, Entry] = {}
        self.counts: Counter[str] = Counter()

    def add(self, member: str, symbols: list[str], nimag: int = 0) -> None:
        """Store a fake log under a member name and index it."""
        point = self.store.add(_fake(self.base, symbols, nimag))
        self.index[member] = Entry("ok", str(point.id), nimag)

    def species(self, mol: str, symbols: list[str], nimag: int = 0) -> None:
        """Add a species log."""
        self.add(member_of_species(mol), symbols, nimag)

    def ts(self, channel: str, symbols: list[str]) -> None:
        """Add a TS log."""
        self.add(member_of_ts(channel), symbols, 1)

    def react(self, channel: str, reactants: list[str], products: list[str]) -> str:
        """Add a reaction."""
        return add_reaction(
            self.session, channel, (reactants, products), self.index, self.counts
        )


NH3 = ["N", "H", "H", "H"]
N2H4 = ["N", "N", "H", "H", "H", "H"]
H2 = ["H", "H"]
N2H6 = ["N", "N", "H", "H", "H", "H", "H", "H"]


@pytest.fixture
def world(tmp_path: Path, logs: dict[str, ParsedLog]) -> Iterator[World]:
    with Database(tmp_path / "r.db") as db, db.session() as session:
        w = World(session, logs["rgd1_nh3_minimum"])
        w.species("mol1", NH3)
        w.species("mol2", N2H4)
        w.species("mol3", H2)
        w.ts("MR_1_0", N2H6)
        w.ts("MR_1_1", N2H6)
        session.commit()
        yield w


def test_a_plus_a_is_one_point_with_count_two(world: World) -> None:
    assert world.react("MR_1_0", ["mol1", "mol1"], ["mol2", "mol3"]) == "ok"
    world.session.commit()
    assert world.counts["steps"] == 1
    assert world.counts["sides_with_repeats"] == 1
    reactant = next(
        s
        for s in world.session.exec(select(StageRow)).all()
        if not s.is_ts and len(s.stationary_links) == 1
    )
    assert reactant.stationary_links[0].count == 2


def test_stages_are_shared_between_steps(world: World) -> None:
    world.react("MR_1_0", ["mol1", "mol1"], ["mol2", "mol3"])
    world.react("MR_1_1", ["mol1", "mol1"], ["mol2", "mol3"])
    world.session.commit()
    stages = world.session.exec(select(StageRow)).all()
    assert len(stages) == 4  # one reactant, one product, two TS stages
    assert len(world.session.exec(select(StepRow)).all()) == 2


def test_unbalanced_reaction_is_skipped_and_leaves_nothing(world: World) -> None:
    assert world.react("MR_1_0", ["mol1"], ["mol3"]) == "unbalanced"
    assert world.counts["skipped_unbalanced"] == 1
    world.session.commit()  # the failed step must not poison the next commit
    assert world.session.exec(select(StepRow)).all() == []
    assert world.session.exec(select(StageRow)).all() == []
    assert world.react("MR_1_1", ["mol1", "mol1"], ["mol2", "mol3"]) == "ok"
    world.session.commit()


def test_false_minimum_species_is_not_used(world: World) -> None:
    world.species("mol4", NH3, nimag=1)
    assert (
        world.react("MR_1_0", ["mol4", "mol1"], ["mol2", "mol3"])
        == "species_false_minimum"
    )
    assert world.counts["skipped_species_false_minimum"] == 1


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (None, "ts_not_imported"),
        (Entry("broken_hessian"), "ts_broken_hessian"),
    ],
)
def test_missing_ts(world: World, entry: Entry | None, reason: str) -> None:
    del world.index[member_of_ts("MR_1_0")]
    if entry is not None:
        world.index[member_of_ts("MR_1_0")] = entry
    assert world.react("MR_1_0", ["mol1", "mol1"], ["mol2", "mol3"]) == reason


def test_missing_species(world: World) -> None:
    world.index[member_of_species("mol3")] = Entry("parse_error")
    reason = world.react("MR_1_0", ["mol1", "mol1"], ["mol2", "mol3"])
    assert reason == "species_parse_error"
    assert world.react("MR_1_0", ["mol1", "mol9"], ["mol2"]) == "species_not_imported"
