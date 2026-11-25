"""RGD1 reactions: stages and steps from the imported points."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from autostorage import StageRow, StationaryPointRow, StepRow
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from collections import Counter

    from autostorage import AutostorageSession

__all__ = ["Entry", "add_reaction", "member_of_species", "member_of_ts"]


@dataclass(frozen=True)
class Entry:
    """What the checkpoint knows of one log: its status and, if imported, its point."""

    status: str  # ok, or the reason it was skipped
    point_id: str | None = None
    order: int | None = None


def member_of_species(mol: str) -> str:
    """The zip member of a species."""
    return f"raw_output/RP/{mol}.out"


def member_of_ts(channel: str) -> str:
    """The zip member of a reaction's transition state."""
    return f"raw_output/final-TS/{channel}-TS.out"


def _why_missing(entry: Entry | None, label: str) -> str | None:
    """The reason a log gives no usable point, or ``None`` if it gives one."""
    if entry is None:
        return f"{label}_not_imported"
    if entry.status != "ok":
        return f"{label}_{entry.status}"
    if label == "species" and entry.order != 0:
        return "species_false_minimum"
    return None


def _point(session: AutostorageSession, entry: Entry) -> StationaryPointRow:
    """The stored point of a checkpoint entry."""
    point = session.get(StationaryPointRow, uuid.UUID(entry.point_id or ""))
    if point is None:
        msg = f"checkpoint point {entry.point_id} is not in the database"
        raise LookupError(msg)
    return point


def add_reaction(
    session: AutostorageSession,
    channel: str,
    sides: tuple[list[str], list[str]],
    index: dict[str, Entry],
    counts: Counter[str],
) -> str:
    """Add the stages and the step of one reaction, or say why not.

    Returns:
        ``"ok"``, or the reason the reaction was skipped (counted in ``counts``).
    """
    reactants, products = sides
    entries = {member_of_ts(channel): "ts"}
    entries |= {member_of_species(m): "species" for m in (*reactants, *products)}
    for member, label in entries.items():
        reason = _why_missing(index.get(member), label)
        if reason is not None:
            counts[f"skipped_{reason}"] += 1
            return reason
    repeated = [len(set(side)) < len(side) for side in sides]
    pending = set(session.new)
    try:
        with session.begin_nested():
            points = {
                m: _point(session, index[member_of_species(m)])
                for m in {*reactants, *products}
            }
            ts = _point(session, index[member_of_ts(channel)])
            stage1 = StageRow.get_or_create(session, [points[m] for m in reactants])
            stage2 = StageRow.get_or_create(session, [points[m] for m in products])
            stage_ts = StageRow.get_or_create(session, [ts], is_ts=True)
            session.add(StepRow(stage1=stage1, stage2=stage2, stage_ts=stage_ts))
            session.flush()
    except (ValueError, IntegrityError) as exc:
        # The rollback expunges what the savepoint flushed, but not what a failing
        # flush left pending (the step and its new stages).
        for obj in set(session.new) - pending:
            session.expunge(obj)
        # an unbalanced step, or a malformed stage / a step that exists already
        reason = "unbalanced" if isinstance(exc, ValueError) else "duplicate_step"
        counts[f"skipped_{reason}"] += 1
        return reason
    counts["steps"] += 1
    counts["sides_with_repeats"] += sum(repeated)
    return "ok"
