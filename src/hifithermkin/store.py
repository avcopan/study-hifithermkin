"""Write parsed logs into an AutoStorage database."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    GeometryRow,
    ModelRow,
    PropertyValueRow,
    Role,
    StationaryPointRow,
    energy_property_kind,
    geometry_hash,
    hessian_property_kind,
)
from sqlmodel import col, select

if TYPE_CHECKING:
    from collections.abc import Callable

    from autostorage import AutostorageSession

    from .parse import ParsedLog
    from .routes import ModelSpec

__all__ = ["Store"]

type GraphFn = Callable[[ParsedLog, GeometryRow], Any]
"""The graph to pass as a point's ``graph=``; ``None`` lets it be perceived."""


class Store:
    """Adds one log at a time to a session, caching models and reusing geometries.

    Attributes:
        counts: What happened, by name (``geometries_reused``, ``models_created``,
            ``points_order_0``, ...).
        unclassified: Route tokens that no rule classified, with their counts.
    """

    def __init__(
        self, session: AutostorageSession, *, graph_fn: GraphFn | None = None
    ) -> None:
        """Create a store that writes to ``session`` (it never commits)."""
        self.session = session
        self.graph_fn = graph_fn
        self.counts: Counter[str] = Counter()
        self.unclassified: Counter[str] = Counter()
        self._models: dict[tuple, ModelRow] = {}
        self._geometries: dict[str, GeometryRow] = {}  # pending, until committed

    def model(self, spec: ModelSpec) -> ModelRow:
        """The row of a model, created on first use (``ModelRow`` is not unique)."""
        row = self._models.get(spec.key)
        if row is None:
            row = ModelRow(
                program=spec.program,
                program_version=spec.program_version,
                method=spec.method,
                basis=spec.basis,
                keywords=dict(spec.keywords),
            )
            self.session.add(row)
            self._models[spec.key] = row
            self.counts["models_created"] += 1
        return row

    def geometry(self, log: ParsedLog) -> GeometryRow:
        """The row of the log's geometry; an equal one that exists is reused."""
        row = GeometryRow(
            symbols=log.symbols,
            coordinates=log.coordinates,
            charge=log.charge,
            spin=log.spin,
        )
        value = geometry_hash(row)
        found = self._geometries.get(value)
        if found is None:
            with self.session.no_autoflush:
                found = self.session.exec(
                    select(GeometryRow).where(col(GeometryRow.hash) == value)
                ).first()
        if found is not None:
            self.counts["geometries_reused"] += 1
            return found
        self.session.add(row)
        self._geometries[value] = row
        self.counts["geometries_created"] += 1
        return row

    def add(self, log: ParsedLog) -> StationaryPointRow:
        """Add a log's geometry, calculation, properties and stationary point."""
        self.unclassified.update(log.unclassified)
        geometry = self.geometry(log)
        calculation = CalculationRow(
            model=self.model(log.model),
            calc_type=log.calc_type,
            input_provenance={
                "dataset": log.dataset,
                "archive": log.archive,
                "member": log.member,
                "md5": log.md5,
                "routes": log.routes,
            },
            output_provenance=log.output,
        )
        self.session.add(calculation)
        self.session.add(
            CalculationGeometryLink(
                geometry=geometry, calculation=calculation, role=Role.OUTPUT
            )
        )
        for kind, value in (
            (energy_property_kind, log.energy),
            (hessian_property_kind, log.hessian),
        ):
            self.session.add(
                PropertyValueRow(
                    property_kind_name=kind.name,
                    geometry=geometry,
                    calculation=calculation,
                    value=value,
                )
            )
        graph = self.graph_fn(log, geometry) if self.graph_fn else None
        extra = {} if graph is None else {"graph": graph}
        point = StationaryPointRow(
            geometry=geometry, calculation=calculation, order=log.nimag, **extra
        )
        self.session.add(point)
        self.counts[f"points_order_{log.nimag}"] += 1
        self.counts["logs"] += 1
        return point

    def committed(self) -> None:
        """Tell the store that its session committed: pending geometries are stored."""
        self._geometries.clear()
