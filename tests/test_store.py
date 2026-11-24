"""Writing logs into a temporary AutoStorage database."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from autostorage import (
    CalculationRow,
    Database,
    GeometryRow,
    ModelRow,
    PropertyValueRow,
    StationaryPointRow,
)
from sqlmodel import select

from hifithermkin.store import Store

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from hifithermkin.parse import ParsedLog


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    db = Database(tmp_path / "test.db")
    yield db
    db.close()


def test_import_fixture_logs(database: Database, logs: dict[str, ParsedLog]) -> None:
    with database.session() as session:
        store = Store(session)
        for log in logs.values():
            store.add(log)
        session.commit()
        assert len(session.exec(select(CalculationRow)).all()) == 6
        assert len(session.exec(select(GeometryRow)).all()) == 6
        assert len(session.exec(select(PropertyValueRow)).all()) == 12
        points = session.exec(select(StationaryPointRow)).all()
        assert sorted(p.order for p in points) == [0, 0, 0, 0, 0, 1]
        assert all(p.is_validated for p in points)
    # RGD1 minima and TS share a model; the stjohn radicals differ by `stable=opt`
    # and the Gaussian revision
    assert store.counts["models_created"] == 4
    assert store.counts["points_order_1"] == 1
    assert store.counts["geometries_reused"] == 0


def test_models_hash_alike_for_both_references(
    database: Database, logs: dict[str, ParsedLog]
) -> None:
    with database.session() as session:
        store = Store(session)
        closed = store.model(logs["stjohn_closed_shell_rm062x"].model)
        radical = store.model(logs["stjohn_radical_um062x"].model)
        session.commit()
        names = {
            m.id: sorted(
                link.hash.value
                for link in m.hash_links
                if link.hash.algorithm_name == "minimal@1"
            )
            for m in (closed, radical)
        }
        assert names[closed.id] == names[radical.id] == ["m06-2x/def2-tzvp"]
        assert closed.keywords == {"reference": "rks"}


def test_model_cache_reuses_rows(
    database: Database, logs: dict[str, ParsedLog]
) -> None:
    with database.session() as session:
        store = Store(session)
        a = store.model(logs["rgd1_nh3_minimum"].model)
        b = store.model(logs["rgd1_ts_MR_299320_1"].model)
        session.commit()
        assert a is b
        assert len(session.exec(select(ModelRow)).all()) == 1


def test_equal_geometry_is_reused_and_counted(
    database: Database, logs: dict[str, ParsedLog]
) -> None:
    log = logs["rgd1_nh3_minimum"]
    with database.session() as session:
        store = Store(session)
        first = store.add(log)
        second = store.add(log)  # pending reuse
        session.commit()
        store.committed()
        third = store.add(log)  # stored reuse
        session.commit()
        assert first.geometry_id == second.geometry_id == third.geometry_id
        assert store.counts["geometries_reused"] == 2
        assert len(session.exec(select(GeometryRow)).all()) == 1
        assert len(session.exec(select(CalculationRow)).all()) == 3


def test_round_trip(database: Database, logs: dict[str, ParsedLog]) -> None:
    log = logs["stjohn_radical_um062x"]
    with database.session() as session:
        Store(session).add(log)
        session.commit()
    with database.session() as session:
        geometry = session.exec(select(GeometryRow)).one()
        assert geometry.symbols == log.symbols
        assert (geometry.charge, geometry.spin) == (0, 1)
        np.testing.assert_allclose(geometry.coordinates, log.coordinates)
        values = {
            p.property_kind_name: p.value
            for p in session.exec(select(PropertyValueRow))
        }
        assert float(values["energy"]) == log.energy
        np.testing.assert_allclose(values["hessian"], log.hessian, rtol=1e-6, atol=1e-8)
        calculation = session.exec(select(CalculationRow)).one()
        assert (calculation.input_provenance or {})["md5"] == log.md5
        assert (calculation.output_provenance or {})["nimag"] == 0
        (job,) = (calculation.input_provenance or {})["input"]
        assert job["charge"] == 0
        assert job["multiplicity"] == 2
