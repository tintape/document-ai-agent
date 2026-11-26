from datetime import date
from pathlib import Path

import pytest

from doc_agent.budget import BudgetExceeded, BudgetStore, check_budget
from doc_agent.config import Settings

DAY = date(2026, 11, 27)


def test_new_store_has_zero(tmp_path: Path) -> None:
    assert BudgetStore(tmp_path / "b.json").spent_on(DAY) == 0.0


def test_add_accumulates_same_day(tmp_path: Path) -> None:
    store = BudgetStore(tmp_path / "b.json")
    store.add(0.2, DAY)
    assert store.add(0.3, DAY) == pytest.approx(0.5)
    assert BudgetStore(tmp_path / "b.json").spent_on(DAY) == pytest.approx(0.5)


def test_new_day_resets(tmp_path: Path) -> None:
    store = BudgetStore(tmp_path / "b.json")
    store.add(0.9, DAY)
    assert store.spent_on(date(2026, 11, 28)) == 0.0
    assert store.add(0.1, date(2026, 11, 28)) == pytest.approx(0.1)


def test_corrupted_file_treated_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text("{not json")
    store = BudgetStore(path)
    assert store.spent_on(DAY) == 0.0
    assert store.add(0.2, DAY) == pytest.approx(0.2)


def test_wrong_shape_treated_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text('{"day": "2026-11-27", "spent_usd": "lots"}')
    assert BudgetStore(path).spent_on(DAY) == 0.0


def test_creates_parent_directory(tmp_path: Path) -> None:
    store = BudgetStore(tmp_path / "nested" / "b.json")
    store.add(0.1, DAY)
    assert (tmp_path / "nested" / "b.json").exists()


def test_session_limit_checked_first(tmp_path: Path, settings: Settings) -> None:
    store = BudgetStore(tmp_path / "b.json")
    store.add(5.0, DAY)
    with pytest.raises(BudgetExceeded) as err:
        check_budget(store, settings, session_spent=0.10, day=DAY)
    assert err.value.scope == "session"


def test_daily_limit(tmp_path: Path, settings: Settings) -> None:
    store = BudgetStore(tmp_path / "b.json")
    store.add(1.0, DAY)
    with pytest.raises(BudgetExceeded) as err:
        check_budget(store, settings, session_spent=0.0, day=DAY)
    assert err.value.scope == "daily"


def test_under_limits_passes(tmp_path: Path, settings: Settings) -> None:
    store = BudgetStore(tmp_path / "b.json")
    store.add(0.5, DAY)
    check_budget(store, settings, session_spent=0.05, day=DAY)
