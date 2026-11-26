"""Spend limits: a daily total kept in a small JSON file, plus a per-session cap."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal

from doc_agent.config import Settings

log = logging.getLogger(__name__)

BudgetScope = Literal["daily", "session"]


class BudgetExceeded(Exception):
    def __init__(self, scope: BudgetScope) -> None:
        super().__init__(f"{scope} budget exceeded")
        self.scope: BudgetScope = scope


class BudgetStore:
    """Daily spend counter. File format: {"day": "YYYY-MM-DD", "spent_usd": float}."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> tuple[str | None, float]:
        if not self._path.is_file():
            return None, 0.0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            day, spent = data["day"], data["spent_usd"]
            if not isinstance(day, str) or not isinstance(spent, int | float):
                raise ValueError("unexpected budget file shape")
            return day, float(spent)
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("Budget file %s is unreadable; starting from zero", self._path)
            return None, 0.0

    def spent_on(self, day: date) -> float:
        stored_day, spent = self._read()
        return spent if stored_day == day.isoformat() else 0.0

    def add(self, amount: float, day: date) -> float:
        total = self.spent_on(day) + amount
        self._write({"day": day.isoformat(), "spent_usd": total})
        return total

    def _write(self, payload: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def check_budget(store: BudgetStore, settings: Settings, session_spent: float, day: date) -> None:
    if session_spent >= settings.session_budget_usd:
        raise BudgetExceeded("session")
    if store.spent_on(day) >= settings.daily_budget_usd:
        raise BudgetExceeded("daily")
