"""Pre-computed results for the bundled samples, so the demo still shows something when the API can't be used."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from doc_agent.ask import AnswerResult, CheckedQuote
from doc_agent.extract import ExtractionResult
from doc_agent.schemas import DOC_TYPES
from doc_agent.validate import Flag

log = logging.getLogger(__name__)

SAMPLES: dict[str, str] = {
    "invoice_clean.pdf": "invoice",
    "invoice_multipage.pdf": "invoice",
    "invoice_messy_scan.pdf": "invoice",
    "invoice_messy_scan.png": "invoice",
    "contract_services.pdf": "contract",
}
SAMPLE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "invoice_clean.pdf": ("What is the total amount due?", "When is the payment due?"),
    "invoice_multipage.pdf": (
        "How many hours of consulting were billed?",
        "Who issued this invoice?",
    ),
    "invoice_messy_scan.pdf": ("What is the total amount due?", "Is there a due date?"),
    "invoice_messy_scan.png": ("What is the total amount due?", "Who is the buyer?"),
    "contract_services.pdf": (
        "How much notice is needed to end the contract?",
        "What is the monthly fee?",
    ),
}
_BROKEN = (OSError, ValueError, KeyError, TypeError, ValidationError)


def _extract_path(sample: str, cache_dir: Path) -> Path:
    return cache_dir / f"{Path(sample).name}.extract.json"


def _qa_path(sample: str, cache_dir: Path) -> Path:
    return cache_dir / f"{Path(sample).name}.qa.json"


def _read_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except _BROKEN:
        log.warning("Cache file %s is unreadable", path)
        return None


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_extraction(result: ExtractionResult, cache_dir: Path) -> Path:
    payload = {
        "doc_name": result.doc_name,
        "doc_type": result.doc_type,
        "data": result.data.model_dump(mode="json"),
        "flags": [{"field": f.field, "reason": f.reason} for f in result.flags],
    }
    return _write_json(_extract_path(result.doc_name, cache_dir), payload)


def load_extraction(sample: str, cache_dir: Path) -> ExtractionResult | None:
    payload = _read_json(_extract_path(sample, cache_dir))
    try:
        doc_type = payload["doc_type"]
        data = DOC_TYPES[doc_type].model_validate(payload["data"])
        flags = tuple(Flag(**flag) for flag in payload["flags"])
        return ExtractionResult(payload["doc_name"], doc_type, data, flags, call=None, cached=True)
    except _BROKEN:
        return None


def _answer_payload(result: AnswerResult) -> dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "quotes": [q.__dict__ for q in result.quotes],
    }


def save_answer(sample: str, result: AnswerResult, cache_dir: Path) -> Path:
    existing = _read_json(_qa_path(sample, cache_dir))
    entries = existing if isinstance(existing, list) else []
    kept = [e for e in entries if isinstance(e, dict) and e.get("question") != result.question]
    return _write_json(_qa_path(sample, cache_dir), [*kept, _answer_payload(result)])


def load_answer(sample: str, question: str, cache_dir: Path) -> AnswerResult | None:
    entries = _read_json(_qa_path(sample, cache_dir))
    if not isinstance(entries, list):
        return None
    wanted = question.strip()
    for entry in entries:
        try:
            if entry["question"] != wanted:
                continue
            quotes = tuple(CheckedQuote(**q) for q in entry["quotes"])
            return AnswerResult(wanted, entry["answer"], quotes, call=None, cached=True)
        except _BROKEN:
            return None
    return None
