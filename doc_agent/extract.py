"""Field extraction: one model call per document, then our deterministic checks, then export."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass

from pydantic import BaseModel

from doc_agent.documents import LoadedDocument
from doc_agent.llm import CallInfo, LLMClient
from doc_agent.prompts import EXTRACT_PROMPT
from doc_agent.schemas import DOC_TYPES
from doc_agent.validate import Flag, validate_document, walk_fields

EXTRACT_EFFORT = "low"
EXTRACT_MAX_TOKENS = 4000
ROW_KEYS = ("field", "value", "confidence", "page", "issue")
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
PLAIN_NUMBER = re.compile(r"[+-]?[\d.,]+")


@dataclass(frozen=True)
class ExtractionResult:
    doc_name: str
    doc_type: str
    data: BaseModel
    flags: tuple[Flag, ...]
    call: CallInfo | None
    cached: bool


def extract(
    doc: LoadedDocument,
    doc_type: str,
    llm: LLMClient,
    session_spent: float,
    simulate_failures: int = 0,
) -> ExtractionResult:
    schema = DOC_TYPES.get(doc_type)
    if schema is None:
        raise ValueError(f"Unknown document type: {doc_type}")
    result = llm.call(
        kind="extract",
        content=[doc.content_block(), {"type": "text", "text": EXTRACT_PROMPT[doc_type]}],
        schema=schema,
        effort=EXTRACT_EFFORT,
        max_tokens=EXTRACT_MAX_TOKENS,
        session_spent=session_spent,
        simulate_failures=simulate_failures,
    )
    flags = tuple(validate_document(doc_type, result.parsed))
    return ExtractionResult(doc.name, doc_type, result.parsed, flags, result.call, cached=False)


def _issues_by_field(flags: tuple[Flag, ...]) -> dict[str, str]:
    issues: dict[str, list[str]] = {}
    for flag in flags:
        issues.setdefault(flag.field, []).append(flag.reason)
    return {field: "; ".join(reasons) for field, reasons in issues.items()}


def to_rows(result: ExtractionResult) -> list[dict[str, str]]:
    issues = _issues_by_field(result.flags)
    rows = [
        {
            "field": path,
            "value": fv.value or "",
            "confidence": fv.confidence,
            "page": "" if fv.page is None else str(fv.page),
            "issue": issues.pop(path, ""),
        }
        for path, fv in walk_fields(result.data)
    ]
    group_rows = [
        {"field": field, "value": "", "confidence": "", "page": "", "issue": issue}
        for field, issue in issues.items()
    ]
    return group_rows + rows


def csv_safe(cell: str) -> str:
    """Stop spreadsheet apps from running a cell as a formula; plain numbers stay as they are."""
    if cell.startswith(FORMULA_PREFIXES) and not PLAIN_NUMBER.fullmatch(cell):
        return "'" + cell
    return cell


def to_csv(result: ExtractionResult) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_KEYS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: csv_safe(value) for key, value in row.items()} for row in to_rows(result))
    return buf.getvalue()


def to_json(result: ExtractionResult) -> str:
    payload = {
        "document": result.doc_name,
        "type": result.doc_type,
        "fields": result.data.model_dump(mode="json"),
        "issues": [{"field": f.field, "reason": f.reason} for f in result.flags],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
