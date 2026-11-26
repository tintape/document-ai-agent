"""Deterministic checks on extracted fields. No AI here: this is what catches the model's mistakes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from doc_agent.schemas import ContractData, FieldValue, InvoiceData

DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y", "%d %b %Y")
CONFIDENCE_REASONS = {
    "medium": "Model is not fully sure (medium confidence)",
    "low": "Model is not sure (low confidence)",
}
MISSING = "Required field is missing"
SUM_TOLERANCE_ABS = Decimal("0.01")
SUM_TOLERANCE_REL = Decimal("0.005")
INVOICE_REQUIRED = ("vendor", "invoice_number", "invoice_date", "total")
CONTRACT_REQUIRED = ("signing_date",)
_NOT_NUMBER = re.compile(r"[^\d.,\-]")


@dataclass(frozen=True)
class Flag:
    field: str
    reason: str


def _normalize_separators(text: str) -> str:
    if "," in text and "." in text:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal_sep == "," else ","
        return text.replace(thousands, "").replace(decimal_sep, ".")
    if "," in text:
        head, _, tail = text.rpartition(",")
        is_decimal = text.count(",") == 1 and len(tail) in (1, 2)
        return f"{head}.{tail}" if is_decimal else text.replace(",", "")
    if text.count(".") > 1:
        return text.replace(".", "")
    return text


def parse_amount(text: str | None) -> Decimal | None:
    if not text:
        return None
    cleaned = _NOT_NUMBER.sub("", text)
    if not any(ch.isdigit() for ch in cleaned):
        return None
    try:
        return Decimal(_normalize_separators(cleaned))
    except InvalidOperation:
        return None


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def walk_fields(data: BaseModel, prefix: str = "") -> Iterator[tuple[str, FieldValue]]:
    for name, value in data:
        path = f"{prefix}{name}"
        if isinstance(value, FieldValue):
            yield path, value
        elif isinstance(value, BaseModel):
            yield from walk_fields(value, f"{path}.")
        elif isinstance(value, list):
            for i, entry in enumerate(value, start=1):
                if isinstance(entry, FieldValue):
                    yield f"{path}[{i}]", entry
                elif isinstance(entry, BaseModel):
                    yield from walk_fields(entry, f"{path}[{i}].")


def _confidence_flags(data: BaseModel) -> list[Flag]:
    return [
        Flag(path, CONFIDENCE_REASONS[fv.confidence])
        for path, fv in walk_fields(data)
        if fv.confidence in CONFIDENCE_REASONS and fv.value is not None
    ]


def _missing_flags(data: BaseModel, required: tuple[str, ...]) -> list[Flag]:
    return [Flag(name, MISSING) for name in required if not (getattr(data, name).value or "").strip()]


def _date_flags(invoice: InvoiceData) -> list[Flag]:
    flags: list[Flag] = []
    parsed: dict[str, date | None] = {}
    for name in ("invoice_date", "due_date"):
        raw = getattr(invoice, name).value
        parsed[name] = parse_date(raw)
        if raw and parsed[name] is None:
            flags.append(Flag(name, "Date is not in a recognised format"))
    issued, due = parsed["invoice_date"], parsed["due_date"]
    if issued and due and due < issued:
        flags.append(Flag("due_date", "Due date is before the invoice date"))
    return flags


def _sum_flags(invoice: InvoiceData) -> list[Flag]:
    total = parse_amount(invoice.total.value)
    if invoice.total.value and total is None:
        return [Flag("total", "Total is not a readable amount")]
    if total is None or not invoice.line_items:
        return []
    amounts = [parse_amount(item.amount.value) for item in invoice.line_items]
    if any(amount is None for amount in amounts):
        return [Flag("line_items", "Some line item amounts are not readable, so the sum was not checked")]
    tax = parse_amount(invoice.tax.value) or Decimal("0")
    items_sum = sum(amounts, Decimal("0"))
    tolerance = max(SUM_TOLERANCE_ABS, SUM_TOLERANCE_REL * abs(total))
    if abs(items_sum + tax - total) <= tolerance:
        return []
    reason = f"Line items ({items_sum}) plus tax ({tax}) do not add up to the total ({total})"
    return [Flag("total", reason)]


def _invoice_flags(invoice: InvoiceData) -> list[Flag]:
    return _missing_flags(invoice, INVOICE_REQUIRED) + _date_flags(invoice) + _sum_flags(invoice)


def _contract_flags(contract: ContractData) -> list[Flag]:
    flags = _missing_flags(contract, CONTRACT_REQUIRED)
    named = [p for p in contract.parties if (p.value or "").strip()]
    if len(named) < 2:
        flags.append(Flag("parties", "Expected at least two parties"))
    return flags


def validate_document(doc_type: str, data: BaseModel) -> list[Flag]:
    if doc_type == "invoice" and isinstance(data, InvoiceData):
        return _confidence_flags(data) + _invoice_flags(data)
    if doc_type == "contract" and isinstance(data, ContractData):
        return _confidence_flags(data) + _contract_flags(data)
    raise ValueError(f"Unknown document type: {doc_type}")
