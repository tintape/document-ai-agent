"""Deterministic checks on extracted fields. No AI here: this is what catches the model's mistakes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from doc_agent.schemas import ContractData, FieldValue, InvoiceData

DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d.%m.%y", "%m/%d/%Y", "%d/%m/%Y", "%d-%b-%Y",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%b. %d, %Y",
)  # fmt: skip
CONFIDENCE_REASONS = {
    "medium": "Model is not fully sure (medium confidence)",
    "low": "Model is not sure (low confidence)",
}
MISSING = "Required field is missing"
UNREADABLE_DATE = "Date is not in a recognised format"
SUM_TOLERANCE_ABS = Decimal("0.01")
SUM_TOLERANCE_REL = Decimal("0.005")
INVOICE_REQUIRED = ("vendor", "invoice_number", "invoice_date", "total")
CONTRACT_REQUIRED = ("signing_date",)
_NUMBER = re.compile(r"-?\d(?:[\d.,'\s ]*\d)?")
_SLASH_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/\d{4}")


@dataclass(frozen=True)
class Flag:
    field: str
    reason: str


def _normalize_separators(text: str) -> str:
    if "," in text and "." in text:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal_sep == "," else ","
        return text.replace(thousands, "").replace(decimal_sep, ".")
    for sep in (",", "."):
        if sep not in text:
            continue
        head, _, tail = text.rpartition(sep)
        is_thousands = text.count(sep) > 1 or len(tail) == 3
        return text.replace(sep, "") if is_thousands else f"{head.replace(sep, '')}.{tail}"
    return text


def parse_amount(text: str | None) -> Decimal | None:
    """Parse the first number in the text: '$1,200', '1.234,50 €', 'Rs. 1,200', "1'234.50"."""
    match = _NUMBER.search(text or "")
    if match is None:
        return None
    core = re.sub(r"[\s' ]", "", match.group())
    try:
        return Decimal(_normalize_separators(core))
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


def is_ambiguous_date(text: str | None) -> bool:
    """'03/04/2026' reads as March 4 or 3 April: don't build checks on its order."""
    match = _SLASH_DATE.fullmatch((text or "").strip())
    if match is None:
        return False
    first, second = int(match.group(1)), int(match.group(2))
    return first != second and first <= 12 and second <= 12


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
    return [
        Flag(name, MISSING) for name in required if not (getattr(data, name).value or "").strip()
    ]


def _date_flags(invoice: InvoiceData) -> list[Flag]:
    flags: list[Flag] = []
    parsed: dict[str, date | None] = {}
    for name in ("invoice_date", "due_date"):
        raw = getattr(invoice, name).value
        parsed[name] = parse_date(raw)
        if raw and parsed[name] is None:
            flags.append(Flag(name, UNREADABLE_DATE))
    if not (invoice.due_date.value or "").strip():
        flags.append(Flag("due_date", "No due date found"))
    issued, due = parsed["invoice_date"], parsed["due_date"]
    ambiguous = any(
        is_ambiguous_date(getattr(invoice, n).value) for n in ("invoice_date", "due_date")
    )
    if issued and due and not ambiguous and due < issued:
        flags.append(Flag("due_date", "Due date is before the invoice date"))
    return flags


def _tax(invoice: InvoiceData) -> tuple[Decimal, list[Flag]]:
    """Tax amount for the sum check: a rate like '20%' or unreadable text counts as 0 and is flagged."""
    raw = (invoice.tax.value or "").strip()
    if not raw:
        return Decimal("0"), []
    if "%" in raw:
        return Decimal("0"), [
            Flag("tax", "Tax shows a rate, not an amount, so it was left out of the sum")
        ]
    amount = parse_amount(raw)
    if amount is None:
        return Decimal("0"), [Flag("tax", "Tax is not a readable amount")]
    return amount, []


def _sum_flags(invoice: InvoiceData) -> list[Flag]:
    total = parse_amount(invoice.total.value)
    if invoice.total.value and total is None:
        return [Flag("total", "Total is not a readable amount")]
    if total is None or not invoice.line_items:
        return []
    amounts = [parse_amount(item.amount.value) for item in invoice.line_items]
    if any(amount is None for amount in amounts):
        reason = "Some line item amounts are not readable, so the sum was not checked"
        return [Flag("line_items", reason)]
    tax, flags = _tax(invoice)
    items_sum = sum(amounts, Decimal("0"))
    tolerance = max(SUM_TOLERANCE_ABS, SUM_TOLERANCE_REL * abs(total))
    if abs(items_sum + tax - total) <= tolerance:
        return flags
    reason = f"Line items ({items_sum}) plus tax ({tax}) do not add up to the total ({total})"
    return [*flags, Flag("total", reason)]


def _invoice_flags(invoice: InvoiceData) -> list[Flag]:
    return _missing_flags(invoice, INVOICE_REQUIRED) + _date_flags(invoice) + _sum_flags(invoice)


def _contract_flags(contract: ContractData) -> list[Flag]:
    flags = _missing_flags(contract, CONTRACT_REQUIRED)
    named = [p for p in contract.parties if (p.value or "").strip()]
    if len(named) < 2:
        flags.append(Flag("parties", "Expected at least two parties"))
    signed = contract.signing_date.value
    if signed and parse_date(signed) is None:
        flags.append(Flag("signing_date", UNREADABLE_DATE))
    return flags


def validate_document(doc_type: str, data: BaseModel) -> list[Flag]:
    if doc_type == "invoice" and isinstance(data, InvoiceData):
        return _confidence_flags(data) + _invoice_flags(data)
    if doc_type == "contract" and isinstance(data, ContractData):
        return _confidence_flags(data) + _contract_flags(data)
    raise ValueError(f"Unknown document type: {doc_type}")
