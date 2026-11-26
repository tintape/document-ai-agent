from datetime import date
from decimal import Decimal

import pytest

from doc_agent.schemas import ContractData, FieldValue, InvoiceData, LineItem
from doc_agent.validate import Flag, parse_amount, parse_date, validate_document


def fv(value: str | None, confidence: str = "high", page: int | None = 1) -> FieldValue:
    return FieldValue(value=value, confidence=confidence, page=page)


def item(desc: str, qty: str, price: str, amount: str) -> LineItem:
    return LineItem(description=fv(desc), quantity=fv(qty), unit_price=fv(price), amount=fv(amount))


def make_invoice(**overrides: object) -> InvoiceData:
    fields: dict[str, object] = {
        "vendor": fv("Blue Harbor Supplies Ltd."),
        "buyer": fv("Northwind Cafe"),
        "invoice_number": fv("INV-2041"),
        "invoice_date": fv("2026-11-02"),
        "due_date": fv("2026-12-02"),
        "currency": fv("USD"),
        "tax": fv("0"),
        "total": fv("332.50"),
        "line_items": [
            item("Espresso grinder service", "2", "120.00", "240.00"),
            item("Coffee beans 1kg", "5", "18.50", "92.50"),
        ],
    }
    fields.update(overrides)
    return InvoiceData(**fields)


def make_contract(**overrides: object) -> ContractData:
    fields: dict[str, object] = {
        "parties": [fv("Lumen Analytics LLC"), fv("Harbor Point Clinics Inc.")],
        "signing_date": fv("2026-10-15"),
        "term": fv("12 months"),
        "amount": fv("USD 24,000"),
        "payment_terms": fv("Monthly, net 15"),
        "termination_notice": fv("30 days", page=2),
        "governing_law": fv("State of Oregon", page=2),
    }
    fields.update(overrides)
    return ContractData(**fields)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("USD 1,234.50", Decimal("1234.50")),
        ("$1,200", Decimal("1200")),
        ("1.234,50 €", Decimal("1234.50")),
        ("332.5", Decimal("332.5")),
        ("-15.00", Decimal("-15.00")),
        ("1 234,50", Decimal("1234.50")),
        ("", None),
        (None, None),
        ("n/a", None),
    ],
)
def test_parse_amount(text: str | None, expected: Decimal | None) -> None:
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-11-02", date(2026, 11, 2)),
        ("02.11.2026", date(2026, 11, 2)),
        ("11/02/2026", date(2026, 11, 2)),
        ("2 November 2026", date(2026, 11, 2)),
        ("November 2, 2026", date(2026, 11, 2)),
        ("2 Nov 2026", date(2026, 11, 2)),
        ("soon", None),
        (None, None),
    ],
)
def test_parse_date(text: str | None, expected: date | None) -> None:
    assert parse_date(text) == expected


def test_clean_invoice_has_no_flags() -> None:
    assert validate_document("invoice", make_invoice()) == []


def test_low_confidence_flagged() -> None:
    flags = validate_document("invoice", make_invoice(vendor=fv("Acme", "low")))
    assert Flag("vendor", "Model is not sure (low confidence)") in flags


def test_medium_confidence_flagged() -> None:
    flags = validate_document("invoice", make_invoice(currency=fv("USD", "medium")))
    assert Flag("currency", "Model is not fully sure (medium confidence)") in flags


def test_line_item_confidence_uses_indexed_path() -> None:
    items = [item("A", "1", "10", "10"), item("B", "1", "5", "5")]
    items[1] = LineItem(
        description=fv("B"), quantity=fv("1"), unit_price=fv("5"), amount=fv("5", "low")
    )
    flags = validate_document("invoice", make_invoice(line_items=items, total=fv("15")))
    assert any(f.field == "line_items[2].amount" for f in flags)


def test_missing_required_field_flagged() -> None:
    flags = validate_document("invoice", make_invoice(invoice_number=fv(None)))
    assert Flag("invoice_number", "Required field is missing") in flags


def test_sum_mismatch_flagged() -> None:
    flags = validate_document("invoice", make_invoice(total=fv("400.00")))
    assert any(f.field == "total" and "do not add up" in f.reason for f in flags)


def test_sum_within_tolerance_not_flagged() -> None:
    assert validate_document("invoice", make_invoice(total=fv("332.51"))) == []


def test_tax_is_added_to_items() -> None:
    assert validate_document("invoice", make_invoice(tax=fv("10.00"), total=fv("342.50"))) == []


def test_due_before_invoice_date_flagged() -> None:
    flags = validate_document("invoice", make_invoice(due_date=fv("2026-10-01")))
    assert any(f.field == "due_date" for f in flags)


def test_unreadable_date_flagged() -> None:
    flags = validate_document("invoice", make_invoice(invoice_date=fv("sometime")))
    assert any(f.field == "invoice_date" for f in flags)


def test_unparseable_total_flagged_not_crash() -> None:
    flags = validate_document("invoice", make_invoice(total=fv("about three hundred")))
    assert any(f.field == "total" for f in flags)


def test_unparseable_item_amount_skips_sum_check() -> None:
    items = [item("A", "1", "10", "ten"), item("B", "1", "5", "5")]
    flags = validate_document("invoice", make_invoice(line_items=items, total=fv("999")))
    assert any(f.field == "line_items" for f in flags)
    assert not any(f.field == "total" and "do not add up" in f.reason for f in flags)


def test_clean_contract_has_no_flags() -> None:
    assert validate_document("contract", make_contract()) == []


def test_contract_needs_two_parties() -> None:
    flags = validate_document("contract", make_contract(parties=[fv("Acme")]))
    assert any(f.field == "parties" for f in flags)


def test_contract_missing_signing_date() -> None:
    flags = validate_document("contract", make_contract(signing_date=fv(None)))
    assert Flag("signing_date", "Required field is missing") in flags


def test_unknown_doc_type_rejected() -> None:
    with pytest.raises(ValueError):
        validate_document("receipt", make_invoice())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Nov 27, 2026", date(2026, 11, 27)),
        ("Nov. 27, 2026", date(2026, 11, 27)),
        ("27/11/2026", date(2026, 11, 27)),
        ("27-Nov-2026", date(2026, 11, 27)),
        ("2026/11/27", date(2026, 11, 27)),
        ("27.11.26", date(2026, 11, 27)),
    ],
)
def test_parse_more_real_world_dates(text: str, expected: date) -> None:
    assert parse_date(text) == expected


def test_ambiguous_slash_dates_do_not_trigger_order_flag() -> None:
    invoice = make_invoice(invoice_date=fv("03/04/2026"), due_date=fv("02/05/2026"))
    flags = validate_document("invoice", invoice)
    assert not any(f.field == "due_date" for f in flags)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rs. 1,200", Decimal("1200")),
        ("approx. 500", Decimal("500")),
        ("1.200 €", Decimal("1200")),
        ("€ 1.234", Decimal("1234")),
        ("1'234.50", Decimal("1234.50")),
        ("12,5", Decimal("12.5")),
        ("5,000", Decimal("5000")),
        ("5,00", Decimal("5.00")),
        ("1.234.567", Decimal("1234567")),
        ("US$1,200.00", Decimal("1200.00")),
    ],
)
def test_parse_amount_real_world(text: str, expected: Decimal) -> None:
    assert parse_amount(text) == expected
