"""Shapes the model must answer in. Values stay strings: our own code parses and checks them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Confidence = Literal["high", "medium", "low"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FieldValue(_Frozen):
    value: str | None
    confidence: Confidence
    page: int | None


class LineItem(_Frozen):
    description: FieldValue
    quantity: FieldValue
    unit_price: FieldValue
    amount: FieldValue


class InvoiceData(_Frozen):
    vendor: FieldValue
    buyer: FieldValue
    invoice_number: FieldValue
    invoice_date: FieldValue
    due_date: FieldValue
    currency: FieldValue
    tax: FieldValue
    total: FieldValue
    line_items: list[LineItem]


class ContractData(_Frozen):
    parties: list[FieldValue]
    signing_date: FieldValue
    term: FieldValue
    amount: FieldValue
    payment_terms: FieldValue
    termination_notice: FieldValue
    governing_law: FieldValue


class Quote(_Frozen):
    document: int
    page: int
    text: str


class AnswerData(_Frozen):
    answer: str
    quotes: list[Quote]


DOC_TYPES: dict[str, type[BaseModel]] = {"invoice": InvoiceData, "contract": ContractData}
