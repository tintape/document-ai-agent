"""Builders for test documents (all content fictional)."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

INVOICE_LINES = (
    "INVOICE #INV-2041",
    "Vendor: Blue Harbor Supplies Ltd.",
    "Bill to: Northwind Cafe",
    "Invoice date: 2026-11-02",
    "Due date: 2026-12-02",
    "2 x Espresso grinder service @ 120.00 = 240.00",
    "5 x Coffee beans 1kg @ 18.50 = 92.50",
    "Total due: USD 332.50",
)
CONTRACT_PAGE_1 = (
    "SERVICES AGREEMENT",
    "Between Lumen Analytics LLC and Harbor Point Clinics Inc.",
    "Signed on 2026-10-15.",
)
CONTRACT_PAGE_2 = (
    "Termination notice: 30 days in writing.",
    "Governing law: State of Oregon.",
)


def pdf_from_pages(pages: list[tuple[str, ...]], encrypt: str | None = None) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4, encrypt=encrypt)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(60, y, line)
            y -= 22
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def invoice_pdf() -> bytes:
    return pdf_from_pages([INVOICE_LINES])


def contract_pdf() -> bytes:
    return pdf_from_pages([CONTRACT_PAGE_1, CONTRACT_PAGE_2])


def invoice_image() -> Image.Image:
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(INVOICE_LINES):
        draw.text((40, 40 + i * 40), line, fill="black")
    return image


def png_bytes() -> bytes:
    buf = io.BytesIO()
    invoice_image().save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    invoice_image().save(buf, format="JPEG")
    return buf.getvalue()


def scanned_pdf() -> bytes:
    buf = io.BytesIO()
    invoice_image().save(buf, format="PDF")
    return buf.getvalue()
