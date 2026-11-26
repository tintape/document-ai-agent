"""Generate the bundled sample documents. All companies, people and numbers are fictional.

Run: uv run python scripts/make_samples.py
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "samples"
LEFT, TOP, LINE = 60, 790, 18

CLEAN_ITEMS = [("Espresso grinder service", 2, 120.00), ("Coffee beans, 1 kg bag", 5, 18.50)]
MULTI_ITEMS = [
    ("Discovery workshop", 1, 1800.00),
    ("Strategy consulting (hours)", 12, 150.00),
    ("Implementation consulting (hours)", 20, 135.00),
    ("Data migration, product catalog", 1, 950.00),
    ("Data migration, customer records", 1, 1100.00),
    ("Inventory sync setup", 1, 700.00),
    ("Checkout flow review", 1, 450.00),
    ("Staff training session", 2, 380.00),
    ("Training materials", 1, 220.00),
    ("Analytics dashboard setup", 1, 640.00),
    ("Payment provider integration", 1, 880.00),
    ("Load testing", 1, 520.00),
    ("Post-launch support (hours)", 8, 120.00),
    ("Project management", 1, 900.00),
]
TAX_RATE = 0.08


def money(value: float) -> str:
    return f"{value:,.2f}"


class Writer:
    """Tiny helper that writes lines top-down and starts a new page when needed."""

    def __init__(self, path: Path) -> None:
        self.pdf = canvas.Canvas(str(path), pagesize=A4)
        self.y = TOP

    def line(self, text: str, size: int = 10, bold: bool = False, x: int = LEFT) -> None:
        self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.pdf.drawString(x, self.y, text)
        self.y -= LINE + (size - 10)

    def row(self, cells: list[str], xs: list[int], bold: bool = False) -> None:
        self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        for text, x in zip(cells, xs, strict=True):
            self.pdf.drawString(x, self.y, text)
        self.y -= LINE

    def gap(self, lines: int = 1) -> None:
        self.y -= LINE * lines

    def new_page(self) -> None:
        self.pdf.showPage()
        self.y = TOP

    def save(self) -> None:
        self.pdf.save()


def write_items(w: Writer, items: list[tuple[str, int, float]], start: int, stop: int) -> None:
    xs = [LEFT, 330, 400, 480]
    w.row(["Description", "Qty", "Unit price", "Amount"], xs, bold=True)
    for desc, qty, price in items[start:stop]:
        w.row([desc, str(qty), money(price), money(qty * price)], xs)


def write_totals(w: Writer, items: list[tuple[str, int, float]], currency: str) -> None:
    subtotal = sum(qty * price for _, qty, price in items)
    tax = round(subtotal * TAX_RATE, 2)
    w.gap()
    w.line(f"Subtotal: {currency} {money(subtotal)}", x=360)
    w.line(f"Sales tax (8%): {currency} {money(tax)}", x=360)
    w.line(f"Total due: {currency} {money(subtotal + tax)}", bold=True, x=360)


def invoice_clean() -> None:
    w = Writer(OUT / "invoice_clean.pdf")
    w.line("Blue Harbor Supplies Ltd.", 16, bold=True)
    w.line("14 Pier Road, Portland, OR 97201")
    w.gap()
    w.line("INVOICE", 14, bold=True)
    w.line("Invoice number: INV-2041")
    w.line("Invoice date: 2026-11-02")
    w.line("Due date: 2026-12-02")
    w.gap()
    w.line("Bill to: Northwind Cafe, 300 Alder St, Portland, OR 97204")
    w.gap()
    write_items(w, CLEAN_ITEMS, 0, len(CLEAN_ITEMS))
    write_totals(w, CLEAN_ITEMS, "USD")
    w.gap(2)
    w.line("Payment by bank transfer within 30 days. Thank you for your business.")
    w.save()


def invoice_multipage() -> None:
    w = Writer(OUT / "invoice_multipage.pdf")
    w.line("Copperline Consulting GmbH", 16, bold=True)
    w.line("Hafenstrasse 8, 20457 Hamburg, Germany")
    w.gap()
    w.line("INVOICE", 14, bold=True)
    w.line("Invoice number: CC-7781")
    w.line("Invoice date: 15 October 2026")
    w.line("Due date: 14 November 2026")
    w.line("Bill to: Altitude Outdoor Gear Inc., 55 Summit Ave, Denver, CO 80202")
    w.gap()
    write_items(w, MULTI_ITEMS, 0, 9)
    w.gap()
    w.line("Continued on page 2.")
    w.new_page()
    w.line("Copperline Consulting GmbH - Invoice CC-7781 (page 2)", bold=True)
    w.gap()
    write_items(w, MULTI_ITEMS, 9, len(MULTI_ITEMS))
    write_totals(w, MULTI_ITEMS, "USD")
    w.save()


def contract_services() -> None:
    w = Writer(OUT / "contract_services.pdf")
    w.line("SERVICES AGREEMENT", 16, bold=True)
    w.gap()
    w.line("This Services Agreement is made on 2026-10-15 between:")
    w.line("Lumen Analytics LLC, 900 Market St, San Francisco, CA (the Provider), and")
    w.line("Harbor Point Clinics Inc., 12 Bayview Rd, Eugene, OR (the Client).")
    w.gap()
    w.line("1. Services", bold=True)
    w.line(
        "The Provider will run monthly patient-flow reports and maintain the reporting dashboard."
    )
    w.line("2. Term", bold=True)
    w.line("The agreement starts on 2026-11-01 and lasts 12 months.")
    w.line("3. Fees", bold=True)
    w.line("The Client pays a fixed fee of USD 2,000 per month, USD 24,000 in total.")
    w.line("4. Payment terms", bold=True)
    w.line("Invoices are issued monthly and are payable within 15 days (net 15).")
    w.new_page()
    w.line("5. Termination", bold=True)
    w.line("Either party may end this agreement with 30 days written notice.")
    w.line("Fees for services already delivered remain payable.")
    w.line("6. Governing law", bold=True)
    w.line("This agreement is governed by the laws of the State of Oregon.")
    w.gap(2)
    w.line(
        "Signed: Maya Ortiz, Lumen Analytics LLC        Signed: Daniel Reyes, Harbor Point Clinics Inc."
    )
    w.save()


def messy_scan() -> Image.Image:
    """A photo-like invoice: no text layer, no due date, and the items do not add up to the total."""
    image = Image.new("L", (1240, 1754), 236)
    draw = ImageDraw.Draw(image)
    big, body = ImageFont.load_default(size=44), ImageFont.load_default(size=30)
    lines = [
        ("Greenleaf Print Shop", big),
        ("77 Willow Lane, Austin, TX 78701", body),
        ("", body),
        ("INVOICE  No. GP-0393", big),
        ("Date: 2026-11-10", body),
        ("Customer: Maple Street Bakery", body),
        ("", body),
        ("Flyers A5, 500 pcs ............ 120.00", body),
        ("Menu cards, 40 pcs ............  60.00", body),
        ("", body),
        ("TOTAL USD 195.00", big),
    ]
    y = 120
    for text, font in lines:
        draw.text((110, y), text, fill=40, font=font)
        y += 70
    rng = random.Random(7)
    for _ in range(9000):
        x, yy = rng.randrange(image.width), rng.randrange(image.height)
        image.putpixel((x, yy), rng.randrange(150, 220))
    rotated = image.rotate(1.8, expand=True, fillcolor=236).filter(ImageFilter.GaussianBlur(0.8))
    return rotated.convert("RGB")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    invoice_clean()
    invoice_multipage()
    contract_services()
    scan = messy_scan()
    scan.save(OUT / "invoice_messy_scan.png")
    scan.save(OUT / "invoice_messy_scan.pdf", resolution=150)
    print("Wrote", ", ".join(sorted(p.name for p in OUT.iterdir() if p.is_file())))


if __name__ == "__main__":
    main()
