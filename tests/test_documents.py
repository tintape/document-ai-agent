import base64

import pytest

from doc_agent.config import Settings
from doc_agent.documents import DocumentError, load_document
from tests import factories


def test_text_pdf_has_pages_and_text(settings: Settings) -> None:
    doc = load_document("inv.pdf", factories.invoice_pdf(), settings)
    assert doc.media_type == "application/pdf"
    assert doc.page_count == 1
    assert doc.has_text_layer
    assert "INV-2041" in doc.page_texts[0]
    block = doc.content_block()
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(block["source"]["data"]) == doc.data


def test_two_page_pdf_keeps_text_per_page(settings: Settings) -> None:
    doc = load_document("contract.pdf", factories.contract_pdf(), settings)
    assert doc.page_count == 2
    assert "Termination notice" in doc.page_texts[1]
    assert "Termination notice" not in doc.page_texts[0]


def test_image_only_pdf_has_no_text_layer(settings: Settings) -> None:
    doc = load_document("scan.pdf", factories.scanned_pdf(), settings)
    assert doc.page_count == 1
    assert doc.has_text_layer is False


def test_png_is_image_block(settings: Settings) -> None:
    doc = load_document("photo.png", factories.png_bytes(), settings)
    assert doc.page_count == 1
    assert doc.page_texts == ()
    assert doc.has_text_layer is False
    block = doc.content_block()
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"


def test_jpeg_detected(settings: Settings) -> None:
    assert load_document("photo.jpg", factories.jpeg_bytes(), settings).media_type == "image/jpeg"


def test_type_detected_from_bytes_not_name(settings: Settings) -> None:
    assert load_document("report.pdf", factories.png_bytes(), settings).media_type == "image/png"


@pytest.mark.parametrize(
    ("name", "data", "msg"),
    [
        ("x.pdf", b"<html>nope</html>", "not a supported"),
        ("x.pdf", b"%PDF-1.4 broken", "could not be read"),
        ("x.txt", b"hello", "not a supported"),
        ("empty.pdf", b"", "empty"),
    ],
)
def test_bad_files_raise_friendly_error(
    name: str, data: bytes, msg: str, settings: Settings
) -> None:
    with pytest.raises(DocumentError, match=msg):
        load_document(name, data, settings)


def test_encrypted_pdf(settings: Settings) -> None:
    data = factories.pdf_from_pages([factories.INVOICE_LINES], encrypt="pw")
    with pytest.raises(DocumentError, match="password"):
        load_document("locked.pdf", data, settings)


def test_too_many_pages(settings: Settings) -> None:
    data = factories.pdf_from_pages([("page",)] * 21)
    with pytest.raises(DocumentError, match="20 pages"):
        load_document("big.pdf", data, settings)


def test_too_large(settings: Settings) -> None:
    with pytest.raises(DocumentError, match="10 MB"):
        load_document("big.pdf", b"%PDF" + b"0" * (10 * 1024 * 1024), settings)
