"""Uploaded files: type detection by content, size and page limits, per-page text for quote checks."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

from doc_agent.config import Settings

log = logging.getLogger(__name__)

MediaType = Literal["application/pdf", "image/png", "image/jpeg"]
MIN_TEXT_CHARS = 20
BYTES_PER_MB = 1024 * 1024
SIGNATURES: tuple[tuple[bytes, MediaType], ...] = (
    (b"%PDF", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


class DocumentError(Exception):
    """A problem with an uploaded file. The message is shown to the user as is."""


@dataclass(frozen=True)
class LoadedDocument:
    name: str
    media_type: MediaType
    data: bytes
    page_texts: tuple[str, ...]
    page_count: int

    @property
    def has_text_layer(self) -> bool:
        return any(len("".join(text.split())) >= MIN_TEXT_CHARS for text in self.page_texts)

    def content_block(self) -> dict:
        source = {
            "type": "base64",
            "media_type": self.media_type,
            "data": base64.standard_b64encode(self.data).decode("ascii"),
        }
        kind = "document" if self.media_type == "application/pdf" else "image"
        return {"type": kind, "source": source}


def detect_media_type(data: bytes) -> MediaType | None:
    for signature, media_type in SIGNATURES:
        if data.startswith(signature):
            return media_type
    return None


def _read_pdf(name: str, data: bytes, settings: Settings) -> tuple[str, ...]:
    locked = DocumentError(f"“{name}” is password-protected. Please upload an unlocked copy.")
    damaged = DocumentError(f"“{name}” could not be read as a PDF. It may be damaged.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise locked
        page_count = len(reader.pages)
        if page_count == 0:
            raise damaged
        if page_count > settings.max_pages:
            raise DocumentError(
                f"“{name}” has {page_count} pages. The demo accepts up to {settings.max_pages} pages."
            )
        return tuple(page.extract_text() or "" for page in reader.pages)
    except DocumentError:
        raise
    except DependencyError as err:  # AES-encrypted PDFs fail while opening, before is_encrypted
        raise locked from err
    except (PyPdfError, Exception) as err:  # hostile PDFs can raise almost anything inside pypdf
        log.info("Unreadable PDF: %s", type(err).__name__)
        raise damaged from err


def load_document(name: str, data: bytes, settings: Settings) -> LoadedDocument:
    if not data:
        raise DocumentError(f"“{name}” is empty.")
    if len(data) > settings.max_file_mb * BYTES_PER_MB:
        raise DocumentError(f"“{name}” is larger than {settings.max_file_mb} MB.")
    media_type = detect_media_type(data)
    if media_type is None:
        raise DocumentError(f"“{name}” is not a supported file. Please upload a PDF, PNG or JPG.")
    if media_type != "application/pdf":
        return LoadedDocument(name, media_type, data, page_texts=(), page_count=1)
    page_texts = _read_pdf(name, data, settings)
    return LoadedDocument(name, media_type, data, page_texts, page_count=len(page_texts))
