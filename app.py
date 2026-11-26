"""Streamlit UI. All logic lives in doc_agent/; this file only draws it.

Run: uv run streamlit run app.py   (DEMO_DEV=1 adds the "simulate API failure" switch)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from doc_agent.budget import BudgetStore
from doc_agent.cache import SAMPLE_QUESTIONS, SAMPLES
from doc_agent.config import Settings, load_settings
from doc_agent.documents import DocumentError, LoadedDocument, load_document
from doc_agent.extract import ExtractionResult, to_csv, to_json, to_rows
from doc_agent.llm import LLMClient, utc_today
from doc_agent.session import (
    UNEXPECTED_ERROR,
    SessionLedger,
    documents_key,
    run_extraction,
    run_question,
)

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("doc_agent.app")
ISSUE_STYLE = "background-color: #fff3cd; color: #664d03"
LOG_COLUMNS = ["kind", "attempts", "cost_usd", "cost_source", "input_tokens", "output_tokens", "duration_s", "model"]  # fmt: skip
STATUS_LABELS = {
    "verified": ":green-background[verified]",
    "not_found": ":red-background[not found in document]",
    "unverified": ":gray-background[unverified]",
}
CROSS_QUESTIONS = (
    "Which document has the largest total?",
    "List every party or company mentioned.",
)
DEFAULT_SAMPLE = "invoice_clean.pdf"
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<:$])")


@dataclass(frozen=True)
class Selection:
    """The documents a visitor picked, with each one's type and sample name (None for uploads)."""

    docs: tuple[LoadedDocument, ...] = ()
    types: tuple[str, ...] = ()
    samples: tuple[str | None, ...] = ()
    errors: tuple[str, ...] = field(default=())

    @property
    def single_sample(self) -> str | None:
        return self.samples[0] if len(self.docs) == 1 else None


def md_escape(text: str) -> str:
    """Untrusted text (file names, model output) must never be rendered as markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text)


@st.cache_resource
def get_settings() -> Settings:
    return load_settings(os.environ, ROOT)


@st.cache_resource
def get_store() -> BudgetStore:
    return BudgetStore(get_settings().budget_file)


@st.cache_resource
def get_llm() -> LLMClient | None:
    settings = get_settings()
    return LLMClient(settings, get_store()) if settings.api_key else None


@st.cache_data(max_entries=32, show_spinner=False)
def parse_cached(name: str, data: bytes) -> LoadedDocument:
    """PDF parsing is cached by content, so widget clicks don't re-read a 20-page upload."""
    return load_document(name, data, get_settings())


def init_state() -> None:
    for key, value in {"ledger": SessionLedger(), "docs_key": None, "extraction": None,
                       "answer": None, "question": ""}.items():  # fmt: skip
        st.session_state.setdefault(key, value)


def _collect(
    files: list[tuple[str, bytes]], types: list[str], samples: list[str | None]
) -> Selection:
    docs, kept_types, kept_samples, errors = [], [], [], []
    for (name, data), doc_type, sample in zip(files, types, samples, strict=True):
        try:
            docs.append(parse_cached(name, data))
        except DocumentError as err:
            errors.append(str(err))
            continue
        kept_types.append(doc_type)
        kept_samples.append(sample)
    return Selection(tuple(docs), tuple(kept_types), tuple(kept_samples), tuple(errors))


def pick_documents() -> Selection:
    st.sidebar.header("Documents")
    source = st.sidebar.radio("Source", ["Try samples", "Upload your own"], horizontal=True)
    if source == "Try samples":
        names = st.sidebar.multiselect("Samples", list(SAMPLES), default=[DEFAULT_SAMPLE])
        files = [(n, (ROOT / "samples" / n).read_bytes()) for n in names]
        return _collect(files, [SAMPLES[n] for n in names], list(names))
    uploads = st.sidebar.file_uploader(
        "PDF, PNG or JPG, up to 10 MB each",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    doc_type = st.sidebar.selectbox("Document type", ["invoice", "contract"], format_func=str.title)
    files = [(u.name, u.getvalue()) for u in uploads or []]
    return _collect(files, [doc_type] * len(files), [None] * len(files))


def dev_controls(settings: Settings) -> int:
    if settings.dev_mode:
        return 1 if st.sidebar.checkbox("Simulate API failure (1 retry)") else 0
    return 0


def spend_panel(box: DeltaGenerator, settings: Settings) -> None:
    """Drawn last, into a slot reserved at the top, so it already includes this run's request."""
    ledger: SessionLedger = st.session_state.ledger
    box.header("Spend")
    limit = f"Limit ${settings.session_budget_usd:.2f}"
    box.metric("This session", f"${ledger.spent_usd:.4f}", help=limit)
    today = get_store().spent_on(utc_today())
    daily = f"Limit ${settings.daily_budget_usd:.2f}"
    box.metric("Today (all visitors)", f"${today:.4f}", help=daily)
    box.caption(f"Model: {settings.model}" + ("" if settings.api_key else " · live requests off"))


def show_messages(error: str | None, notice: str | None) -> None:
    if notice:
        st.info(md_escape(notice))
    if error:
        st.error(md_escape(error))


def fields_table(result: ExtractionResult) -> None:
    frame = pd.DataFrame(to_rows(result))
    styled = frame.style.apply(lambda row: [ISSUE_STYLE if row["issue"] else ""] * len(row), axis=1)
    columns = {
        "field": st.column_config.TextColumn(width="medium"),
        "confidence": st.column_config.TextColumn(width="small"),
        "page": st.column_config.TextColumn(width="small"),
        "issue": st.column_config.TextColumn(width="large"),
    }
    st.dataframe(styled, hide_index=True, width="stretch", column_config=columns)
    stem = Path(result.doc_name).stem
    left, right = st.columns(2)
    left.download_button("Download CSV", to_csv(result), f"{stem}.csv", "text/csv")
    right.download_button("Download JSON", to_json(result), f"{stem}.json", "application/json")


def extract_tab(sel: Selection, simulate: int) -> None:
    index = 0
    if len(sel.docs) > 1:
        names = [d.name for d in sel.docs]
        index = st.selectbox("Document", range(len(names)), format_func=names.__getitem__)
    doc, doc_type, sample = sel.docs[index], sel.types[index], sel.samples[index]
    key = documents_key([doc])
    if st.button("Extract fields", type="primary"):
        outcome = run_extraction(
            doc, doc_type, get_llm(), st.session_state.ledger, sample,
            get_settings().cache_dir, simulate,
        )  # fmt: skip
        st.session_state.ledger = outcome.ledger
        st.session_state.extraction = (key, outcome)
    stored = st.session_state.extraction
    if stored is None or stored[0] != key:
        st.caption("Click the button to pull the fields out of the document.")
        return
    outcome = stored[1]
    show_messages(outcome.error, outcome.notice)
    if outcome.result is not None:
        if outcome.result.cached:
            st.caption(":gray-background[cached result]")
        flags = outcome.result.flags
        st.write(
            f"**{len(flags)} issue(s)** found by our checks:" if flags else "All checks passed."
        )
        for flag in flags:
            st.markdown(f"- **{md_escape(flag.field)}**: {md_escape(flag.reason)}")
        fields_table(outcome.result)


def question_input(sel: Selection) -> str:
    suggestions = SAMPLE_QUESTIONS.get(sel.single_sample or "", ())
    if len(sel.docs) > 1:
        suggestions = CROSS_QUESTIONS
    if suggestions:
        for col, suggestion in zip(st.columns(len(suggestions)), suggestions, strict=True):
            if col.button(suggestion, width="stretch"):
                st.session_state.question = suggestion
    return st.text_input("Your question", key="question", max_chars=500)


def show_answer(sel: Selection) -> None:
    outcome = st.session_state.answer
    if outcome is None:
        return
    show_messages(outcome.error, outcome.notice)
    if outcome.result is None:
        return
    if outcome.result.cached:
        st.caption(":gray-background[cached result]")
    st.text(outcome.result.answer)
    for quote in outcome.result.quotes:
        where = (
            f"{md_escape(quote.doc_name)}, page {quote.page}"
            if len(sel.docs) > 1
            else f"page {quote.page}"
        )
        note = f" · {md_escape(quote.note)}" if quote.note else ""
        st.markdown(f"{STATUS_LABELS[quote.status]} {where}{note}")
        st.text(f"“{quote.text}”")


def ask_tab(sel: Selection, simulate: int) -> None:
    if len(sel.docs) > 1:
        st.caption(f"Questions go across all {len(sel.docs)} selected documents.")
    question = question_input(sel)
    if st.button("Ask", type="primary"):
        outcome = run_question(
            list(sel.docs), question, get_llm(), st.session_state.ledger, sel.single_sample,
            get_settings().cache_dir, simulate,
        )  # fmt: skip
        st.session_state.ledger = outcome.ledger
        st.session_state.answer = outcome
    show_answer(sel)


def request_log() -> None:
    calls = st.session_state.ledger.calls
    with st.expander(f"Request log ({len(calls)})"):
        if not calls:
            st.caption("No live requests yet in this session.")
            return
        frame = pd.DataFrame([c.__dict__ for c in calls])[LOG_COLUMNS]
        st.dataframe(frame, hide_index=True, width="stretch")


def show_documents(sel: Selection, simulate: int) -> None:
    key = hashlib.sha256((documents_key(sel.docs) + "|".join(sel.types)).encode()).hexdigest()
    if st.session_state.docs_key != key:
        st.session_state.update(docs_key=key, answer=None, question="")
    extract_col, ask_col = st.tabs(["Extract", "Ask"])
    with extract_col:
        extract_tab(sel, simulate)
    with ask_col:
        ask_tab(sel, simulate)
    request_log()


def main() -> None:
    st.set_page_config(page_title="Document AI Agent", page_icon="📄", layout="wide")
    init_state()
    settings = get_settings()
    st.title("Document AI Agent")
    st.caption(
        "Extract fields from invoices and contracts, then ask questions. "
        "Every quote is checked against the document."
    )
    try:
        sel = pick_documents()
    except Exception:  # last resort: never show a traceback to a visitor
        log.exception("Unexpected error while loading documents")
        st.error(UNEXPECTED_ERROR)
        sel = Selection()
    for error in sel.errors:
        st.error(md_escape(error))
    spend_box = st.sidebar.container()
    simulate = dev_controls(settings)
    if sel.docs:
        show_documents(sel, simulate)
    else:
        st.info("Upload documents or pick samples in the sidebar.")
    spend_panel(spend_box, settings)


main()
