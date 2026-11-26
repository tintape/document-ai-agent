"""Streamlit UI. All logic lives in doc_agent/; this file only draws it.

Run: uv run streamlit run app.py   (DEMO_DEV=1 adds the "simulate API failure" switch)
"""

from __future__ import annotations

import logging
import os
import re
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
from doc_agent.session import UNEXPECTED_ERROR, SessionLedger, run_extraction, run_question

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("doc_agent.app")
ISSUE_STYLE = "background-color: #fff3cd; color: #664d03"
LOG_COLUMNS = [
    "kind",
    "attempts",
    "cost_usd",
    "cost_source",
    "input_tokens",
    "output_tokens",
    "duration_s",
    "model",
]
STATUS_LABELS = {
    "verified": ":green-background[verified]",
    "not_found": ":red-background[not found in document]",
    "unverified": ":gray-background[unverified (scan)]",
}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<:])")


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


def init_state() -> None:
    st.session_state.setdefault("ledger", SessionLedger())
    st.session_state.setdefault("doc_key", None)
    st.session_state.setdefault("extraction", None)
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("question", "")


def pick_document(settings: Settings) -> tuple[LoadedDocument | None, str | None, str]:
    st.sidebar.header("Document")
    source = st.sidebar.radio("Source", ["Try a sample", "Upload your own"], horizontal=True)
    if source == "Try a sample":
        name = st.sidebar.selectbox("Sample", list(SAMPLES))
        doc = load_document(name, (ROOT / "samples" / name).read_bytes(), settings)
        return doc, name, SAMPLES[name]
    uploaded = st.sidebar.file_uploader(
        "PDF, PNG or JPG, up to 10 MB", type=["pdf", "png", "jpg", "jpeg"]
    )
    doc_type = st.sidebar.selectbox("Document type", ["invoice", "contract"], format_func=str.title)
    if uploaded is None:
        return None, None, doc_type
    return load_document(uploaded.name, uploaded.getvalue(), settings), None, doc_type


def dev_controls(settings: Settings) -> int:
    if settings.dev_mode:
        return 1 if st.sidebar.checkbox("Simulate API failure (1 retry)") else 0
    return 0


def spend_panel(box: DeltaGenerator, settings: Settings) -> None:
    """Drawn last, into a slot reserved at the top, so it already includes this run's request."""
    ledger: SessionLedger = st.session_state.ledger
    box.header("Spend")
    box.metric(
        "This session", f"${ledger.spent_usd:.4f}", help=f"Limit ${settings.session_budget_usd:.2f}"
    )
    today = get_store().spent_on(utc_today())
    box.metric(
        "Today (all visitors)", f"${today:.4f}", help=f"Limit ${settings.daily_budget_usd:.2f}"
    )
    box.caption(f"Model: {settings.model}" + ("" if settings.api_key else " · live requests off"))


def show_messages(outcome_error: str | None, outcome_notice: str | None) -> None:
    if outcome_notice:
        st.info(md_escape(outcome_notice))
    if outcome_error:
        st.error(md_escape(outcome_error))


def fields_table(result: ExtractionResult) -> None:
    frame = pd.DataFrame(to_rows(result))
    styled = frame.style.apply(lambda row: [ISSUE_STYLE if row["issue"] else ""] * len(row), axis=1)
    columns = {
        "field": st.column_config.TextColumn(width="medium"),
        "issue": st.column_config.TextColumn(width="large"),
    }
    st.dataframe(styled, hide_index=True, width="stretch", column_config=columns)
    stem = Path(result.doc_name).stem
    left, right = st.columns(2)
    left.download_button("Download CSV", to_csv(result), f"{stem}.csv", "text/csv")
    right.download_button("Download JSON", to_json(result), f"{stem}.json", "application/json")


def extract_tab(doc: LoadedDocument, sample: str | None, doc_type: str, simulate: int) -> None:
    if st.button("Extract fields", type="primary"):
        outcome = run_extraction(
            doc,
            doc_type,
            get_llm(),
            st.session_state.ledger,
            sample,
            get_settings().cache_dir,
            simulate,
        )
        st.session_state.ledger = outcome.ledger
        st.session_state.extraction = outcome
    outcome = st.session_state.extraction
    if outcome is None:
        st.caption("Click the button to pull the fields out of the document.")
        return
    show_messages(outcome.error, outcome.notice)
    if outcome.result is not None:
        if outcome.result.cached:
            st.caption(":gray-background[cached result]")
        issues = len(outcome.result.flags)
        st.write(f"**{issues} issue(s)** found by our checks." if issues else "All checks passed.")
        fields_table(outcome.result)


def question_input(sample: str | None) -> str:
    if sample:
        cols = st.columns(len(SAMPLE_QUESTIONS[sample]))
        for col, suggestion in zip(cols, SAMPLE_QUESTIONS[sample], strict=True):
            if col.button(suggestion, width="stretch"):
                st.session_state.question = suggestion
    return st.text_input("Your question", key="question", max_chars=500)


def ask_tab(doc: LoadedDocument, sample: str | None, simulate: int) -> None:
    question = question_input(sample)
    if st.button("Ask", type="primary"):
        outcome = run_question(
            [doc],
            question,
            get_llm(),
            st.session_state.ledger,
            sample,
            get_settings().cache_dir,
            simulate,
        )
        st.session_state.ledger = outcome.ledger
        st.session_state.answer = outcome
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
        note = f" · {md_escape(quote.note)}" if quote.note else ""
        st.markdown(f"{STATUS_LABELS[quote.status]} page {quote.page}{note}")
        st.text(f"“{quote.text}”")


def request_log() -> None:
    calls = st.session_state.ledger.calls
    with st.expander(f"Request log ({len(calls)})"):
        if not calls:
            st.caption("No live requests yet in this session.")
            return
        frame = pd.DataFrame([c.__dict__ for c in calls])[LOG_COLUMNS]
        st.dataframe(frame, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Document AI Agent", page_icon="📄", layout="wide")
    init_state()
    settings = get_settings()
    st.title("Document AI Agent")
    st.caption(
        "Extract fields from invoices and contracts, then ask questions. Every quote is checked against the document."
    )
    try:
        doc, sample, doc_type = pick_document(settings)
    except DocumentError as err:
        st.error(md_escape(str(err)))
        doc, sample, doc_type = None, None, ""
    except Exception:  # last resort: never show a traceback to a visitor
        log.exception("Unexpected error while loading a document")
        st.error(UNEXPECTED_ERROR)
        doc, sample, doc_type = None, None, ""
    spend_box = st.sidebar.container()
    simulate = dev_controls(settings)
    if doc is not None:
        show_document(doc, sample, doc_type, simulate)
    else:
        st.info("Upload a document or pick a sample in the sidebar.")
    spend_panel(spend_box, settings)


def show_document(doc: LoadedDocument, sample: str | None, doc_type: str, simulate: int) -> None:
    doc_key = (doc.name, doc_type, len(doc.data))
    if st.session_state.doc_key != doc_key:
        st.session_state.update(doc_key=doc_key, extraction=None, answer=None, question="")
    extract_col, ask_col = st.tabs(["Extract", "Ask"])
    with extract_col:
        extract_tab(doc, sample, doc_type, simulate)
    with ask_col:
        ask_tab(doc, sample, simulate)
    request_log()


main()
