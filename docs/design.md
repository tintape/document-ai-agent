# Document AI Agent — design

## Why

A demo of a document AI feature built the way production features should be built. A visitor should understand in one or two minutes that the app does not just call a model: it checks the model's work, shows what every request costs, survives API failures and caps spending.

The most common real-world request in this space is "pull data out of documents and answer questions about them": invoices, contracts, forms, knowledge bases. The demo covers exactly that.

## Goals

1. Upload a PDF or a photo of a document and get a table of extracted fields.
2. Ask a question about one or several documents and get an answer with quotes and page numbers.
3. Show the production side on screen: cost per request, visible retries, highlighted fields that are uncertain or fail a check, daily and per-session spend limits.
4. Export the extracted data as CSV and JSON.
5. A clean repository: README, tests, one-command start.
6. Ready to host publicly.

## Non-goals

- Accounts, multiple users, storing documents between sessions.
- Polished visual design. Clear and tidy is enough.
- Search over large archives (vector store). The README says how it would scale.
- Automatic document type detection. The visitor picks the type.

## Anti-goals (a failure even if it "works")

- A visitor click ends in a stack trace instead of a clear message (missing key, used-up budget, API error, broken file).
- Invented claims in the README or UI: clients, accuracy numbers, "used by".
- Real personal data in the samples.
- Code that looks generated and unreviewed: huge files, dead code, no tests.

## Constraints

- Python 3.11+, official `anthropic` SDK.
- Provider: OpenRouter's Anthropic-compatible Messages endpoint (`https://openrouter.ai/api`), key sent as a bearer token. Base URL, key and model are configuration only; pointing them at `api.anthropic.com` switches to the direct API (the key is then sent as `x-api-key`).
- Model: Claude Haiku 5.5 (`anthropic/claude-haiku-5.5` on OpenRouter), $0.10 / $0.50 per million tokens for prompts up to 100K tokens, $0.50 / $2.50 above. OpenRouter is asked to prefer Anthropic's own endpoint, with fallbacks allowed.
- Key only from `OPENROUTER_API_KEY` in the environment or a local, git-ignored `.env`.
- Input: PDF, PNG or JPG, up to 10 MB and 20 pages each.
- All visible text is English.

## Approaches considered

| Approach | For | Against |
|---|---|---|
| **Streamlit app over a UI-free core package** (chosen) | Fast to build, Python only, free hosting options | Standard look |
| FastAPI backend + small HTML/JS front end | Shows more backend work | Twice the UI work |
| CLI + notebook | Cheapest | Nothing for a visitor to click |

The core (`doc_agent/`) has no Streamlit dependency, so an HTTP API can be added later without rework.

## How it works

**Extraction.** One request per document. A PDF goes to the model as a `document` block (the model reads scans itself, no separate OCR), an image as an `image` block. The answer is constrained with structured outputs to a Pydantic schema; every field carries a value, a confidence (`high` / `medium` / `low`) and a page number. Two schemas: invoice (vendor, buyer, number, dates, currency, line items, tax, total) and contract (parties, signing date, term, amount, payment terms, termination notice, governing law).

**Deterministic checks** (plain code, no AI): line items plus tax must match the total (tolerance: the larger of 1 cent and 0.5%), dates must be readable and the due date not before the invoice date (skipped when a slash date is ambiguous between day-first and month-first), required fields present, a tax rate in place of a tax amount flagged, a missing due date noted. Uncertain or failing fields are highlighted with the reason.

**Questions.** OpenRouter does not pass Anthropic's built-in citations through (checked on 2026-11-27), so our code verifies quotes itself:
1. The model answers with structured output: `answer` plus `quotes` (document, page, verbatim text).
2. The code extracts each PDF page's text locally (`pypdf`) and looks for the quote on the stated page, then on the document's other pages, after folding case, spacing, quotes, dashes, ligatures and soft hyphens.
3. Found: `verified` (with the real page if the model named the wrong one). Not found: `not found in document`. Scans, scanned pages and very short quotes: `unverified`, because there is no text to check against.

**Production side.**
- Cost comes from the provider's `usage.cost`; tokens times the price table is the fallback. Every answered request is billed and counted, including refusals and truncated answers.
- Retries: the SDK's own retries are off; our loop retries 429, 5xx, connection errors and timeouts (3 attempts, backoff 1 s then 2 s plus jitter) so attempts show in the request log. A "simulate API failure" switch exists only with `DEMO_DEV=1`.
- Spend limits: $1 per UTC day across all visitors (file counter with thread and file locks) and $0.10 per browser session.
- Bundled samples have pre-computed results, shown with a "cached result" badge when the key is missing or revoked, the budget is used up or the API is down. Uploads are always live.
- Untrusted text (file names, model output) is never rendered as markdown; CSV cells that look like spreadsheet formulas are neutralized; visitors never see tracebacks.

**Tests.** pytest on the core with a fake client and with the real SDK over a mocked transport; one opt-in live test.
