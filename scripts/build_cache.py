"""Run every bundled sample through the live API once and store the results in samples/cache/.

Run: uv run python scripts/build_cache.py   (needs OPENROUTER_API_KEY; costs about one cent)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doc_agent.ask import ask  # noqa: E402
from doc_agent.budget import BudgetStore  # noqa: E402
from doc_agent.cache import SAMPLE_QUESTIONS, SAMPLES, save_answer, save_extraction  # noqa: E402
from doc_agent.config import load_settings  # noqa: E402
from doc_agent.documents import load_document  # noqa: E402
from doc_agent.extract import extract  # noqa: E402
from doc_agent.llm import LLMClient  # noqa: E402


def main() -> None:
    settings = load_settings(os.environ, ROOT)
    if not settings.api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    llm = LLMClient(settings, BudgetStore(settings.budget_file))
    total = 0.0
    for name, doc_type in SAMPLES.items():
        doc = load_document(name, (ROOT / "samples" / name).read_bytes(), settings)
        result = extract(doc, doc_type, llm, session_spent=0.0)
        save_extraction(result, settings.cache_dir)
        total += result.call.cost_usd
        print(f"{name}: {len(result.flags)} issue(s), ${result.call.cost_usd:.5f}")
        for question in SAMPLE_QUESTIONS[name]:
            answer = ask([doc], question, llm, session_spent=0.0)
            save_answer(name, answer, settings.cache_dir)
            total += answer.call.cost_usd
            statuses = ", ".join(q.status for q in answer.quotes) or "no quotes"
            print(f"  Q: {question} -> {answer.answer[:70]!r} [{statuses}]")
    print(f"Total cost: ${total:.4f}")


if __name__ == "__main__":
    main()
