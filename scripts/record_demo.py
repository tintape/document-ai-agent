"""Record the demo video: a real browser session against the running app, with captions and a cursor.

Run: uv run --with playwright python scripts/record_demo.py
Needs Google Chrome and OPENROUTER_API_KEY (three live requests, well under one cent).
Writes docs/demo.mp4 and docs/demo-poster.jpg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PORT = 8503
URL = f"http://localhost:{PORT}"
SIZE = {"width": 1024, "height": 640}
OUT_VIDEO = ROOT / "docs" / "demo.mp4"
OUT_POSTER = ROOT / "docs" / "demo-poster.jpg"
LIVE_TIMEOUT_MS = 90_000

OVERLAY_JS = """
(() => {
  const install = () => {
    if (document.getElementById('__cursor')) return;
    const cursor = document.createElement('div');
    cursor.id = '__cursor';
    cursor.style.cssText = 'position:fixed;z-index:2147483646;width:22px;height:22px;margin:-11px 0 0 -11px;' +
      'border-radius:50%;background:rgba(255,75,75,.35);border:2px solid #ff4b4b;pointer-events:none;' +
      'transition:transform .12s;left:-50px;top:-50px';
    const caption = document.createElement('div');
    caption.id = '__caption';
    caption.style.cssText = 'position:fixed;z-index:2147483645;left:50%;top:14px;transform:translateX(-50%);' +
      'max-width:820px;padding:12px 22px;border-radius:12px;background:rgba(15,17,22,.92);color:#fff;' +
      'font:600 19px/1.35 -apple-system,Segoe UI,Arial,sans-serif;text-align:center;' +
      'box-shadow:0 8px 30px rgba(0,0,0,.45);pointer-events:none;opacity:0;transition:opacity .3s';
    document.body.append(cursor, caption);
    document.addEventListener('mousemove', e => { cursor.style.left = e.clientX + 'px'; cursor.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => cursor.style.transform = 'scale(.7)', true);
    document.addEventListener('mouseup', () => cursor.style.transform = 'scale(1)', true);
    window.__caption = text => { caption.textContent = text; caption.style.opacity = text ? 1 : 0; };
  };
  if (document.body) install(); else document.addEventListener('DOMContentLoaded', install);
})();
"""

INTRO = "Extract fields from invoices and contracts.<br>Ask questions. Every answer is checked."
OUTRO = (
    "Python · Claude Haiku 5.5 · structured outputs · verified quotes<br>"
    "visible retries · real costs · spend limits · 200+ tests"
)
CARD_JS = """
([title, subtitle]) => {
  document.getElementById('__card')?.remove();
  if (!title) return;
  const card = document.createElement('div');
  card.id = '__card';
  card.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:#0e1117;color:#fafafa;' +
    'display:flex;flex-direction:column;align-items:center;justify-content:center;' +
    'font-family:-apple-system,Segoe UI,Arial,sans-serif;text-align:center';
  card.innerHTML = `<div style="font-size:52px;font-weight:800;letter-spacing:-1px">${title}</div>` +
    `<div style="margin-top:20px;font-size:22px;color:#c9ccd3;max-width:860px;line-height:1.45">${subtitle}</div>`;
  document.body.append(card);
}
"""


class Director:
    """Small helpers so the scene list below reads like a script."""

    def __init__(self, page: Page) -> None:
        self.page = page

    def caption(self, text: str, hold_s: float = 0.0) -> None:
        self.page.evaluate("t => window.__caption && window.__caption(t)", text)
        self.pause(hold_s)

    def pause(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def move_to(self, target: Locator) -> None:
        target.scroll_into_view_if_needed()
        box = target.bounding_box()
        if box:
            self.page.mouse.move(
                box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=25
            )
        self.pause(0.35)

    def click(self, target: Locator) -> None:
        self.move_to(target)
        target.click()
        self.pause(0.6)

    def card(self, title: str, subtitle: str, hold_s: float) -> None:
        """A full-screen title card drawn over the app (static text only)."""
        self.page.evaluate(CARD_JS, [title, subtitle])
        self.pause(hold_s)
        self.page.evaluate(CARD_JS, ["", ""])

    def choose_samples(self, names: list[str]) -> None:
        """Add the wanted samples first, then remove the rest, so the page never goes empty."""
        sidebar = self.page.get_by_test_id("stSidebar")
        combo = sidebar.get_by_role("combobox", name="Samples")
        current = [el.get_attribute("aria-label") for el in sidebar.locator("[data-tag]").all()]
        for name in (n for n in names if n not in current):
            self.click(combo)
            self.page.get_by_role("option", name=name, exact=True).click()
            self.pause(0.8)
        self.page.keyboard.press("Escape")
        for name in (n for n in current if n not in names):
            self.click(sidebar.get_by_role("button", name=f"Remove {name}", exact=True))
        self.pause(1.0)

    def wait_text(self, text: str) -> None:
        self.page.get_by_text(text, exact=False).first.wait_for(timeout=LIVE_TIMEOUT_MS)


def scene_extract(d: Director) -> None:
    page = d.page
    d.caption("Pick a sample, or upload your own invoices and contracts.", 2.5)
    d.choose_samples(["invoice_messy_scan.pdf"])
    d.caption("A messy scan: crooked, noisy, no text layer.", 1.5)
    d.click(page.get_by_text("Simulate API failure"))
    d.caption("We'll also make the first API call fail on purpose.", 2.0)
    d.click(page.get_by_role("button", name="Extract fields"))
    d.caption("Claude Haiku 5.5 reads the scan. The failed call is retried automatically.")
    d.wait_text("found by our checks")
    d.pause(1.0)
    d.move_to(page.get_by_text("do not add up", exact=False).first)
    d.caption("Plain code checks the result: the line items don't add up to the total.", 4.0)
    d.move_to(page.get_by_test_id("stDataFrame").first)
    d.caption("Every field has a confidence and a page. Problems are highlighted.", 3.0)
    log = page.get_by_text("Request log", exact=False).first
    d.click(log)
    d.move_to(page.get_by_test_id("stDataFrame").last)
    d.caption("The request log: 2 attempts, real cost from the provider (~$0.0005).", 4.0)


def scene_ask(d: Director) -> None:
    page = d.page
    d.click(page.get_by_text("Simulate API failure"))
    d.choose_samples(["contract_services.pdf"])
    d.click(page.get_by_role("tab", name="Ask"))
    d.caption("Now ask the contract a question.", 1.5)
    d.click(page.get_by_role("button", name="How much notice is needed to end the contract?"))
    d.click(page.get_by_role("button", name="Ask", exact=True))
    d.caption("The answer comes with verbatim quotes...")
    d.wait_text("verified")
    d.pause(0.8)
    d.move_to(page.get_by_text("verified", exact=False).first)
    d.caption("...and our code finds each quote in the PDF text. No trust needed.", 4.0)


def scene_multi(d: Director) -> None:
    page = d.page
    d.choose_samples(["invoice_clean.pdf", "invoice_multipage.pdf"])
    d.click(page.get_by_role("tab", name="Ask"))
    d.caption("Questions can span several documents.", 1.5)
    d.click(page.get_by_role("button", name="Which document has the largest total?"))
    d.click(page.get_by_role("button", name="Ask", exact=True))
    d.wait_text("verified")
    d.pause(0.8)
    d.move_to(page.get_by_text("verified", exact=False).first)
    d.caption("Each quote names the file and page it was verified in.", 4.0)
    d.move_to(page.get_by_test_id("stSidebar").get_by_text("This session"))
    d.caption(
        "Spend is capped per visitor and per day. No API key? Samples fall back to cached results.",
        4.5,
    )
    d.caption("")


def wait_for_server(timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("Streamlit did not start")


def record(video_dir: Path) -> tuple[Path, float]:
    """Returns the raw video and the second at which the app was ready (everything before is cut)."""
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport=SIZE, record_video_dir=str(video_dir), record_video_size=SIZE,
                                      color_scheme="dark", device_scale_factor=1)  # fmt: skip
        context.add_init_script(OVERLAY_JS)
        page = context.new_page()
        started = time.monotonic()
        d = Director(page)
        page.goto(URL)
        page.get_by_role("button", name="Extract fields").wait_for(timeout=30_000)
        page.evaluate(OVERLAY_JS)
        page.evaluate(CARD_JS, ["Document AI Agent", INTRO])
        ready_s = time.monotonic() - started
        d.pause(3.5)
        page.evaluate(CARD_JS, ["", ""])
        d.pause(0.8)
        scene_extract(d)
        scene_ask(d)
        scene_multi(d)
        d.card("Document AI Agent", OUTRO, 4.0)
        video_path = Path(page.video.path())
        context.close()
        browser.close()
    return video_path, ready_s


def encode(webm: Path, start_s: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.2f}", "-i", str(webm), "-c:v", "libx264", "-preset", "slow",
         "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT_VIDEO)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "24", "-i", str(OUT_VIDEO), "-frames:v", "1",
         "-q:v", "3", str(OUT_POSTER)],
        check=True,
    )  # fmt: skip


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    env = {**os.environ, "DEMO_DEV": "1"}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        wait_for_server()
        with tempfile.TemporaryDirectory() as tmp:
            webm, ready_s = record(Path(tmp))
            encode(webm, ready_s + 0.15)
    finally:
        server.terminate()
        server.wait(timeout=10)
    print(f"Wrote {OUT_VIDEO.relative_to(ROOT)} and {OUT_POSTER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
