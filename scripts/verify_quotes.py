"""Check that every quote in data/evidence.csv appears in the document it cites.

For each URL the text is taken from the Wayback snapshot that sources.csv records
for it (the version that was read), else from the live page. Hugging Face model
pages add the raw README, arXiv abstract pages the PDF. HTML is reduced to its
visible text, PDFs need pypdf (run with a Python that has it). Quotes are
compared after normalisation (scripts/rubric.py): Unicode forms, quotes and
dashes, hyphenation and whitespace, case; "found (loose)" ignores punctuation
and markup. A quote may skip text with "...".

    python scripts/verify_quotes.py            # writes data/quote_checks.csv
    python scripts/verify_quotes.py --refresh  # re-download cached documents

Documents are cached in ~/.cache/sovereign-models/docs.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import html
import io
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from rubric import find_quote, needs_quote, quote_key
from scorecard import DATA, read

OUT = DATA / "quote_checks.csv"
FIELDS = ["key", "url", "status", "method", "checked"]
CACHE = Path.home() / ".cache" / "sovereign-models" / "docs"
UA = "Mozilla/5.0 (sovereign-models quote check; +https://github.com/here-be-dragons-ai/sovereign-models)"
TOKEN = Path.home() / ".cache" / "huggingface" / "token"


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "td", "th"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(raw):
    p = _Text()
    p.feed(raw)
    return html.unescape(" ".join(p.out))


def pdf_text(data):
    try:
        import pypdf
    except ImportError:
        return None
    try:
        return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(data)).pages)
    except Exception:  # noqa: BLE001 - a broken PDF is just unreadable here
        return None


def fetch(url):
    headers = {"User-Agent": UA}
    if url.startswith("https://huggingface.co/") and TOKEN.exists():
        headers["Authorization"] = f"Bearer {TOKEN.read_text().strip()}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        data = r.read()
        # The Wayback Machine serves some raw snapshots still gzip-encoded.
        return (gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data), r.headers.get("Content-Type", "")


def to_text(data, ctype, url):
    if data[:5] == b"%PDF-" or "pdf" in ctype:
        return pdf_text(data)
    raw = data.decode("utf-8", errors="replace")
    if "html" in ctype or raw.lstrip().lower().startswith(("<!doctype", "<html")):
        return html_text(raw)
    return raw


def raw_snapshot(archive_url):
    """The archived page without the Wayback toolbar."""
    return re.sub(r"(web\.archive\.org/web/\d+)/", r"\1id_/", archive_url, count=1)


def extra_urls(url):
    # Blob pages show a viewer; the file itself is what the source says.
    m = re.match(r"https://github\.com/([\w.\-]+/[\w.\-]+)/blob/(.+)$", url)
    if m:
        return [f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}"]
    m = re.match(r"https://huggingface\.co/(.+?)/blob/(.+)$", url)
    if m:
        return [f"https://huggingface.co/{m.group(1)}/resolve/{m.group(2)}"]
    m = re.match(r"https://huggingface\.co/(?!datasets/|spaces/)([\w.\-]+/[\w.\-]+)/?$", url)
    if m:
        return [f"https://huggingface.co/{m.group(1)}/raw/main/README.md"]
    m = re.match(r"https://huggingface\.co/datasets/([\w.\-]+/[\w.\-]+)/?$", url)
    if m:
        return [f"https://huggingface.co/datasets/{m.group(1)}/raw/main/README.md"]
    m = re.match(r"https://arxiv\.org/abs/([\w.\-/]+?)(v\d+)?$", url)
    if m:
        return [f"https://arxiv.org/pdf/{m.group(1)}"]
    return []


def document(url, archive, refresh):
    """(text, method) for a URL, cached; text is None if nothing could be read."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".txt")
    meta = path.with_suffix(".method")
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8"), meta.read_text().strip() if meta.exists() else "cache"
    texts, methods = [], []
    for kind, u in ([("snapshot", raw_snapshot(archive))] if archive else []) + [("live", url)] + \
            [("live", e) for e in extra_urls(url)]:
        if kind == "live" and u == url and texts:
            continue  # the snapshot was readable; the live page is only a fallback
        try:
            data, ctype = fetch(u)
        except Exception as e:  # noqa: BLE001 - any fetch failure just means no text from here
            methods.append(f"{kind} failed ({type(e).__name__})")
            continue
        text = to_text(data, ctype, u)
        if text is None:
            methods.append(f"{kind}: pdf without pypdf")
            continue
        texts.append(text)
        methods.append(kind if u in (url, raw_snapshot(archive or "")) else f"{kind} {u.split('/')[-1] or u}")
    if not texts:
        return None, "; ".join(methods)
    text, method = "\n\n".join(texts), "; ".join(methods)
    path.write_text(text, encoding="utf-8")
    meta.write_text(method)
    return text, method


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    evidence = read("evidence.csv")
    archives = {}
    for s in read("sources.csv"):
        if s["archive_url"]:
            archives.setdefault(s["url"], s["archive_url"])
    today = dt.date.today().isoformat()
    rows, seen = [], set()
    for e in evidence:
        if not needs_quote(e["question"], e["answer"]):
            continue
        key = quote_key(e["url"], e["quote"])
        if key in seen:
            continue
        seen.add(key)
        if not e["quote"].strip():
            status, method = "no quote", ""
        else:
            text, method = document(e["url"], e.get("archive_url") or archives.get(e["url"]), args.refresh)
            status = find_quote(e["quote"], text) if text else "no text"
        rows.append(dict(key=key, url=e["url"], status=status, method=method, checked=today))
        print(f"{status:14s} {e['subject']}/{e['question']}: {e['quote'][:70]}", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no quotes to check", file=sys.stderr)


if __name__ == "__main__":
    main()
