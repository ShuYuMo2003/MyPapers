#!/usr/bin/env python
"""Download an arXiv PDF, extract readable text, and scaffold a compact paper folder."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz


ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)")
CATEGORY_SAFE_RE = re.compile(r"[^a-z0-9-]+")
WINDOWS_BAD_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SKILL_ROOT = Path(__file__).resolve().parents[1]
CATEGORY_REGISTRY = SKILL_ROOT / "references" / "categories.txt"
DEFAULT_HEADERS = {"User-Agent": "codex-import-arxiv-paper/1.0 (+https://arxiv.org)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="arXiv URL or bare arXiv id")
    parser.add_argument("--kb-root", default=".", help="Knowledge base root directory")
    parser.add_argument("--category", required=True, help="Paper category folder")
    parser.add_argument("--max-pages", type=int, default=8, help="Pages to extract from the PDF")
    parser.add_argument(
        "--skip-bilingual",
        action="store_true",
        help="Skip generating bilingual.pdf through helper_repo/zotero-pdf2zh",
    )
    return parser.parse_args()


def normalize_category(value: str) -> str:
    category = value.strip().lower().replace("_", "-").replace(" ", "-")
    category = CATEGORY_SAFE_RE.sub("-", category)
    category = re.sub(r"-{2,}", "-", category).strip("-")
    if not category:
        raise ValueError("Category must not be empty")
    return category


def register_category(category: str) -> None:
    CATEGORY_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if CATEGORY_REGISTRY.exists():
        existing = [
            line.strip()
            for line in CATEGORY_REGISTRY.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        existing = []
    if category not in existing:
        CATEGORY_REGISTRY.write_text("\n".join(existing + [category]) + "\n", encoding="utf-8")


def ensure_category_dirs(kb_root: Path, category: str) -> None:
    (kb_root / "paper" / category).mkdir(parents=True, exist_ok=True)


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        if "arxiv.org" not in parsed.netloc:
            raise ValueError(f"Unsupported host: {parsed.netloc}")
        match = ARXIV_ID_RE.search(value)
        if match:
            return match.group("id")
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            return path_parts[-1]
        raise ValueError("Could not parse arXiv id from URL")
    return value.replace("arXiv:", "")


def urlopen_with_retry(
    request: str | urllib.request.Request,
    *,
    timeout: int,
    attempts: int = 5,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in retry_statuses or attempt == attempts:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == attempts:
                raise
        time.sleep(min(2 ** (attempt - 1), 20))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed without a captured error")


def fetch_atom_metadata(arxiv_id: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urlopen_with_retry(request, timeout=30) as response:
        xml_text = response.read()
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise RuntimeError(f"No arXiv entry returned for {arxiv_id}")
    title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
    summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
    published = entry.findtext("atom:published", default="", namespaces=ns) or ""
    authors = [
        author.findtext("atom:name", default="", namespaces=ns) or ""
        for author in entry.findall("atom:author", ns)
    ]
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": summary,
        "published": published,
        "authors": authors,
    }


def extract_meta_content(html_text: str, meta_name: str) -> list[str]:
    pattern = re.compile(
        rf'<meta\s+(?:name|property)=["\']{re.escape(meta_name)}["\']\s+content=["\'](.*?)["\']',
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [html.unescape(match).strip() for match in pattern.findall(html_text) if match.strip()]


def strip_html_tags(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def fetch_abs_metadata(arxiv_id: str) -> dict:
    url = f"https://arxiv.org/abs/{urllib.parse.quote(arxiv_id)}"
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urlopen_with_retry(request, timeout=30) as response:
        html_text = response.read().decode("utf-8", errors="replace")

    titles = extract_meta_content(html_text, "citation_title")
    authors = extract_meta_content(html_text, "citation_author")
    published = next(iter(extract_meta_content(html_text, "citation_date")), "")

    abstract_match = re.search(
        r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(?P<body>.*?)</blockquote>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    abstract = ""
    if abstract_match:
        abstract = strip_html_tags(abstract_match.group("body"))
        abstract = re.sub(r"^Abstract:\s*", "", abstract, flags=re.IGNORECASE)

    title = titles[0] if titles else ""
    if not title:
        title_match = re.search(r"<title>(?P<title>.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            title = strip_html_tags(title_match.group("title")).replace("arXiv:", "").strip()

    if not title:
        raise RuntimeError(f"Could not parse metadata from arXiv abstract page for {arxiv_id}")

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "published": published,
        "authors": authors,
    }


def fetch_metadata(arxiv_id: str) -> dict:
    try:
        return fetch_atom_metadata(arxiv_id)
    except urllib.error.HTTPError as exc:
        if exc.code != 429:
            raise
    return fetch_abs_metadata(arxiv_id)


def download_pdf(arxiv_id: str, destination: Path) -> None:
    pdf_url = f"https://arxiv.org/pdf/{urllib.parse.quote(arxiv_id)}.pdf"
    request = urllib.request.Request(pdf_url, headers=DEFAULT_HEADERS)
    with urlopen_with_retry(request, timeout=60) as response:
        destination.write_bytes(response.read())


def extract_layout(pdf_path: Path, max_pages: int) -> str:
    doc = fitz.open(pdf_path)
    markdown_parts: list[str] = []

    for page_index in range(min(max_pages, len(doc))):
        page = doc[page_index]
        page_dict = page.get_text("dict")
        markdown_parts.append(f"# Page {page_index + 1}")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            line_texts = []
            for line in block.get("lines", []):
                spans = [(span.get("text") or "").strip() for span in line.get("spans", [])]
                spans = [text for text in spans if text]
                if spans:
                    line_texts.append(" ".join(spans).strip())

            block_text = "\n".join(line_texts).strip()
            if block_text:
                markdown_parts.append(block_text)
                markdown_parts.append("")

    doc.close()
    return "\n".join(markdown_parts).strip() + "\n"


def detect_short_name(title: str, layout_markdown: str) -> tuple[str | None, str]:
    title = " ".join(title.split())
    title_match = re.match(r"^(?P<short>[A-Z][A-Za-z0-9+-]{1,31})\s*:\s*(?P<rest>.+)$", title)
    if title_match:
        return title_match.group("short"), title_match.group("rest").strip()

    search_text = "\n".join(layout_markdown.splitlines()[:220])
    patterns = [
        r"\bwe present (?P<short>[A-Z][A-Za-z0-9+-]{1,31})\b",
        r"\bour method(?:,)? (?P<short>[A-Z][A-Za-z0-9+-]{1,31})\b",
        r"\bcalled (?P<short>[A-Z][A-Za-z0-9+-]{1,31})\b",
        r"\b(?P<short>[A-Z][A-Za-z0-9+-]{1,31})\s+is a\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, search_text, flags=re.IGNORECASE)
        if match:
            short = match.group("short")
            if short and short.lower() not in {"figure", "section", "table"}:
                return short, title

    return None, title


def sanitize_folder_name(value: str) -> str:
    cleaned = WINDOWS_BAD_CHARS_RE.sub("", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    if not cleaned:
        raise ValueError("Folder name became empty after sanitization")
    return cleaned


def build_paper_folder_name(title: str, layout_markdown: str) -> tuple[str, str | None]:
    short_name, display_title = detect_short_name(title, layout_markdown)
    folder_name = f"[{short_name}] {display_title}" if short_name else display_title
    return sanitize_folder_name(folder_name), short_name


def ensure_paper_dir(kb_root: Path, category: str, paper_folder: str) -> Path:
    paper_dir = kb_root / "paper" / category / paper_folder
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir


def relative_display(path: Path, kb_root: Path) -> str:
    try:
        return str(path.relative_to(kb_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def generate_bilingual_pdf(kb_root: Path, pdf_path: Path, bilingual_pdf_path: Path) -> None:
    helper_script = kb_root / "helper_repo" / "zotero-pdf2zh" / "local_bilingual_pdf.py"
    if not helper_script.exists():
        raise FileNotFoundError(
            "Missing helper_repo/zotero-pdf2zh/local_bilingual_pdf.py for bilingual generation"
        )
    subprocess.run(
        [
            sys.executable,
            str(helper_script),
            "--pdf",
            str(pdf_path),
            "--output",
            str(bilingual_pdf_path),
        ],
        check=True,
        cwd=kb_root,
    )


def main() -> int:
    args = parse_args()
    kb_root = Path(args.kb_root).expanduser().resolve()
    arxiv_id = normalize_arxiv_id(args.url)
    meta = fetch_metadata(arxiv_id)
    category = normalize_category(args.category)
    register_category(category)
    ensure_category_dirs(kb_root, category)

    category_dir = kb_root / "paper" / category

    temp_pdf_path = category_dir / f"{arxiv_id}.tmp.pdf"
    download_pdf(arxiv_id, temp_pdf_path)
    layout_markdown = extract_layout(temp_pdf_path, args.max_pages)
    paper_folder_name, short_name = build_paper_folder_name(meta["title"], layout_markdown)
    paper_dir = ensure_paper_dir(kb_root, category, paper_folder_name)

    pdf_path = paper_dir / "paper.pdf"
    layout_md_path = paper_dir / "layout.md"
    bilingual_pdf_path = paper_dir / "bilingual.pdf"

    if pdf_path.exists():
        pdf_path.unlink()
    temp_pdf_path.replace(pdf_path)
    layout_md_path.write_text(layout_markdown, encoding="utf-8")

    bilingual_status = "skipped"
    if not args.skip_bilingual:
        generate_bilingual_pdf(kb_root, pdf_path, bilingual_pdf_path)
        if pdf_path.exists():
            pdf_path.unlink()
        bilingual_status = "created"

    payload = {
        "arxiv_id": arxiv_id,
        "title": meta["title"],
        "short_name": short_name,
        "category": category,
        "paper_folder": relative_display(paper_dir, kb_root),
        "pdf_path": relative_display(pdf_path, kb_root) if pdf_path.exists() else None,
        "layout_markdown_path": relative_display(layout_md_path, kb_root),
        "bilingual_pdf_path": relative_display(bilingual_pdf_path, kb_root)
        if bilingual_pdf_path.exists()
        else None,
        "bilingual_status": bilingual_status,
        "max_pages_extracted": args.max_pages,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Error: bilingual helper failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
