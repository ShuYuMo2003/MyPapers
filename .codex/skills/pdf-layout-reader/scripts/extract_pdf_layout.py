#!/usr/bin/env python
"""Extract a readable markdown layout from a local PDF."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import fitz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Local PDF path")
    parser.add_argument("--output", help="Optional output layout.md path")
    parser.add_argument("--max-pages", type=int, default=8, help="Pages to extract from the PDF")
    return parser.parse_args()


def extract_layout(pdf_path: Path, max_pages: int) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    markdown_parts: list[str] = []
    page_count = len(doc)

    for page_index in range(min(max_pages, page_count)):
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
    return "\n".join(markdown_parts).strip() + "\n", page_count


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")

    if args.output:
        layout_path = Path(args.output).expanduser().resolve()
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir: str | None = None
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="pdf-layout-reader-"))
        layout_path = temp_root / "layout.md"
        temp_dir = str(temp_root)

    layout_markdown, page_count = extract_layout(pdf_path, args.max_pages)
    layout_path.write_text(layout_markdown, encoding="utf-8")

    payload = {
        "pdf_path": str(pdf_path),
        "layout_markdown_path": str(layout_path),
        "temp_dir": temp_dir,
        "page_count": page_count,
        "max_pages_extracted": args.max_pages,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
