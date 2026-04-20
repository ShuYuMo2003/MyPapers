---
name: pdf-layout-reader
description: Read a local PDF and generate a temporary or user-specified `layout.md` file for downstream analysis. Use when Codex is given a PDF file instead of a precomputed layout file and needs a quick, readable text extraction pass before summarization, note writing, or method analysis.
---

# PDF Layout Reader

## Workflow

Follow this workflow every time:

1. Confirm the input is a local PDF path.
2. Run `scripts/extract_pdf_layout.py` on the PDF.
3. If the user did not specify an output path, let the script create a temporary directory and temporary `layout.md`.
4. Read the generated `layout.md` for downstream analysis.
5. Keep the temporary layout file unless the user asks to clean it up.

## Quick Start

Run:

```powershell
python ./pdf-layout-reader/scripts/extract_pdf_layout.py `
  --pdf <local-pdf-path>
```

Write to a chosen path:

```powershell
python ./pdf-layout-reader/scripts/extract_pdf_layout.py `
  --pdf <local-pdf-path> `
  --output <target-layout-md-path>
```

## Output

The script prints JSON with:

- `pdf_path`
- `layout_markdown_path`
- `temp_dir`
- `page_count`
- `max_pages_extracted`

If `--output` is provided, `temp_dir` is `null`.

## Extraction Rules

- Use the layout file as a convenience layer for reading, not as a perfect reconstruction of the PDF.
- Preserve page boundaries with `# Page N` headers.
- Prefer a fast, readable extraction over complex document reconstruction.
- If the PDF is image-only or extraction quality is poor, say so explicitly after reading the output.

## Resources

- `scripts/extract_pdf_layout.py`: extract readable text blocks from a local PDF and write a temporary or explicit `layout.md`.
