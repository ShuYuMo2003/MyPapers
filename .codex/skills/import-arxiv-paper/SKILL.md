---
name: import-arxiv-paper
description: Import an arXiv paper into a local research knowledge base under a user-specified category such as `./paper/egocentric`, storing each paper in its own folder with `layout.md` and optionally `bilingual.pdf`. Use when Codex is given an arXiv URL or arXiv id and should archive the paper, preserve a readable layout-extracted version for analysis, and generate a side-by-side English-Chinese bilingual PDF by reusing `helper_repo/zotero-pdf2zh` with local translator config from `.codex/local/pdf2zh.json` (DeepSeek default, OpenAI fallback). If the user does not provide the category, stop immediately and ask for it instead of inferring one. If the user provides a category that has not appeared before, normalize it into a folder-safe slug, create the new category folder, and register it in the skill's category list.
---

# Import Arxiv Paper

## Workflow

Follow this workflow every time:

1. If the user did not provide the paper category, ask for it immediately. Do not infer or suggest a category.
2. Parse the arXiv link or id and run `scripts/import_arxiv_paper.py` with the user-provided category.
3. Review the script output for the normalized id, paper folder path, layout markdown path, PDF retention status, and bilingual PDF path.
4. If bilingual generation fails, or if the translator needs to be changed, read `references/troubleshooting.md` before patching config or helper code.
5. Read `layout.md` before writing any interpretation.

## Quick Start

Run:

```powershell
python ./.codex/skills/import-arxiv-paper/scripts/import_arxiv_paper.py `
  --url <arxiv-url-or-id> `
  --category <category-name> `
  --kb-root .
```

If the user wants the paper imported even when translation is blocked, rerun with:

```powershell
python ./.codex/skills/import-arxiv-paper/scripts/import_arxiv_paper.py `
  --url <arxiv-url-or-id> `
  --category <category-name> `
  --kb-root . `
  --skip-bilingual
```

If `pypdf` is missing, install it first:

```powershell
python -m pip install pypdf
```

Default knowledge-base layout:

```text
<kb-root>/
  paper/
    egocentric/
      <[ShortName] Full Title>/
        layout.md
        bilingual.pdf
```

Choose the paper folder name only after generating and reading `layout.md`.

- If the paper has a clear short name or abbreviation, use `[ShortName] Full Title`.
- Prefer a short name explicitly used by the paper itself, not one invented ad hoc.
- The short name may appear in the title, abstract, or introduction.
- If there is no clear short name, use just the full title.
- Do not prefix the folder name with the arXiv id.

## Bilingual PDF Setup

The bilingual helper reads local translator settings from `.codex/local/pdf2zh.json`.

- Prefer DeepSeek as the local default translator unless the user explicitly wants another provider.
- Keep OpenAI settings only as an optional fallback; do not assume they are usable.
- Read `references/troubleshooting.md` before changing providers, editing translator config, or debugging `pdf2zh_next`.

## Category Input

Use the category provided by the user. Do not classify the paper automatically.

For category state and definitions, read:

- `references/categories.txt` for the current registered category slugs
- `references/category-reference.md` for the category rules and descriptions

## Analysis Style

Anchor claims in the PDF text or abstract. If a claim is an inference, label it as an inference.

Match the user's requested conversation language. If the user asks for Chinese, conduct the rest of the interaction in Chinese.

Prefer this reading order when extracting evidence:

1. `layout.md`
2. abstract metadata

When the paper is video-heavy, explicitly answer:

- Does the video model predict pixels, latent states, actions, or all three?
- Is it used for planning, control, representation learning, synthetic data, or evaluation?
- What would be required to make it useful for a robot setting rather than a pure internet-video setting?
- What is the likely bridge from the paper to VLA or world-model systems?

## Files Created By The Script

The script creates:

- `paper/<category>/<paper-folder>/layout.md`
- `paper/<category>/<paper-folder>/bilingual.pdf`

During import, the script uses a temporary `paper.pdf` to extract `layout.md` and render `bilingual.pdf`.

- After `bilingual.pdf` is created successfully, it deletes `paper.pdf`.
- If bilingual generation fails or the run uses `--skip-bilingual`, `paper.pdf` remains in the paper folder for retry/debugging.

Do not keep extra log-like or database-like artifacts in the project tree for now.

## Resources

- `scripts/import_arxiv_paper.py`: download, extract layout-aware text, and call the bilingual helper under the user-specified category.
- `../../../helper_repo/zotero-pdf2zh/local_bilingual_pdf.py`: reuse `pdf2zh_next` locally and save the dual-language PDF as `bilingual.pdf`.
- `references/troubleshooting.md`: read when category registration fails, bilingual generation fails, or the translator/config needs to be changed.
- `references/categories.txt`: registered category slugs for this project.
- `references/category-reference.md`: human-readable category rules and descriptions.
- `.codex/local/pdf2zh.json`: local translator configuration for the bilingual helper.
