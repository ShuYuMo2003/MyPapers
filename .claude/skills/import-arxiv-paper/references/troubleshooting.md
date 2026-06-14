# Troubleshooting

Read this file when import succeeds only partially, `bilingual.pdf` generation fails, the translator needs to be changed, or the paper folder keeps `paper.pdf` after a run.

## Category Registration

- `scripts/import_arxiv_paper.py` normally normalizes new categories, appends them to `references/categories.txt`, and creates `paper/<category>/`.
- If registration fails because the skill directory is not writable in the current environment, manually append the normalized slug to `references/categories.txt`, create `paper/<category>/`, and rerun.
- If the new category meaning is stable, add a short description to `references/category-reference.md`.

## Translator Configuration

- Local bilingual translation config lives at `.codex/local/pdf2zh.json`.
- Preferred default:

```json
{
  "translator": "deepseek",
  "deepseek": {
    "model": "deepseek-v4-flash",
    "api_key": "<your-deepseek-key>"
  }
}
```

- `helper_repo/zotero-pdf2zh/local_bilingual_pdf.py` still supports an `openai` block as fallback, but do not assume the OpenAI account has quota.
- The helper also accepts environment-variable fallbacks such as `DEEPSEEK_API_KEY`, `PDF2ZH_DEEPSEEK_API_KEY`, `PDF2ZH_DEEPSEEK_MODEL`, and `PDF2ZH_DEEPSEEK_ENABLE_JSON_MODE`.

## Windows And Sandbox Pitfalls

- `pdf2zh_next` may try to write config and cache files under `~/.config/pdf2zh`, `~/.cache/pdf2zh_next`, and BabelDOC cache paths. The local helper works around this by redirecting `HOME`, `USERPROFILE`, `XDG_CONFIG_HOME`, and `XDG_CACHE_HOME` into `helper_repo/zotero-pdf2zh/.sandbox-home`.
- `pdf2zh_next` can be fragile on Windows when given long paper paths or folder names containing square brackets. The local helper stages the source PDF to `.pdf2zh-output/input.pdf` before translation; preserve that behavior when debugging or rewriting the helper.

## Bilingual Failure Handling

- Do not trust process exit code alone. `pdf2zh_next` can return without producing a `*.dual.pdf`; inspect the output directory and captured stderr/stdout.
- Provider failures may surface as authentication errors, quota errors, retries, or a missing output PDF. Fix config first, then rerun.
- If the user wants the paper imported even when translation is blocked, rerun `scripts/import_arxiv_paper.py` with `--skip-bilingual`.
- If bilingual generation fails or is skipped, `paper.pdf` remains in the paper folder intentionally so the next run can retry translation without re-downloading.
