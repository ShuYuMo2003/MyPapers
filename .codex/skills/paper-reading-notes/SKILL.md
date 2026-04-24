---
name: paper-reading-notes
description: Interactively read and discuss a paper while keeping optional reading notes in the paper's own folder. Default to discussion-first behavior, and only write or revise `notes.md` when the user explicitly asks for notes, a summary, or a written recap.
---

# Paper Reading Notes

Use this skill as a lightweight reading companion.

The goal is to help the user think while reading, not to produce a polished report. Answer questions in the moment, keep the conversation fluid, and do not write notes by default during the discussion. Only save notes when the user explicitly asks for a written summary, recap, or note update.

## Workflow

1. Identify the target paper folder and its `layout.md`.
2. Read `layout.md` before answering substantive paper questions.
3. Continue the conversation interactively.
4. Only if the user explicitly asks for notes or a written recap, create or update `notes.md` in the same paper folder.

## Finding The Paper

Prefer the most explicit source first:

- A user-provided `layout.md` path
- A user-provided paper folder path that contains `layout.md`
- The paper currently open in the IDE or editor, when a specific paper tab or paper-folder tab is clearly visible in context
- A paper folder or `layout.md` already established in the current conversation

If multiple open tabs are visible, prefer the one that most clearly points to a paper folder containing `layout.md` or `notes.md`.
If the paper cannot be identified safely, ask a short clarifying question.

## Reading Behavior

Read `layout.md` before interpreting the paper.

Use the layout as the primary source of truth for:

- title and short name
- abstract and stated contributions
- method details
- evidence for claims

If a point is not directly supported by the layout text, label it as an inference.

## Interaction Style

Treat the session as ongoing collaborative reading.

- Answer the user's question directly.
- If the user shares a good sentence, idea, criticism, or comparison, react to it and keep it in conversational memory during the session.
- If the user wants to focus on a section, stay local to that section instead of drifting into a full-paper summary.
- Keep answers clear and natural.

Do not turn every turn into a large summary. Stay responsive to the user's current reading thread.
Do not proactively write notes unless the user clearly asks for written notes or a summary to be saved.

## Notes File

Write notes to `<paper-folder>/notes.md`.

If `notes.md` does not exist, create it.
If it already exists, preserve prior content and append or lightly reorganize only when helpful.

When notes are requested, optimize them for later rereading rather than minimal logging. Use tables, bold, italics, and short sections when they improve scanability.
Write in the user's language unless asked otherwise.
Prefer straightforward, plain wording that sounds close to how the user has been describing the paper.

Useful note types include:

- key claims
- promising ideas
- quotes or near-quotes the user highlighted
- open questions
- confusions or caveats
- possible relevance to the user's research
- comparisons to other papers mentioned in the conversation

## Notes Format

Use a readable study-note structure rather than a terse log when the user asks for a summary. Keep the structure simple and optimized for rereading. A good default is:

```markdown
# Notes

## 3-Minute Version
- ...

## Paper
- A small table with title / folder / quick feeling

## Quick Take
- ...

## Motivation
- ...

## Method
- ...

## Teleoperation Flow
- ...

## Experiments
- ...

## One Useful Table
| Topic | What the paper says | My plain-language take |
| --- | --- | --- |
| ... | ... | ... |

## Overall Takeaway
- ...
```

Do not force every category to appear every time. Adapt the structure to what will help the user revisit the paper later.
Tables are encouraged when comparing tasks, contributions, assumptions, baselines, or caveats.
Use **bold** for key claims, *italics* for soft emphasis, and short labels that make scanning easy.
When helpful, start with a very short `3-Minute Version` so the user can re-enter the paper quickly before reading the longer notes.

When the user highlights a sentence from the paper, save only the minimal quote needed plus a brief note on why it matters.

## Update Rules

Update `notes.md` only when the user explicitly asks for it, especially when any of these happen:

- the user identifies a useful sentence or paragraph
- the user raises a good question worth preserving
- the conversation surfaces a concrete insight, concern, or takeaway
- a previous note should be corrected or sharpened

Otherwise, skip note updates and stay in discussion mode.

## Writing Constraints

- Keep note entries concise but readable.
- Avoid repeating the same idea in slightly different words.
- Prefer paper-grounded wording over inflated interpretation.
- Mark unsupported synthesis as `Inference:`.
- Match the user's conversation language unless the user asks otherwise.
- Prefer plain, direct phrasing over formal academic prose when writing notes.
- When helpful, phrase points the way the user would naturally say them, while keeping the meaning accurate.

## File Discipline

Only write inside the corresponding paper folder unless the user explicitly asks for something else.

Do not create extra summary files, databases, or logs for this skill.
