---
name: embodied-paper-summary
description: Summarize a paper from a provided `layout.md` or similar layout-extracted article file, then write a concise Chinese report tailored to the user's embodied AI research interests. Use when Codex needs to read a paper layout, or is given a local PDF that should first be converted into a temporary layout file, then explain the method and conclusions, highlight innovations, and selectively expand details related to VLA, embodied foundation models, video generation for policy/planning, manipulation, data sources, simulation, cross-embodiment transfer, or other ideas that may transfer into embodied intelligence.
---

# Embodied Paper Summary

## Workflow

Follow this workflow every time:

1. Read `references/research-focus.md` to align the summary with the user's interests.
2. Read `references/output-format.md` to match the expected structure and level of detail.
3. If the user provided a PDF instead of `layout.md`, first use `../pdf-layout-reader` to generate a temporary `layout.md`.
4. Read the provided `layout.md` or generated layout file before writing any interpretation.
5. Infer the paper's main thesis, method, evidence, and limitations from the text. Label clear inferences as `推测`.
6. Write or overwrite `summary.md` in the same folder as the input layout file unless the user explicitly requests another output path.

## Reading Priorities

Use this reading order:

1. title
2. abstract
3. introduction
4. method
5. experiments
6. conclusion

Do not aim for a uniform section-by-section rewrite. Compress background and standard setup. Spend detail budget on what is most decision-relevant for the user.

## Detail Allocation

Expand details when the paper is relevant to one or more of these:

- VLA architecture, training, scaling, evaluation, deployment, or comparisons across labs or companies
- using video generation or world models for control, planning, data synthesis, representation learning, or action generation
- manipulation-specific techniques, especially data collection, human data, simulation data, pi0-like directions, or cross-embodiment transfer
- ideas outside embodied AI that may transfer into embodied model training or system design

Compress details when they are mostly:

- standard dataset descriptions with little method novelty
- conventional implementation details that do not affect the key conclusion
- benchmark tables that do not change the takeaway for embodied AI

## Output Rules

Write in Chinese. Keep the report concise and high-signal. Prefer sharp judgments over exhaustive restatement.

Always include:

- `一句话结论`
- `核心创新`
- `方法概述`
- `实验结论`
- `和我关注点的关系`
- `启发 / 可迁移点`
- `局限与疑问`

If the paper is especially relevant, also include:

- `值得细看部分`
- `可做的后续实验`

If the paper is weakly relevant, say so directly and keep the summary short.

## Interpretation Rules

- Distinguish what the paper explicitly claims from your own synthesis.
- When discussing relevance to embodied AI, focus on mechanism transfer, training recipe transfer, data pipeline transfer, or evaluation transfer.
- When the paper involves video generation, explicitly ask whether it can help policy learning, planning, synthetic data generation, reward shaping, or latent imagination for control.
- When the paper involves robotics, explicitly identify whether it improves perception, planning, control, data efficiency, generalization, or cross-platform transfer.
- When the paper discusses a company, lab, or system trend, note it briefly only if it helps the user understand the broader embodied model landscape.

## Resources

- `references/research-focus.md`: the user's standing research interests and what to prioritize.
- `references/output-format.md`: the required summary format and emphasis rules.
- `../pdf-layout-reader`: use this first when the user provides a PDF instead of a precomputed layout file.
