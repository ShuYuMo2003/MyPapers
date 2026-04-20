# Output Format

Write the report to `summary.md` in the same directory as the input `layout.md` unless the user specifies another destination.

Use this structure:

```markdown
# <Paper Title>

## 一句话结论

## 核心创新
- 2-4 bullets

## 方法概述
- Focus on the mechanism, not a section-by-section paraphrase

## 实验结论
- Include the strongest evidence only

## 和我关注点的关系
- Explicitly map to VLA, UMI, video generation for embodiment, manipulation/data, or broader transferable ideas

## 启发 / 可迁移点
- What could be borrowed into embodied work

## 局限与疑问
- Include assumptions, missing ablations, real-world gaps, scaling concerns, or robotics gaps

## 值得细看部分
- Optional

## 可做的后续实验
- Optional
```

## Emphasis Rules

- If the paper is directly about VLA or embodied foundation models, spend more words on architecture, data, training recipe, and evaluation gaps.
- If the paper is directly about UMI or closely related manipulation-learning setups, spend more words on embodiment assumptions, data collection protocol, policy interface, transfer recipe, and what is reusable for the user's own manipulation work.
- If the paper is about video generation or world models, spend more words on how it could bridge to policy, planning, or control.
- If the paper is about manipulation or data, spend more words on data source, supervision form, embodiment gap, and transferability.
- If the paper is only indirectly relevant, keep the summary compact and make the transfer path explicit.

## Style Rules

- Write in concise Chinese.
- Prefer concrete statements over vague praise.
- If something is not proven in the paper but seems promising, mark it as `推测`.
- Make it easy for the user to decide whether the paper deserves deeper reading.
