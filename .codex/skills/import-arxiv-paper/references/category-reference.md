# Category Reference

Use this file for the human-readable meaning of categories and the maintenance rules for adding new ones.

## Current Default

This project currently starts with one registered category:

- `egocentric`

The machine-readable list of active category slugs lives in `categories.txt`.

## Rules

- Use the category provided by the user. Do not infer a category if the user did not supply one.
- Keep category names folder-safe and stable.
- Normalize new categories to lowercase hyphen-case before using them as folder names.
- If the user provides a category that is not yet in `categories.txt`, add it to the registry and create the matching folder under `paper/`.
- Treat `categories.txt` as the source of truth for what categories currently exist in this project.

## Category Descriptions

### egocentric

Use for first-person or wearable-view papers centered on egocentric perception, egocentric video understanding, hand-object interaction from first-person views, or human-to-robot transfer signals grounded in egocentric observation.

### umi

Use for papers the user explicitly groups under the UMI research track, especially manipulation, demonstration, imitation, teleoperation, or data/interface work that feeds into that line of work.

## Adding New Categories

When a new category appears:

1. Normalize the user-provided name into a folder-safe slug.
2. Append the slug to `categories.txt` if it is not already present.
3. Create `paper/<category>/`.
4. Add a short description for the new category here when the meaning is stable and worth documenting.
