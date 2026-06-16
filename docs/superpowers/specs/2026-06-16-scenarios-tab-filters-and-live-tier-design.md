# Scenarios tab: live difficulty/category metadata + difficulty & category filters

Date: 2026-06-16
Status: approved (design)
Scope: `toolery/tui/scenarios_tab.py` only (display-only; no data migration)

## Problem

The Scenarios tab browses scenarios via a single `Select` picker labelled
`{tier} · {display_name}`. Two defects:

1. **Stale-looking difficulty.** `load_all_scenarios` returns files in path order
   (`sorted(rglob("*.yaml"))`), so the picker is grouped by *folder*
   (`scenarios/easy/`, `/hard/`, …) — the **old** hand-assigned difficulty —
   while labels show the **new** empirical `tier`. 85/143 scenarios were
   re-tiered, so the list shows jumbled difficulties (`easy · …`, `very_hard ·
   03-…`, `medium · 08-…`) and reads as if the difficulty tag is outdated. The
   Task panel also shows no difficulty/category/tags metadata at all.

2. **No way to narrow the list.** Finding a specific scenario means scrolling the
   whole 143-entry picker.

`id` is a stable key joining to historical runs in `runs.db`, so IDs and files
are NOT renamed (confirmed with user). This is a display/UX fix.

## Design

All changes in `toolery/tui/scenarios_tab.py`.

### 1. Filter row (two `Select`s above the picker)

- `#sc-filter-difficulty`: `All difficulties / Easy / Medium / Hard / Very hard`
  (values `all / easy / medium / hard / very_hard`), default `all`.
- `#sc-filter-category`: `All categories` + one entry per `Category` enum value,
  English labels via `_humanize(value)` = `value.replace("_", " ").capitalize()`
  (e.g. `tool_selection` → "Tool selection", `adversarial_robustness` →
  "Adversarial robustness"). Default `all`.

Both default to `all` (not blank). English labels throughout.

### 2. Picker rebuilt on filter change — `_apply_filters()`

- Filter the full scenario list by the two selects (`all` = no constraint).
- **Sort by current `tier`** (easy→medium→hard→very_hard) then `display_name`,
  so the list is coherent by *current* difficulty — fixes the folder-order jumble.
- Rebuild `#sc-pick` options. If the previously-selected scenario is still in the
  filtered set, keep it selected; otherwise select the first entry and render it.
- If the filtered set is empty: clear the picker, show "No scenarios match the
  current filters." in the Task panel, clear the results table and stats.

### 3. Task panel metadata — `_render_task()`

Prepend a metadata block using the **live** `tier` (single source of truth for
difficulty) + category + real tags, rendered as styled chips:

```
Difficulty: Very hard      Category: Tool selection
Tags: tool_selection · single_turn · distractor · tool_agnostic
```

(No tags → "Tags: —".) The historical id prefix never appears as difficulty.

### 4. Picker label

Kept concise: `{tier} · {display_name}` using live `s.tier`. Category is
filterable and shown in the Task panel, so it is not added to the label.

## Event routing

`on_select_changed` dispatches by `event.select.id`: the two filter selects call
`_apply_filters()`; `#sc-pick` calls `_render_scenario()`.

## Out of scope

- Renaming scenario IDs / files (would break runs.db joins).
- Free-text search (declined — two dropdowns chosen).
- Any other tab.

## Testing

`App.run_test()` headless (per existing `tests/tui/`):

- Difficulty select has 5 options (incl. All); category select has
  `len(Category) + 1`.
- Difficulty=`very_hard` narrows the picker to only scenarios whose current
  `tier == very_hard`; count matches the loaded set.
- Combined difficulty+category filter intersects correctly.
- Task panel for a re-tiered scenario (e.g. `easy-03-refuse-trivial-math`) shows
  "Very hard" (live tier), its category, and its real tags — not "Easy".
- Picker options are ordered by current tier (easy entries precede very_hard).
