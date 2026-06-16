# Rankings tab: frozen meta columns + stable horizontal scroll

Date: 2026-06-16
Status: approved (design)
Scope: `toolery/tui/rankings_tab.py` only

## Problem

The Rankings tab (`RankingsTab`, a Textual `DataTable` with ~25 columns) has two
usability defects when the table is wider than the viewport:

1. **No frozen columns.** Scrolling right to read perf/stability/cluster columns
   pushes the `#` (rank) and `Model` columns off-screen, so the user loses track
   of which row is which.

2. **Horizontal scroll randomly jumps back to the left.** The tab polls the DB
   every 5 s (`set_interval(5.0, self.reload)`). There is already an anti-jump
   mechanism (`_last_render_sig` render-skip + `scroll_x/scroll_y` restore in
   `_populate_rows`), but it has a hole, so the scroll resets intermittently.

### Root cause of the jump

- `reload()` calls `tbl.clear(columns=True)` (line ~491) on the data-changed
  path. That **resets scroll to (0,0)** and rebuilds the columns.
- `_populate_rows()` only captures `prev_x = tbl.scroll_x` afterwards, by which
  point scroll is already 0 — so `if prev_x or prev_y` is false and the restore
  never fires.
- The render signature rounds time-decayed scores to 3 dp. The continuous decay
  drift *occasionally* crosses a rounding boundary, flipping the signature and
  forcing the full-rebuild path — hence "every once in a while", not every 5 s.

Columns are **structurally static** — only row data changes between polls — so
`clear(columns=True)` per refresh is both wasteful and the thing that destroys
scroll hardest.

## Design

Two changes, both confined to `toolery/tui/rankings_tab.py`.

### 1. Freeze `#` and `Model`

Set `fixed_columns=2` on the `DataTable` in `compose()`. Textual 8.2.7 supports
this natively; the first two columns (`#`, `Model`) stay pinned while everything
from `Adapter` rightward scrolls. `Adapter` stays scrollable (explicit user
decision).

### 2. Build columns once; refresh rows without touching columns

Separate column setup from row population so the data-refresh path never clears
columns and never loses scroll:

- Extract column + `_sort_keys` registration into `_ensure_columns()`, invoked
  only when the table has no columns yet (first load, or recovering from an
  empty/no-DB state).
- On the data-changed path, `reload()` calls `_ensure_columns()` (no-op if
  columns already exist) then `_populate_rows()`. `_populate_rows()` already
  captures `scroll_x/scroll_y` before its own row-only `tbl.clear()` and
  restores them via `call_after_refresh`, so scroll survives every poll.
- The empty / no-DB branches keep `tbl.clear(columns=True)` and reset
  `_last_render_sig = None`; that is the only place columns are torn down, and
  scroll there is irrelevant (no rows).

`fixed_columns` is compatible with the scroll restore: `scroll_x` addresses the
scrollable region only, so restoring it after a row refresh behaves correctly
with the two pinned columns.

## Out of scope

- The 5 s poll interval itself (kept — it surfaces completed runs live).
- The signature/rounding logic (kept — the column-rebuild fix makes the
  occasional signature flip harmless instead of disruptive).
- Any other tab.

## Testing

- `tests/` already exercises the tab; add/extend a test asserting `fixed_columns
  == 2` and that a no-op `reload()` (unchanged signature) preserves a non-zero
  `scroll_x`. Headless Textual via `App.run_test()` per existing TUI tests.
- Manual: launch TUI, scroll Rankings right, confirm `#`/`Model` stay pinned and
  that scroll position holds across at least two 5 s poll cycles.
