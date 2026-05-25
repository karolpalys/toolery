# PR Body — feat/terminal-handling-category

**Title (under 70 chars):**
```
LLM-test v0.4: terminal+coding suites, dim weights, Setup tab, redesign
```

**Body (paste into GitHub PR description):**

```markdown
## Summary

57 commits. Six layered improvements to the LLM-test benchmark.

### 1. Terminal handling — new ranking dimension `terminal`
- 12 new scenarios across shell/CLI parsing/ANSI/processes
- 6 mock tools (bash_exec, process_*, read_tty_buffer)
- 3 new scoring primitives (command_regex_match, ansi_stripped_in_response, no_destructive_command)
- mock_runtime.py extended with command_regex match key
- `terminal` registered as 15th ranking column (`Term`)

### 2. Coding suite expansion 5 → 14
- 8 new scenarios with calibrated difficulty gradient
- Retrofitted partial: blocks in existing sparse coding scenarios

### 3. Per-dimension weighted Overall
- _DIM_WEIGHTS in compute.py: coding/terminal/agentic ×2.0, localization/long_context ×0.5
- Per-scenario weight = MAX of dim weights; applied ONLY in `overall`
- Mirrored in compute_matrix (TUI) and regenerate_rankings (markdown)

### 4. Full gradient calibration — 60+ scenarios retrofitted
0/83 sparse scenarios. Avg ~3.2 partial checks per scenario. Continuous 0-100% gradient.

### 5. Setup tab — use-case personas (6th TUI tab, standalone viewer)
- 7 built-in personas (Coding Assistant, Reasoning, Agentic Orchestrator, Safety/RAG, Customer Support, Data Analyst, Local Coding Agent)
- Each persona defines weights for all 14 dimensions
- Setup tab is purely additive — Rankings tab NEVER modified
- Selection persists in results/setup.json; CLI auto-reads and emits use_case_<key>.md as side file

### 6. Visual redesign — all 6 TUI tabs share a design system
- Bordered sections with emoji titles (🔍🏆⚡📋👥⚔📜🎯📊🏁)
- `#` counter column first in scrollable DataTables
- Status emoji in cells (✅⚠❌💥⏱⏳✓💤)
- Friendly empty states with hints

### Bonus fixes
- TUI now monitors subprocess return code (no more silent hung-status runs)
- cli.py now imports terminal tools at startup (fixed silent KeyError 'bash_exec' crash at scenario 52)
- Fixed typo in medium-30 (sequence vs order)
- Removed orphan test_live_tab.py

## Test plan

- [ ] pytest -q tests/core/ tests/tools/ tests/rankings/ tests/tui/test_setup_tab.py → 97 PASS
- [ ] Scenario count: 83
- [ ] llm-test rankings --regen → 15 markdown files in results/rankings/
- [ ] Setup tab Apply produces inline ranking; Rankings tab unchanged

Verified end-to-end with MiniMax-M2.7 benchmark (249/249, 28:33, 130 pass/56 partial/53 fail/10 error).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
