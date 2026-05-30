#!/usr/bin/env bash
#
# Toolery quick-launcher — opens the TUI dashboard from anywhere.
#
#   ./toolery.sh                 # launch the panel
#   ./toolery.sh --help          # any extra args are forwarded to `toolery tui`
#
# Tip: symlink it onto your PATH so you can launch from any directory:
#   sudo ln -s "$(pwd)/toolery.sh" /usr/local/bin/toolery
#   toolery                      # opens the dashboard from anywhere
#
set -euo pipefail

# Resolve the repo directory even when invoked through a symlink.
cd "$(dirname "$(readlink -f "$0")")"

exec uv run toolery tui "$@"
