#!/usr/bin/env bash
# Read-only Ruff lint and format checks for explicitly supplied Python files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/workflow.check_format.py" --language python "$@"
