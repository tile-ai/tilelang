#!/usr/bin/env bash
# Fix Python file format issues using ruff
#
# Usage:
#   fix-python.sh <file.py> [file.pyi ...]

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"

cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Error: provide at least one Python file path." >&2
    exit 2
fi

FILE_ARRAY=()
for file in "$@"; do
    case "$file" in
        *.py|*.pyi)
            if [ -f "$file" ]; then
                FILE_ARRAY+=("$file")
            elif [ -f "$REPO_ROOT/$file" ]; then
                FILE_ARRAY+=("$REPO_ROOT/$file")
            fi
            ;;
    esac
done

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo "Error: no existing Python files were provided." >&2
    exit 2
fi

echo "Fixing ${#FILE_ARRAY[@]} Python file(s)..."

# Run ruff check --fix
echo "Running ruff check --fix..."
ruff check --fix "${FILE_ARRAY[@]}" || true

# Run ruff format
echo "Running ruff format..."
ruff format "${FILE_ARRAY[@]}"

echo "Python files have been fixed."
