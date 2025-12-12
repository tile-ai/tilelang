#!/usr/bin/env bash
# Usage:
#    # Do work and commit your work.
#
#    # Format files that differ from origin/main.
#    bash format.sh
#
#    # Format all files.
#    bash format.sh --all
#
#
# Ruff (format) + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

if [[ -z "${BASH_VERSION}" ]]; then
    echo "Please run this script using bash." >&2
    exit 1
fi

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

ALL_FILES=''
ONLY_CHANGED=''
FILES=()
if (($# == 0)); then
    # Default: allow dirty workspace; run on changed files (committed + worktree)
    ONLY_CHANGED='true'
else
    while (($# > 0)); do
        case "$1" in
        --files)
            shift
            while (($# > 0)); do
                FILES+=("$1")
                shift
            done
            ;;
        --all)
            ALL_FILES='true'
            shift
            ;;
        *)
            echo "Unknown argument: '$1'" >&2
            exit 1
            ;;
        esac
    done
fi

MERGE_BASE=""
get_merge_base() {
    UPSTREAM_REPO="https://github.com/tile-ai/tilelang"
    if git ls-remote --exit-code "${UPSTREAM_REPO}" main &>/dev/null; then
        # First try to use the upstream repository directly
        MERGE_BASE="$(git fetch "${UPSTREAM_REPO}" main &>/dev/null && git merge-base FETCH_HEAD HEAD)"
    elif git show-ref --verify --quiet refs/remotes/origin/main; then
        # Fall back to origin/main if available
        BASE_BRANCH="origin/main"
        MERGE_BASE="$(git merge-base "${BASE_BRANCH}" HEAD)"
    else
        # Last resort, use local main
        BASE_BRANCH="main"
        MERGE_BASE="$(git merge-base "${BASE_BRANCH}" HEAD)"
    fi
    echo "${MERGE_BASE}"
}

if [[ -n "${ALL_FILES}" ]]; then
    echo "Checking all files..." >&2
elif [[ -n "${ONLY_CHANGED}" ]]; then
    MERGE_BASE="$(get_merge_base)"
    echo "Checking changed files vs merge base (${MERGE_BASE}) and working tree..." >&2
elif [[ "${#FILES[@]}" -gt 0 ]]; then
    echo "Checking specified files: ${FILES[*]}..." >&2
fi

# Some systems set pip's default to --user, which breaks isolated virtualenvs.
export PIP_USER=0

# If pre-commit is not installed, install it.
if ! python3 -m pre_commit --version &>/dev/null; then
    python3 -m pip install pre-commit --user
fi

echo 'tile-lang pre-commit: Check Start'

if [[ -n "${ALL_FILES}" ]]; then
    python3 -m pre_commit run --all-files
elif [[ -n "${ONLY_CHANGED}" ]]; then
    # Collect changed files (committed since merge-base + current worktree)
    CHANGED_FILES="$(git diff --name-only --diff-filter=ACM "${MERGE_BASE}" 2>/dev/null || true)"
    if [[ -n "${CHANGED_FILES}" ]]; then
        echo "Running pre-commit on changed files:"
        echo "${CHANGED_FILES}"
        # Convert newline-separated files to space-separated and run pre-commit once
        CHANGED_FILES_SPACE="$(echo "${CHANGED_FILES}" | tr '\n' ' ')"
        python3 -m pre_commit run --files ${CHANGED_FILES_SPACE}
    else
        echo "No files changed relative to merge base and worktree. Skipping pre-commit."
    fi
elif [[ "${#FILES[@]}" -gt 0 ]]; then
    python3 -m pre_commit run --files "${FILES[@]}"
fi

echo 'tile-lang pre-commit: Done'

echo 'tile-lang clang-tidy: Check Start'
# If clang-tidy is available, run it; otherwise, skip
if [[ -x "$(command -v run-clang-tidy)" ]]; then
    # Check if clang-tidy is available
    if [[ ! -x "$(command -v clang-tidy)" ]]; then
        python3 -m pip install --upgrade --requirements "${ROOT}/requirements-lint.txt" --user
    fi
    # Get clang-tidy version
    CLANG_TIDY_VERSION="$(clang-tidy --version | head -n1 | awk '{print $4}')"
    echo "Using clang-tidy version: ${CLANG_TIDY_VERSION}"

    # Check if build directory exists
    if [[ ! -d "${ROOT}/build" ]]; then
        echo "Build directory not found. Skipping clang-tidy checks."
    else
        # Run clang-tidy on specified files
        clang_tidy_files() {
            run-clang-tidy -j 64 "$@" -p build
        }

        # Run clang-tidy on all C/C++ source files
        clang_tidy_all() {
            run-clang-tidy -j 64 src/*.cc -p build
        }

        # Run clang-tidy on changed C/C++ files relative to main
        clang_tidy_changed() {
            # Get changed C/C++ files
            CHANGED_FILES="$(git diff --name-only --diff-filter=ACM "${MERGE_BASE}" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' 2>/dev/null || true)"

            if [[ -n "${CHANGED_FILES}" ]]; then
                echo "Running clang-tidy on changed files:"
                echo "${CHANGED_FILES}"
                # Convert newline-separated files to space-separated and run clang-tidy once
                CHANGED_FILES_SPACE="$(echo "${CHANGED_FILES}" | tr '\n' ' ')"
                run-clang-tidy -j 64 ${CHANGED_FILES_SPACE} -p build -fix
            else
                echo "No C/C++ files changed. Skipping clang-tidy."
            fi
        }

        if [[ -n "${ALL_FILES}" ]]; then
            # If --all is given, run clang-tidy on all source files
            clang_tidy_all
        elif [[ -n "${ONLY_CHANGED}" ]]; then
            # Otherwise, run clang-tidy only on changed C/C++ files
            clang_tidy_changed
        elif [[ "${#FILES[@]}" -gt 0 ]]; then
            # If --files is given, run clang-tidy only on the provided files
            clang_tidy_files "${FILES[@]}"
        fi
    fi

else
    echo "run-clang-tidy not found. Skipping clang-tidy checks."
    echo "To install clang-tidy tools, you may need to install clang-tidy and run-clang-tidy."
fi
echo 'tile-lang clang-tidy: Done'

# Check if there are any uncommitted changes after all formatting steps.
# If there are, ask the user to review and stage them.
if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

echo 'tile-lang: All checks passed'
