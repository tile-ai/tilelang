#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')
CODESPELL_VERSION=$(codespell --version)

# # params: tool name, tool version, required version
tool_version_check() {
    if [[ $2 != $3 ]]; then
        echo "Wrong $1 version installed: $3 is required, not $2."
        exit 1
    fi
}

tool_version_check "yapf" $YAPF_VERSION "$(grep yapf requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "ruff" $RUFF_VERSION "$(grep "ruff==" requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "codespell" "$CODESPELL_VERSION" "$(grep codespell requirements-dev.txt | cut -d'=' -f3)"

echo 'tile-lang yapf: Check Start'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
    '--exclude' '3rdparty/**'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format files that differ from main branch. Ignores dirs that are not slated
# for autoformat yet.
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause yapf to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi

}

# Format all files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

## This flag formats individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   format "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
   format_all
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'tile-lang yapf: Done'

echo 'tile-lang codespell: Check Start'
# check spelling of specified files
spell_check() {
    codespell "$@"
}

spell_check_all(){
  codespell --toml pyproject.toml
}

# Spelling  check of files that differ from main branch.
spell_check_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             codespell
    fi
}

# Run Codespell
## This flag runs spell check of individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   spell_check "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   spell_check_all
else
   # Check spelling only of the files that changed in last commit.
   spell_check_changed
fi
echo 'tile-lang codespell: Done'

echo 'tile-lang ruff: Check Start'
# Lint specified files
lint() {
    ruff check "$@"
}

# Lint files that differ from main branch. Ignores dirs that are not slated
# for autolint yet.
lint_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi

    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff check
    fi

}

# Run Ruff
### This flag lints individual files. --files *must* be the first command line
### arg to use this option.
if [[ "$1" == '--files' ]]; then
   lint "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   lint python testing
else
   # Format only the files that changed in last commit.
   lint_changed
fi

echo 'tile-lang ruff: Done'

echo 'tile-lang clang-format: Check Start'
# If clang-format is available, run it; otherwise, skip
if command -v clang-format &>/dev/null; then
    CLANG_FORMAT_VERSION=$(clang-format --version | awk '{print $3}')
    tool_version_check "clang-format" "$CLANG_FORMAT_VERSION" "$(grep clang-format requirements-dev.txt | cut -d'=' -f3)"

    CLANG_FORMAT_FLAGS=("-i")

    # Apply clang-format to specified files
    clang_format() {
        clang-format "${CLANG_FORMAT_FLAGS[@]}" "$@"
    }

    # Format all C/C++ files in the repo, excluding specified directories
    clang_format_all() {
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) \
            -not -path "./3rdparty/*" \
            -not -path "./build/*" \
            -exec clang-format -i {} +
    }

    # Format changed C/C++ files relative to main
    clang_format_changed() {
        if git show-ref --verify --quiet refs/remotes/origin/main; then
            BASE_BRANCH="origin/main"
        else
            BASE_BRANCH="main"
        fi

        MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"

        if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' &>/dev/null; then
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' | xargs clang-format -i
        fi
    }

    if [[ "$1" == '--files' ]]; then
       # If --files is given, format only the provided files
       clang_format "${@:2}"
    elif [[ "$1" == '--all' ]]; then
       # If --all is given, format all eligible C/C++ files
       clang_format_all
    else
       # Otherwise, format only changed C/C++ files
       clang_format_changed
    fi
else
    echo "clang-format not found. Skipping C/C++ formatting."
fi
echo 'tile-lang clang-format: Done'

# Check if there are any uncommitted changes after all formatting steps.
# If there are, ask the user to review and stage them.
if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

echo 'tile-lang: All checks passed'
