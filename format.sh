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
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# Ensure pip installs into the active environment even if pip.conf forces --user installs.
export PIP_USER=0

if [[ -z "${BASH_VERSION}" ]]; then
    echo "Please run this script using bash." >&2
    exit 1
fi

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

MODE="changed"
MERGE_BASE=""
declare -a FILE_ARGUMENTS=()
declare -a WORKTREE_FILES=()
declare -a WORKTREE_CLANG_FILES=()
declare -A WORKTREE_SEEN=()

add_worktree_file() {
    local path="$1"
    [[ -z "${path}" ]] && return
    if [[ -n "${WORKTREE_SEEN[${path}]}" ]]; then
        return
    fi
    if [[ ! -e "${path}" && ! -L "${path}" ]]; then
        return
    fi
    WORKTREE_SEEN["${path}"]=1
    WORKTREE_FILES+=("${path}")
    case "${path}" in
    *.c|*.cc|*.cpp|*.h|*.hpp)
        WORKTREE_CLANG_FILES+=("${path}")
        ;;
    esac
}

collect_worktree_files() {
    local path
    while IFS= read -r path; do
        add_worktree_file "${path}"
    done < <(git diff --name-only || true)

    while IFS= read -r path; do
        add_worktree_file "${path}"
    done < <(git diff --name-only --cached || true)

    while IFS= read -r path; do
        add_worktree_file "${path}"
    done < <(git ls-files -o --exclude-standard || true)

    if ((${#WORKTREE_FILES[@]} > 0)); then
        echo 'Detected uncommitted changes. Running formatters on working tree files as well.' >&2
    fi
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
        --all)
            MODE="all"
            shift
            ;;
        --files)
            MODE="files"
            shift
            if (($# == 0)); then
                echo "--files requires at least one path." >&2
                exit 1
            fi
            while (($# > 0)); do
                FILE_ARGUMENTS+=("$1")
                shift
            done
            return
            ;;
        *)
            echo "Unknown argument: '$1'" >&2
            exit 1
            ;;
        esac
    done
}

get_merge_base() {
    local upstream_repo="https://github.com/tile-ai/tilelang"
    if git ls-remote --exit-code "${upstream_repo}" main &>/dev/null; then
        MERGE_BASE="$(git fetch "${upstream_repo}" main &>/dev/null && git merge-base FETCH_HEAD HEAD)"
    elif git show-ref --verify --quiet refs/remotes/origin/main; then
        local base_branch="origin/main"
        MERGE_BASE="$(git merge-base "${base_branch}" HEAD)"
    else
        local base_branch="main"
        MERGE_BASE="$(git merge-base "${base_branch}" HEAD)"
    fi
}

ensure_pre_commit() {
    if ! python3 -m pre_commit --version &>/dev/null; then
        if ! python3 -m pip install --no-user pre-commit; then
            python3 -m pip install --user pre-commit
        fi
    fi

    if [[ ! -f "${ROOT}/.git/hooks/pre-commit" ]]; then
        echo "Installing and initializing pre-commit hooks..."
        if ! python3 -m pre_commit install --install-hooks; then
            python3 -m pre_commit install --install-hooks --user
        fi
    fi
}

run_pre_commit() {
    echo 'tile-lang pre-commit: Check Start'
    case "${MODE}" in
    all)
        echo "Checking all files..." >&2
        python3 -m pre_commit run --all-files
        ;;
    files)
        echo "Checking specified files: ${FILE_ARGUMENTS[*]}..." >&2
        python3 -m pre_commit run --files "${FILE_ARGUMENTS[@]}"
        ;;
    changed)
        get_merge_base
        echo "Checking changed files compared to merge base (${MERGE_BASE})..." >&2
        python3 -m pre_commit run --from-ref "${MERGE_BASE}" --to-ref HEAD
        if ((${#WORKTREE_FILES[@]} > 0)); then
            python3 -m pre_commit run --files "${WORKTREE_FILES[@]}"
        fi
        ;;
    esac
    echo 'tile-lang pre-commit: Done'
}

run_clang_tidy() {
    echo 'tile-lang clang-tidy: Check Start'

    if ! command -v run-clang-tidy &>/dev/null; then
        echo "run-clang-tidy not found. Skipping clang-tidy checks."
        echo "To install clang-tidy tools, you may need to install clang-tidy and run-clang-tidy."
        echo 'tile-lang clang-tidy: Done'
        return
    fi

    if [[ ! -x "$(command -v clang-tidy)" ]]; then
        python3 -m pip install --no-user --upgrade --requirements "${ROOT}/requirements-lint.txt" || true
    fi

    local clang_tidy_version
    if ! clang_tidy_version="$(clang-tidy --version 2>/dev/null | head -n1 | awk '{print $4}')"; then
        echo "clang-tidy found but could not be executed. Skipping clang-tidy checks."
        echo "Ensure clang-tidy and its dependencies are installed correctly to enable these checks."
        echo 'tile-lang clang-tidy: Done'
        return
    fi
    echo "Using clang-tidy version: ${clang_tidy_version}"

    if [[ ! -d "${ROOT}/build" ]]; then
        echo "Build directory not found. Skipping clang-tidy checks."
        echo 'tile-lang clang-tidy: Done'
        return
    fi

    if [[ "${MODE}" == "all" ]]; then
        if compgen -G 'src/*.cc' >/dev/null; then
            if ! run-clang-tidy -j 64 src/*.cc -p build; then
                echo "run-clang-tidy failed. Skipping clang-tidy checks."
            fi
        else
            echo "No C/C++ files changed. Skipping clang-tidy."
        fi
        echo 'tile-lang clang-tidy: Done'
        return
    fi

    local -a candidates=()
    local path
    declare -A seen=()

    if [[ "${MODE}" == "files" ]]; then
        for path in "${FILE_ARGUMENTS[@]}"; do
            case "${path}" in
            *.c|*.cc|*.cpp|*.h|*.hpp) ;;
            *) continue ;;
            esac
            if [[ ! -e "${path}" && ! -L "${path}" ]]; then
                continue
            fi
            if [[ -z "${seen[${path}]}" ]]; then
                seen["${path}"]=1
                candidates+=("${path}")
            fi
        done
    else
        while IFS= read -r path; do
            case "${path}" in
            *.c|*.cc|*.cpp|*.h|*.hpp) ;;
            *) continue ;;
            esac
            if [[ ! -e "${path}" && ! -L "${path}" ]]; then
                continue
            fi
            if [[ -z "${seen[${path}]}" ]]; then
                seen["${path}"]=1
                candidates+=("${path}")
            fi
        done < <(git diff --name-only --diff-filter=ACM "${MERGE_BASE}" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' 2>/dev/null || true)

        for path in "${WORKTREE_CLANG_FILES[@]}"; do
            if [[ -z "${seen[${path}]}" ]]; then
                seen["${path}"]=1
                candidates+=("${path}")
            fi
        done
    fi

    if ((${#candidates[@]} == 0)); then
        echo "No C/C++ files changed. Skipping clang-tidy."
        echo 'tile-lang clang-tidy: Done'
        return
    fi

    echo "Running clang-tidy on changed files:"
    printf '%s\n' "${candidates[@]}"

    local -a tidy_cmd=(run-clang-tidy -j 64)
    tidy_cmd+=("${candidates[@]}")
    tidy_cmd+=(-p build)
    if [[ "${MODE}" == "changed" ]]; then
        tidy_cmd+=(-fix)
    fi

    if ! "${tidy_cmd[@]}"; then
        echo "run-clang-tidy failed. Skipping clang-tidy checks."
    fi

    echo 'tile-lang clang-tidy: Done'
}

main() {
    parse_args "$@"

    case "${MODE}" in
    changed)
        collect_worktree_files
        ;;
    files)
        if ((${#FILE_ARGUMENTS[@]} == 0)); then
            echo "No files provided for --files option." >&2
            exit 1
        fi
        ;;
    esac

    ensure_pre_commit
    run_pre_commit
    run_clang_tidy

    if ! git diff --quiet &>/dev/null; then
        echo 'Reformatted files. Please review and stage the changes.'
        echo 'Changes not staged for commit:'
        echo
        git --no-pager diff --name-only
        exit 1
    fi

    echo 'tile-lang: All checks passed'
}

main "$@"
