#!/usr/bin/env python3
"""Check only explicit files and report tool failures separately from issues."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


EXTENSIONS = {
    'python': {'.py', '.pyi'},
    'cpp': {'.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hh', '.icc'},
}
TOOL = {'python': 'ruff', 'cpp': 'clang-format'}


def _decode(value: bytes) -> str:
    return value.decode('utf-8', errors='replace').strip()


def _failure(results: dict, label: str, check: str, command: list[str], error: str, exit_code: int | None = None) -> None:
    item = {'file_path': label, 'tool': TOOL[results['language']], 'check': check, 'command': shlex.join(command), 'error': error}
    if exit_code is not None:
        item['exit_code'] = exit_code
    results['execution_errors'].append(item)


def _run(command: list[str], repo_root: Path) -> tuple[subprocess.CompletedProcess[bytes] | None, str | None]:
    try:
        return subprocess.run(command, cwd=repo_root, capture_output=True, timeout=120, check=False), None
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error)


def _label(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _inputs(raw_files: list[str], language: str, repo_root: Path, results: dict) -> list[tuple[Path, str]]:
    selected = []
    seen = set()
    for raw in raw_files:
        candidate = Path(raw)
        if candidate.suffix.lower() not in EXTENSIONS[language]:
            results['skipped_files'].append({'file_path': raw, 'reason': 'unsupported extension'})
            continue
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            path = candidate.resolve(strict=True)
        except OSError as error:
            results['skipped_files'].append({'file_path': raw, 'reason': error.strerror or 'cannot resolve file'})
            continue
        if not path.is_file():
            results['skipped_files'].append({'file_path': raw, 'reason': 'not a regular file'})
            continue
        if path in seen:
            continue
        seen.add(path)
        selected.append((path, _label(path, repo_root)))
    return selected


def _check_python(files: list[tuple[Path, str]], repo_root: Path, results: dict) -> None:
    if not shutil.which('ruff'):
        for _, label in files:
            _failure(results, label, 'lint+format', ['ruff'], 'ruff executable not found')
        return

    lint_checked = set()
    format_checked = set()
    for path, label in files:
        command = ['ruff', 'check', '--output-format=json', label]
        run, run_error = _run(command, repo_root)
        if run_error:
            _failure(results, label, 'lint', command, run_error)
        elif run.returncode not in (0, 1):
            _failure(results, label, 'lint', command, _decode(run.stderr) or 'ruff check failed', run.returncode)
        else:
            try:
                issues = json.loads(run.stdout.decode('utf-8'))
                if not isinstance(issues, list) or (run.returncode == 1 and not issues):
                    raise ValueError('Ruff returned no JSON diagnostics despite exit code 1')
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                _failure(results, label, 'lint', command, f'invalid Ruff JSON: {error}; stderr: {_decode(run.stderr)}', run.returncode)
            else:
                lint_checked.add(label)
                results['issues'].extend(issues)

        command = ['ruff', 'format', '--diff', label]
        run, run_error = _run(command, repo_root)
        if run_error:
            _failure(results, label, 'format', command, run_error)
        elif run.returncode == 0 and not run.stdout.strip():
            format_checked.add(label)
        elif run.returncode == 1 and run.stdout.startswith(b'--- '):
            format_checked.add(label)
            results['format_issues'].append(label)
        else:
            _failure(results, label, 'format', command, _decode(run.stderr) or 'ruff format returned no valid diff', run.returncode)

    results['lint_files_checked'] = sorted(lint_checked)
    results['format_files_checked'] = sorted(format_checked)
    results['checked_files'] = sorted(lint_checked & format_checked)
    results['files_checked'] = len(results['checked_files'])


def _check_cpp(files: list[tuple[Path, str]], repo_root: Path, results: dict) -> None:
    if not shutil.which('clang-format'):
        for _, label in files:
            _failure(results, label, 'format', ['clang-format'], 'clang-format executable not found')
        return

    for path, label in files:
        command = ['clang-format', '--style=file', label]
        try:
            original = path.read_bytes()
        except OSError as error:
            _failure(results, label, 'format', command, str(error))
            continue
        run, run_error = _run(command, repo_root)
        if run_error:
            _failure(results, label, 'format', command, run_error)
        elif run.returncode != 0 or (original and not run.stdout):
            _failure(results, label, 'format', command, _decode(run.stderr) or 'clang-format produced no valid output', run.returncode)
        else:
            results['checked_files'].append(label)
            if run.stdout != original:
                results['issues'].append({'file': label, 'message': 'File needs formatting'})
    results['files_checked'] = len(results['checked_files'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=tuple(EXTENSIONS), required=True)
    parser.add_argument('files', nargs='*')
    args = parser.parse_args()
    results = {
        'language': args.language,
        'issues': [],
        'format_issues': [],
        'files_checked': 0,
        'checked_files': [],
        'execution_errors': [],
        'skipped_files': [],
    }
    git_root, git_error = _run(['git', 'rev-parse', '--show-toplevel'], Path.cwd())
    if git_error or git_root.returncode != 0:
        _failure(
            results, '<repository>', 'setup', ['git', 'rev-parse', '--show-toplevel'], git_error or _decode(git_root.stderr) or 'not a Git repository'
        )
    else:
        repo_root = Path(_decode(git_root.stdout)).resolve()
        files = _inputs(args.files, args.language, repo_root, results)
        if args.language == 'python':
            _check_python(files, repo_root, results)
        else:
            _check_cpp(files, repo_root, results)
    json.dump(results, sys.stdout, ensure_ascii=False)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
