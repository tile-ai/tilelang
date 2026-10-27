# Contributing

That would be awesome if you want to contribute something to TileLang!

## Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Report Bugs](#report-bugs)
- [Ask Questions](#ask-questions)
- [Submit Pull Requests](#submit-pull-requests)
  - [Pull request titles](#pull-request-titles)
  - [Pull request descriptions](#pull-request-descriptions)
  - [AI-assisted contributions](#ai-assisted-contributions)
  - [Testing policy](#testing-policy)
- [Coding Style](#coding-style)
- [Setup Development Environment](#setup-development-environment)
- [Install Develop Version](#install-develop-version)
- [Lint Check](#lint-check)
- [Test Locally](#test-locally)
- [Build Wheels](#build-wheels)
- [Documentation](#documentation)

## Report Bugs

If you run into any weird behavior while using TileLang, feel free to open a new issue in this repository! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open must include:

- Code snippet that reproduces the bug with a minimal setup.
- A clear explanation of what the issue is.

## Ask Questions

Please ask questions in issues.

## Submit Pull Requests

All pull requests are super welcomed and greatly appreciated! Issues in need of a solution are marked with a [`♥ help`](https://github.com/tile-ai/tilelang/issues?q=is%3Aissue+is%3Aopen+label%3A%22%E2%99%A5+help%22) label if you're looking for somewhere to start.

### Pull request titles

Use a concise English title with a specific bracketed component tag, such as
`[CI] Skip backend tests for documentation-only PRs`. Describe the concrete
change and keep the title aligned with the final diff.

### Pull request descriptions

Give reviewers enough context to understand the problem, the proposed solution,
and the evidence behind it. These expectations apply to all contributions.
A small change may need only a few sentences; scale the detail to the change
and combine or remove template sections that do not help explain it.

- **Problem:** Describe the original and expected behavior, with a linked issue
  or a small example when useful. For a bug fix, describe the triggering input
  and observed failure on current main or the intended base revision; identify
  the tested revision, or say the failure was not reproduced. For a feature,
  explain the concrete use case and why existing functionality is insufficient.
- **Approach:** Explain the root cause and how the change addresses it. Support
  a bug-fix claim with the changed behavior for the same input, or identify a
  relevant regression test, its result, and what it checks. When adding an API,
  compiler pass, abstraction, or backend-specific behavior, explain why it belongs
  at that layer. For a nontrivial design choice, describe the main alternative
  and the reason for the chosen approach.
- **Scope and tradeoffs:** Identify affected backends or execution modes,
  compatibility changes, and known limitations where relevant. Keep independent
  fixes, features, and cleanups in separate PRs when they can be reviewed on
  their own.

Put evidence alongside the claim it supports. Performance claims need comparable
before/after measurements and relevant hardware or toolchain settings; include
end-to-end costs when moving work to the host or adding allocations,
synchronization, or per-call preparation. Include commands and environment
details when they help reproduce or interpret a result, and clearly identify
important behavior that remains unverified.

A separate validation section or a routine list of passing lint/test commands
is not required. Documentation changes and simple cleanups usually do not need
to repeat CI results in the description.

Keep the description aligned with the final diff when the scope changes during
review. The [PR template](.github/PULL_REQUEST_TEMPLATE.md) provides short prompts
for this information. Regression coverage follows the [testing policy](#testing-policy)
below; providing evidence does not require adding a new test file.

### AI-assisted contributions

AI-assisted contributions are welcome. The submitting contributor is responsible
for understanding the change, reviewing the complete diff, verifying the PR
description and reported validation results, and addressing review feedback.
Clearly distinguish verified behavior from assumptions and untested claims.

Before opening a PR or changing its title, an agent should show the exact
proposed title and ask the contributor to confirm or replace it. A title supplied
by the contributor, or explicit permission to choose the title without asking,
satisfies this step. Do not repeat confirmation when the title is unchanged.
The [submission skill](.agents/skills/tilelang-submit-pr/SKILL.md#confirm-the-pr-title)
describes this interaction.

You may optionally include `AI assistance: <tool name>` in the PR description.
Disclosing the tool is not a requirement for review or merging. The repository's
submission skill offers this choice when preparing a new PR and adds the field
only when the contributor opts in.

### Testing policy

- Provide appropriate regression coverage; a bug fix does not automatically require
  a new standalone test.
- Prefer extending existing tests with the smallest necessary case.
- Add independent tests or parameter combinations only when they cover distinct
  behavior not already covered.
- Explain the additional coverage when introducing new tests.
- Update documentation when applicable.

For test placement, case selection, and examples, follow the
[Testing Style Guide](docs/developer_guide/testing_style.md).

## Coding Style

Run the repository formatter on the files touched by your pull request:

```bash
bash format.sh --files <changed-file>...
```

Python code is checked with Ruff through pre-commit. C and C++ code is checked
with clang-format, and TileLang-owned C++ APIs should follow the
[C++ Style Guide](docs/developer_guide/cpp_style.md) for naming, TVM object
usage, header boundaries, and incremental migration.

For a focused C++ API naming or header-boundary cleanup, run the advisory audit
locally:

```bash
python3 maint/scripts/audit_cpp_api_style.py
```

The audit is a review aid, not a blanket rename command. Check each finding for
FFI visibility, generated code, backend shims, and external API constraints
before changing it.

## Setup Development Environment

Before contributing to TileLang, please follow the instructions below to setup.

1. Fork TileLang ([fork](https://github.com/tile-ai/tilelang/fork)) on GitHub and clone the repository.

    ```bash
    git clone --recurse-submodules git@github.com:<your username>/tilelang.git  # use the SSH protocol
    cd tilelang

    git remote add upstream git@github.com:tile-ai/tilelang.git
    ```

2. Setup a development environment:

    ```bash
    uv venv --seed .venv  # use `python3 -m venv .venv` if you don't have `uv`

    source .venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel "build[uv]"
    uv pip install --requirements requirements-dev.txt
    ```

3. Setup the [`pre-commit`](https://pre-commit.com) hooks:

    ```bash
    pre-commit install --install-hooks
    ```

Then you are ready to rock. Thanks for contributing to TileLang!

## Install Develop Version

To install TileLang in an "editable" mode, run:

```bash
python3 -m pip install --no-build-isolation --verbose --editable .
```

in the main directory. This installation is removable by:

```bash
python3 -m pip uninstall tilelang
```

We also recommend installing TileLang in a more manual way for better control over the build process, by compiling the C++ extensions first and set the `PYTHONPATH`. See [Working from Source via `PYTHONPATH`](https://tilelang.com/get_started/Installation.html#working-from-source-via-pythonpath) for detailed instructions.

## Lint Check

To check the linting, run:

```bash
pre-commit run --all-files
```

## Test Locally

To run the tests, start by building the project as described in the [Setup Development Environment](#setup-development-environment) section.

Then you can rerun the tests with:

```bash
python3 -m pytest testing
```

## Build Wheels

_TBA_

## Documentation

_TBA_
