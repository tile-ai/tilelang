---
name: tilelang-dev-guide
description: Apply TileLang C++ development conventions and plan focused regression coverage. Use for C++ edits, style reviews or documentation, and before choosing tests for bug fixes, behavior changes, or test extensions. Not for merely running existing builds or tests or publishing PRs.
---

# TileLang Development Guide

Apply development conventions while writing changes, rather than waiting for
PR preparation. Use only the guidance relevant to the task.

## Select the Relevant Guidance

- For C++ edits, style reviews, or incremental cleanup, read the
  [C++ conventions](references/cpp-style.md). When changing documented
  conventions, also consult the public
  [C++ Style Guide](../../../docs/developer_guide/cpp_style.md).
- For regression coverage, new tests, or additional parameter cases, read the
  [Testing Style Guide](../../../docs/developer_guide/testing_style.md) before
  deciding whether to add or expand tests.

Read both for C++ behavior changes. Test-only tasks do not need the C++ guidance
unless they change C++ code.

## Validate Within the Requested Scope

When implementing a behavior change or modifying tests, run focused tests and
the relevant existing coverage for the affected behavior. Report what actually
ran and any unavailable backend or toolchain. If only a plan was requested, stop
at the proposed coverage.

Planning, writing, or running tests does not authorize committing, pushing, or
changing a PR. Publication readiness remains a separate workflow.
