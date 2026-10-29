---
name: tilelang-submit-pr
description: Prepare, update, and submit TileLang pull requests with repository-specific CI readiness checks. Use when asked to commit, push, publish, open, or make an existing TileLang PR ready; covers scoped diffs, backend-aware test gating, build freshness, formatting, validation, and post-push verification. Do not use for ordinary local builds or test runs without PR intent.
---

# Submit a TileLang Pull Request

## Objective

Make the intended patch reviewable and CI-ready before it is pushed. A request
to inspect or test changes does not authorize committing, pushing, opening a PR,
or changing an existing PR. When updating an existing PR, push its current head
branch instead of opening a duplicate PR.

## Prepare the PR Description

Read [Pull request descriptions](../../../CONTRIBUTING.md#pull-request-descriptions)
for the shared contribution requirements, and use the
[PR template](../../../.github/PULL_REQUEST_TEMPLATE.md) where helpful. Keep the
detail proportional to the change rather than filling every heading mechanically.
Base the description on the complete final diff and actual validation records;
revise it when the implementation or scope changes. The contributor remains
responsible for the description and any claims it makes.

Put supporting evidence in the problem or approach explanation: describe the
before/after behavior or a relevant regression test's result and what it checks.
Use comparable measurements for performance claims. Identify important unverified
behavior. Do not automatically append a `Validation` section or a routine list
of passing formatting/test commands. Include commands and environment details
when they help reproduce or interpret a claim. Keep performing the validation
required below; the PR description should explain what its evidence establishes.

## Optional AI Assistance Disclosure

Follow the [AI-assisted contribution policy](../../../CONTRIBUTING.md#ai-assisted-contributions).
When preparing a new PR description, offer the contributor a choice about
including the agent's name. Ask early enough to answer while patch inspection
and validation continue, or combine this choice with the final title confirmation.
Reuse an explicit preference for this PR or a standing preference from the
conversation; do not ask again on each invocation or push.
For an existing PR, preserve its disclosure unless the user requests a change.
Commit-only work does not need this question.

Use the host's structured question tool when available and permitted in the
current mode, such as `request_user_input_async`, `request_user_input`, or its
equivalent. The client renders the choices; a skill cannot guarantee a native
dialog on every host. If structured questions are unavailable, ask once in
plain text. In a non-interactive run, use an existing explicit preference or
omit the field.

Ask in the user's language, showing the exact proposed text. For example, when
Codex is the known tool used for this contribution:

> Would you like to include `AI assistance: Codex` in the PR description?
>
> - Omit agent information (default).
> - Include `AI assistance: Codex`.

Allow free text for another tool name or multiple tools. Use only tool names
known from the session or supplied by the contributor; do not guess a model
version or imply that this agent authored the entire PR.

Include the field only after an explicit opt-in. A preselected option, timeout,
or unanswered question is not opt-in. Give the contributor a reasonable chance
to respond while continuing independent preparation. If no answer arrives,
proceed with the otherwise authorized task without adding the field; do not
make optional disclosure a publication gate. Do not persist a global preference
or change Git author/commit metadata as a side effect of this choice.

## Inspect the Patch

- Check the worktree, branch, upstream, remotes, Git identity, PR head, and base.
  Preserve unrelated edits and real changes inside `3rdparty/tvm`.
- Review the complete diff against the PR base, not only unstaged changes.
  Separate the intended fix from cleanup or generated files before staging.
- After switching branches or changing the TVM gitlink, align a clean TVM
  submodule and rebuild before trusting tests that load native libraries. Never
  overwrite a dirty submodule to make the checkout appear clean.

## Audit Test Portability

Read the current CI matrix in `.github/workflows/ci.yml`. Do not assume a test
in a generic `testing/python` directory runs only on the backend used locally.

- Decide whether each added or changed test is portable or backend-specific.
  Add `@tilelang.testing.requires_cuda`, `requires_rocm`, or `requires_metal`
  when it needs that backend's runtime, registered FFI passes, source syntax,
  headers, intrinsics, or code generator.
- Explicit source-only lowering to a target such as
  `{"kind": "cuda", "arch": "sm_100a"}` still requires the CUDA backend to be
  built and registered. Mark it with `requires_cuda` even if no kernel runs.
- Use a `requires_cuda_compute_version*` decorator when execution requires that
  physical GPU capability. An explicit-architecture source-generation test
  normally needs only `requires_cuda`.
- Do not infer CUDA exclusivity from a PyTorch tensor using `device="cuda"`;
  PyTorch's ROCm build uses the same device spelling.
- Do not rely on auto-target selection while asserting backend-specific source.
  Select the target explicitly and add its backend requirement.
- If only one assertion is target-specific, split it from portable lowering or
  runtime coverage. Do not hide useful cross-backend coverage by marking a
  whole file or broad test group.
- Before adding a skip marker to fix CI, confirm the failure is an unsupported
  test contract rather than a real regression in a supposedly portable path.

Apply the requirement directly to every affected test, including parameterized
tests:

```python
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["sum", "max"])
def test_cuda_packed_codegen(op): ...
```

## Validate the Current Head

- Load `tilelang-build` when an install or rebuild is needed. Rebuild native
  code after a branch checkout when the existing build may be stale.
- Run focused tests first, then the containing test file or suite. When
  practical, verify that an unsupported backend skips a target-specific test
  instead of failing during lowering, compilation, source inspection, or device
  setup.
- Run `git diff --check` and `./format.sh`. Inspect every formatter change and
  keep only changes that belong to the PR.
- Treat a green focused test as limited evidence. Report unrelated or
  environment-specific containing-suite failures separately instead of claiming
  that the whole suite passed.
- Reinspect the final staged diff and ensure the recorded validation matches
  commands that actually completed successfully.

## Confirm the PR Title

Follow the [title guidelines](../../../CONTRIBUTING.md#pull-request-titles).
Complete the patch inspection, validation, and PR description before asking.
Before opening a PR or changing its title, show the exact proposed title and
ask the contributor to confirm it or supply a replacement. Use a structured
question tool when available and permitted for confirmation; otherwise ask in
plain text. Ask in the contributor's language while keeping the proposed title
in English. The optional disclosure choice may share the prompt, but title
confirmation does not imply consent to disclosure.

Use a title already supplied or confirmed by the contributor without asking
again. Explicit permission to choose the title without confirmation also
satisfies this step. Preserve an existing PR's title unless a change is needed
or requested; updates that keep the title unchanged and commit/push-only work
do not need this question. If a previously confirmed title changes, confirm
the new text unless the contributor has delegated title selection.

A preselected choice, timeout, or unanswered question is not confirmation.
Without a supplied or confirmed title or delegated title selection, keep PR
creation or renaming pending, including in non-interactive runs. Complete
the otherwise authorized preparation so only the title decision remains.

## Commit, Push, and Verify

- Stage only the intended files and commit with the configured human identity.
  Keep commit messages and public PR text free of unsolicited generated-by
  footers and local machine paths. The optional PR-body disclosure above is
  allowed when the contributor opts in.
- Prefer the existing writable fork remote and never force-push without explicit
  authorization. For an existing PR, verify its repository and head branch
  before pushing.
- Write concise English PR text with specific bracketed component tags, following
  the shared description guidance above. Do not create a draft unless the user
  requests it.
- After pushing, verify that the PR's reported head SHA equals the pushed commit
  and report newly triggered checks as pending, passing, or failing separately.
  Do not wait for CI unless the user asks for monitoring.
