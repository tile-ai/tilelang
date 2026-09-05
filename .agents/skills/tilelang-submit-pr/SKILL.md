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
def test_cuda_packed_codegen(op):
    ...
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

## Commit, Push, and Verify

- Stage only the intended files and commit with the configured human identity.
  Keep commit messages and public PR text free of automation traces and local
  machine paths.
- Prefer the existing writable fork remote and never force-push without explicit
  authorization. For an existing PR, verify its repository and head branch
  before pushing.
- Write concise English PR text with specific bracketed component tags and list
  only completed validation. Do not create a draft unless the user requests it.
- After pushing, verify that the PR's reported head SHA equals the pushed commit
  and report newly triggered checks as pending, passing, or failing separately.
  Do not wait for CI unless the user asks for monitoring.
