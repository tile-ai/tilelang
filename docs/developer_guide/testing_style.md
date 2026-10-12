# Testing Style Guide

Bug fixes need regression coverage, not automatically another test. Add or
expand tests only for behavior the existing suite does not already protect.

- **Identify the gap.** Name the failure and the assertion that catches it.
  Check nearby tests and their target/pass settings; a passing suite alone is
  not proof of coverage. Verify failure before the fix when practical.
- **Extend before duplicating.** Prefer a small case or assertion in the relevant
  test family. Reuse kernels, verifiers, and setup instead of copying a test file
  for a minor variant. A separate test is fine for different behavior or setup.
- **Assert behavior, not the patch.** Do not scan repository source for the fix,
  assert constants or mock results unrelated to the bug, or compute the expected
  result with the implementation under test. Keep runtime coverage when only
  execution exposes the failure.
- **Keep cases focused.** Use the smallest input that still triggers the bug.
  Add shapes, dtypes, and configurations only for distinct affected paths,
  boundaries, or necessary interactions. Avoid unrelated parameter products and
  repeated kernel compilation that add cost without coverage.
- **Keep the change local.** Reuse existing helpers; do not introduce a fixture
  framework or refactor unrelated tests for a small regression.

If existing tests already catch the failure, run them rather than add an
equivalent permanent test. Preserve distinct regression coverage: fewer test
functions or files is not the goal.
