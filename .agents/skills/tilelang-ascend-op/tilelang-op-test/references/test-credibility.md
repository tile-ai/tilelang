# Test Credibility

Read this reference when determining whether existing or generated tests can support a correctness conclusion.

## Credibility Requires Both Breadth and Depth

An operator conclusion can be strengthened along two independent dimensions:

- **Case credibility (depth)**: Every case has a traceable contract, valid inputs, an independent oracle, complete assertions, path evidence, and execution evidence;
- **Coverage completeness (breadth)**: Trustworthy cases cover all applicable requirements for functionality, correctness, boundaries, gradients, state, invalid-input rejection, layout and interfaces, backends and paths, and randomness.

Complete coverage design therefore increases credibility, but it cannot make an incorrect oracle or weak assertion trustworthy. Conversely, a highly credible case cannot prove behavior outside the requirements and paths it covers.

Treat customer-confirmed requirements and authoritative specifications as the primary standard. Repository patterns summarized from mature tests may guide general choices of comparators, case structure, and evidence forms, but they cannot override the customer contract or directly establish an operator-specific tolerance without evidence.

## Assess Credibility Separately for Each Verification Conclusion

Do not label an entire test broadly as credible or not credible. Record what each assertion actually proves. For example, `actual.shape == expected.shape` supports an output-shape conclusion but does not prove numerical correctness.

Use these labels:

- `TRUSTED`: Sufficient traceable evidence supports the stated conclusion.
- `PARTIAL`: Useful evidence exists, but a material part of the conclusion remains unchecked.
- `UNTRUSTED`: The assertion could still pass when the claimed behavior is wrong, or the oracle/input is itself invalid.
- `UNKNOWN`: The required contract or execution evidence cannot be obtained.

## Six Gates

### 1. Contract Source

State the expected behavior and cite its source. Reliable sources include customer- or user-confirmed requirements, public specifications, public API documentation, and explicit mathematical definitions. Public docstrings and independent references are often the most reliable local evidence in the current repository.

Assertions in the implementation and existing tests only describe current behavior. When no more reliable source exists, they may provisionally establish interface limits, but the limitation must be stated. When implementation and documentation conflict, do not silently choose one side.

### 2. Input Validity

For a positive test, prove that dtype, shape, layout, optional parameters, scalar ranges, and cross-parameter constraints together form a supported input. For a negative test, cite the invalid-input rejection contract and expected exception.

Without evidence, do not turn an internal alignment requirement into a public input restriction. Conversely, if the interface requires an input to be divisible by `B`, do not describe `B+1` as a valid tail case.

### 3. Oracle Independence

Prefer these primary oracles:

- Direct exact answers for small deterministic inputs;
- Independent PyTorch or CPU implementations;
- Explicit mathematical computation;
- A well-justified metamorphic relation when a complete reference is impractical.

A mature CUDA or legacy implementation provides important differential-validation value, especially for an Ascend port, but it is usually supporting evidence only because different implementations may share assumptions or copy the same logic. Compare `reference ↔ CUDA ↔ Ascend` when possible. Investigate disagreements; do not decide by majority vote.

Check whether the reference calls the DUT, reuses its generated kernel, or copies a suspicious implementation shortcut. A floating-point reference may intentionally use the same operation order to satisfy an existing documented numerical contract. Record this dependency rather than claiming full independence.

### 4. Assertion Completeness

Choose assertions according to the contract:

- Exact integer, index, mask, or bit-pattern outputs: use exact comparison;
- Floating-point outputs: use the repository- or operator-specific comparison method and tolerance;
- Multiple outputs: verify every material output and optional-output behavior;
- Gradients: verify the gradient for every required input;
- In-place or stateful operations: verify changed state and protected regions or regions that must remain unchanged;
- Special values: verify NaN locations, Inf signs, and finite values as required;
- Invalid inputs: verify the explicit exception type and meaningful error text when it is stable.

Never weaken the oracle, relax tolerances, reduce cases, or weaken assertions to make an incorrect implementation pass.

#### Correctness Criteria

Choose the comparison method from output semantics before selecting numerical values:

- Use exact equality for integers, indices, booleans, masks, counters, and bit patterns that must match;
- Use exact equality for floating-point output only when the contract requires deterministic bitwise identity and both paths have identical permitted rounding behavior;
- Otherwise, use approximate comparison and specify `rtol` and `atol` explicitly for every output.

Approximate comparison usually applies this condition:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

`atol` protects values near zero, while `rtol` scales with nonzero reference values. Determine tolerances in this priority order: customer or contract requirements; authoritative prototype or upstream compatibility requirements; a justified numerical error budget; and only then a mature repository precedent with the same operator semantics, input/output dtypes, reduction scale, reference-conversion method, and backend path. A tolerance observed in another BF16 test is only a candidate, not a universal standard for all BF16 operators.

As applicable, the error budget should account for sources such as input quantization, approximation methods, accumulation dtype and length, operation order, and final output-type conversion. Record the standard's source and rationale beside the assertion. Establish separate criteria when primary outputs, scale/amax, auxiliary counters, and gradients have different numerical behavior.

When an approximate comparison fails, inspect the distributions of absolute and relative error, special values, and worst input regions. Fix the implementation, reference, input, or existing standard based on evidence; do not tune the tolerance to one failing sample merely to obtain a pass. As applicable, include zeros, tiny values, ordinary values, large valid values, and cancellation-prone values so that a purely relative or absolute threshold does not conceal defects.

For a stochastic operator, a fixed random seed guarantees only reproducibility, not a correct distribution. Check deterministic invariants on every run and use a justified sample size and statistical acceptance criteria to assess the distribution.

### 5. Path Evidence

Prove that the final case actually enters the claimed device, dispatch path, and boundary path. Evidence may come from explicit dispatch conditions, final runtime dimensions, collected parameter IDs, kernel source output, or profiler/trace data when necessary.

If a generator aligns, pads, filters, or otherwise transforms the original requested size, do not infer the execution path directly from the original size. If a function has an explicit JIT target, do not claim that a backend was used based only on the directory name or environment variable.

### 6. Defect-Detection Evidence

State the real defect class that a test can detect. A valid case should fail when the corresponding defect is present. For novel test patterns intended for reuse, validate representative defect-detection behavior in an isolated fixture or temporary copy, for example:

- Omitting the last valid element;
- Swapping two output fields;
- Ignoring an optional weight;
- Corrupting protected padding;
- Accepting an input that should be rejected.

Do not inject defects into the implementation in the user's worktree unless the task explicitly requires it. Not every case needs fault-injection testing; use it only for a new assertion pattern or when credibility remains unclear.

## Execution State

Record lifecycle separately from credibility:

`designed → implemented → collected → executed`

The result of an executed case is `passed`, `failed`, `skipped`, `xfailed`, or `error`. Existing in source, collected, skipped, and executed successfully are distinct facts. A missing device may explain `skipped` or `error`, but it cannot upgrade a case to verified coverage.
