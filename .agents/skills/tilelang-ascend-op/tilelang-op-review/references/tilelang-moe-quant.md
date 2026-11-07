# Domain-Specific Code Review Rules for TileLang MoE and Quantization Operators

<applicability>
Language: Python, TileLang DSL
Side: All
Domain: true
Triggers: moe, expert, topk, top_k, stable_topk, quant, scale, sf_block, packed_ue8m0, FP4, FP8
Enabled by default: true

Excluded scenarios: `expert` or `quant` appears only in a variable name or comment, with no actual MoE or quantization logic
</applicability>

<review_load>
General review subagent rule capacity limit: 2
</review_load>

## Purpose

Check for issues related to TileLang MoE routing and low-precision quantization.

## Quick Index

| Rule ID | Title | Applicable Scenarios | Severity |
|---------|-------|----------------------|----------|
| MQ-01 | Expert Index Bounds | MoE | High |
| MQ-02 | MoE Parameter and Output Contracts | MoE | High |
| MQ-03 | Stable TopK, Mask, and Normalization Semantics | MoE, TopK | High |
| MQ-04 | Consistency Among Quantization Dtype and Logical/Physical Storage | Quant | High |
| MQ-05 | Consistency Among Scale Block, Layout, Stride, and Packed Protocol | Quant | High |
| MQ-06 | Correctness of Scale Computation, Rounding, Saturation, and Tail-Block Precision | Quant | High |

## Scope Overview

> This file covers MoE expert indices, parameter contracts, and quantization correctness in TileLang.
>
> **Rule category overview**:
>
> | Category | Rule Range | Core Review Concerns |
> |----------|------------|----------------------|
> | MoE Expert Routing | MQ-01~03 | Expert indices and mapping, parameter contracts, stable TopK, and mask semantics |
> | Quantization Correctness | MQ-04~06 | Dtype/storage protocols, scale layout, scale computation, and correctness protection |
>
> **Applicable scenarios**: Code review of TileLang MoE and quantization operators

## Glossary

| Term | Meaning |
|------|---------|
| Expert Index | The expert number selected by a token; its valid range is `[0, num_experts)` |
| Scale Layout | The block shape, packed representation, and row/column layout of a quantization scale |
| Logical/Physical Shape | The public shape of numerical elements and the actual shape occupied by packed storage |

## PR Diff-to-Rule Quick Reference

> **How to use**: Inspect keywords in the PR diff, match them against the table below, and read only the rules in the corresponding category. There is no need to read every rule at once.

| Diff Keywords | Corresponding Category | Rules to Read | Typical Search Command |
|---------------|------------------------|---------------|------------------------|
| `expert` / `topk` / `mask` / `routing` | MoE Expert Routing | MQ-01~03 | `rg -n -i 'expert|topk|routing|mask' <operator_path> <test_path> -g '*.py'` |
| `quant` / `scale` / `FP4` / `FP8` / `sf_block` | Quantization Correctness | MQ-04~06 | `rg -n -i 'quant|scale|sf_block|packed|fp4|fp8' <operator_path> <test_path> -g '*.py'` |

## Domain Identification Rules

### Core Features (Any One Is Sufficient)

- The change modifies expert selection, mapping, or TopK in the current operator.
- The change modifies the quantization format or scale layout in the current operator.

### Excluded Scenarios

- Ordinary single-device TopK that does not involve an expert index or MoE parameters.
- Ordinary casts that do not involve a quantization format, scale, or low-precision numerical semantics.

---

## I. MoE Expert Routing Rules

### MQ-01: Expert Index Bounds `[Applicable: All]` `[Red Line]`

**Issue Description**

A valid expert index must fall within `[0, num_experts)` for its corresponding logical or physical expert space. Padding sentinels, negative values, uninitialized indices, and logical expert IDs that have not yet been mapped must not participate in physical address computation.

**Review Method**

1. Trace the complete index lifecycle: generation, tie-breaking, padding, TopK, logical-to-physical mapping, writeback, and downstream consumption.
2. At each stage, identify the index space, valid upper bound, and sentinel explicitly. Do not use the same `num_experts` ambiguously for different spaces.
3. Check `1 <= num_topk <= num_experts` and the output width after shared/duplicate expert expansion.
4. Range checks for dynamic indices must occur before the actual GM load/store; accessing first and masking afterward is ineffective.

**Exclusion Rules**

A sentinel is acceptable only when it is written to an invalid output position explicitly defined by the interface and every consumer reliably filters it before address computation.

**Decision Method**

Assign `FAIL` when an invalid index can reach a valid output or address computation, a mapping-table index can go out of bounds, or logical and physical expert spaces are mixed.

---

### MQ-02: MoE Parameter and Output Contracts `[Applicable: Host]` `[Red Line]`

**Issue Description**

The MoE wrapper, Kernel factory, Kernel signature, reference, and optional `out` parameter must use a single contract for the expert count, topk, score dtype, mapping table, and output shape/dtype/device. An assertion only inside the Kernel causes a public entry point to fail late during compilation or execution.

**Review Method**

- Check the logits/score rank, dtype, device, contiguous/stride properties, and position of the expert axis.
- Check the lower bounds, upper bounds, divisibility, and mutually exclusive combinations of `num_topk`, shared experts, and group/expert counts.
- Cross-check optional parameters in pairs, including bias/mask, mapping/count, and fix-routing/unmapped-index; determine whether one is required when the other is present.
- Verify that newly allocated outputs and user-provided `out` buffers match in shape, dtype, layout, and device, including the empty-token return path.

**Decision Method**

Assign `FAIL` when a publicly accepted input has inconsistent meaning at any of the wrapper, Kernel, or reference layers, or when output-buffer contracts are inconsistent. Assign `PASS` when complete entry-point validation dominates the invocation.

---

### MQ-03: Stable TopK, Mask, and Normalization Semantics `[Applicable: All]` `[Red Line]`

**Issue Description**

TopK is not merely selecting the largest values: stable ordering for equal scores, NaN/Inf handling, bias, mask precedence, fixed/random routing, normalization, and scaling order are all part of the public numerical semantics. Changing a comparison operator or padding value may fail only with duplicate values, all-zero rows, or the trailing expert group.

**Review Method**

1. Determine tie-breaking behavior from the public wrapper/reference. In this repository, equal scores in `topk_gate` select the smaller expert index.
2. Verify that padding lanes use values that cannot win and that invalid lanes are cleared before index/weight writeback.
3. Build a precedence table for mask, force-random, fixed-routing, bias/image-bias, and check every combination and the empty-token path.
4. Verify the ordering among scoring modes such as softmax/sigmoid/identity, weight normalization, and `routed_scaling_factor`.

**Exclusion Rules**

If a special value, random path, or normalization mode is outside the public interface, the entry point must explicitly reject it or route it to a supported implementation. It must not be excluded by default merely because tests do not cover it.

**Decision Method**

Assign `FAIL` when stable ordering, mask/sentinel behavior, scoring, or normalization differs from the public reference. Random-distribution quality requires reproducible statistical tests and must not be judged solely from code shape.

---

## II. Quantization Correctness Rules

### MQ-04: Consistency Among Quantization Dtype and Logical/Physical Storage `[Applicable: All]` `[Red Line]`

**Issue Description**

A quantization format name, torch dtype, TileLang dtype, and actual storage dtype do not necessarily correspond one-to-one. For example, packed FP4 may be carried by wider integer storage, making the logical hidden size different from the physical element count. Treating the storage dtype as the numerical dtype causes errors in shape, indexing, and output protocols.

**Review Method**

- Trace input/output formats from the configuration object to torch dtype, TileLang dtype, storage dtype, and pack factor.
- Verify total bytes, alignment, and logical/physical hidden-size conversion before and after `T.view`/torch view.
- Check that the wrapper, Kernel, reference, consumer, and output type annotation use the same format semantics.

**Decision Method**

Assign `FAIL` when the dtype, pack factor, or logical/physical shape changes without justification and alters the mapping of valid elements. `T.reinterpret` is allowed only for an explicit bit protocol; it is not a substitute for numerical conversion.

---

### MQ-05: Consistency Among Scale Block, Layout, Stride, and Packed Protocol `[Applicable: All]` `[Red Line]`

**Issue Description**

The scale Tensor's `sf_block`, shape, row/column-major layout, stride, packed/unpacked representation, and post-transpose view must agree between producer and consumer. Obtain a device-specific pack factor from a helper/configuration in the current repository; do not copy it from another backend.

**Review Method**

1. Independently compute the expected scale shape from the input logical shape and `sf_block`.
2. Trace column-major transposition, packed UE8M0 views, slicing, and the final layout visible to the user after the epilogue.
3. Verify the unit and contiguous dimension of dynamic strides, especially for short scale rows and tail blocks.
4. Check that wrapper dispatch and Kernel specialization agree for every supported configuration.

```python
expected_sf_shape = get_sf_shape(input_shape, config)
assert scale.shape == expected_sf_shape
```

**Decision Method**

Assign `FAIL` when producer and consumer interpret the same scale element position, pack word, or stride differently. An equal total Tensor element count alone does not rule out a layout error.

---

### MQ-06: Correctness of Scale Computation, Rounding, Saturation, and Tail-Block Precision `[Applicable: Kernel]` `[Red Line]`

**Issue Description**

The statistical dtype for amax/scale, positive floor for zero amax, power-of-two rounding, saturation range, and quantization/dequantization order jointly determine the error. Invalid lanes in a tail block contaminate the entire block's scale if they participate in amax. Uncleared packed output may leak uninitialized bits.

**Review Method**

- Use precision that satisfies the reference for intermediate amax and scale computations, and use the configuration-defined positive floor for an all-zero block.
- Verify the ordering of round-to-power-of-two, reciprocal scale, format maximum, and final cast.
- Check FP4/FP8 overflow, underflow, NaN/Inf, positive/negative zero, and stochastic-rounding semantics.
- Include only valid inputs in tail-block statistics; clear invalid packed lanes according to the interface protocol before writeback.

**Evidence Requirements**

Use fp32 reference tests covering zero blocks, extreme values, positive and negative values, nonintegral blocks, packed/unpacked representations, and round-trip dequantization. Do not obtain a pass by relaxing tolerances or masking valid lanes.

**Decision Method**

Assign `FAIL` when the implementation's formula, rounding, saturation, or valid-lane results differ from the public reference. Do not report an issue under this rule when correctness is preserved and only performance differs.

## Complete Search Keyword List

```bash
rg -n -i 'expert|topk|routing|quant|scale|sf_block|packed|fp4|fp8' <operator_path> <test_path> -g '*.py'
```
