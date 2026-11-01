# File-Review Code Summary

## Purpose

Dispatch one subagent to read `file_input` and generate a factual summary shared by `plan-design` and subsequent item-by-item reviews.

This step organizes only facts that can be confirmed directly from the code or documentation. It does not determine whether rules pass, generate an issue report, or modify files under review.

## Inputs

- Files under review: `{file_input}`
- Summary output path: `{code_summary_output_path}`

## Dispatch Requirements

Pass these inputs to the code-summary subagent and require it to execute the "Subagent Execution Guide" in this file in full. The summary must be written to the specified path. After completion, the subagent returns only the file type, code side, operator name, functional overview, and summary path.

---

## Subagent Execution Guide

The code-summary subagent executes the following steps.

### 1. Define the Review Scope

Enumerate the files explicitly specified by the user in `file_input` to form the list of "files formally under review."

The following related files inside the repository may be read to understand context:

- Public modules imported directly by files formally under review.
- Clearly matching `*_asc.py`, `*_kernel.py`, or public wrappers.
- In-repository code that directly invokes the target entrypoint.
- Test files that correspond directly to the target operator.

Related files are context only and must not be added automatically to the formal review scope. The summary must distinguish "files formally under review" from "context files." Subsequent rule reviews may draw conclusions only about files formally under review.

### 2. Identify File Types and Code Sides

Identify each file independently; do not determine its type solely from its filename. Paths and names may serve only as supporting evidence.

| File type | Identification basis | Code side |
|---|---|---|
| TileLang Kernel | Contains TileLang DSL features such as `@tilelang.jit`, `T.Kernel`, `T.*`, `S.*`, `@T.prim_func`, or `@T.macro` | Kernel |
| Python Host | Ordinary Python logic such as parameter validation, output allocation, backend selection, Kernel construction or invocation, tests, and benchmarks | Host |
| Mixed Python | The same review scope contains both TileLang Kernel and Host invocation logic, or one file serves both responsibilities | Mixed |
| Markdown | `.md` document | N/A |

For multi-file input, record each file's type and side in the summary, then provide an overall conclusion for the current input:

- Kernel files only: overall side is `Kernel`.
- Host files only: overall side is `Host`.
- Both Kernel and Host: overall side is `Mixed`.
- Markdown only: overall side is `N/A`.
- Mixed code and Markdown: determine the code side from code files and mark Markdown separately as `N/A`.

### 3. Analyze Python and TileLang Code

Execute this section only for Python or TileLang files in the formal review scope.

#### 3.1 Entrypoints and Call Relationships

Confirm:

- Public entrypoints callable by users.
- Kernel builders or compilation entrypoints.
- Host-to-Kernel call relationships.
- Key helper functions.
- How tests or upper-level wrappers invoke the target code.

Use `rg` in files formally under review and necessary context files to locate definitions and call sites. If a caller cannot be confirmed, mark it "Unconfirmed"; do not infer that a function is a public entrypoint from its name.

#### 3.2 Dataflow and Computational Semantics

Trace dataflow in this order:

```text
Host inputs and configuration -> Kernel build parameters -> Kernel input Buffers -> movement/computation -> output Buffer -> Host return value
```

Record:

- Input/output counts, shapes, dtypes, strides, and business roles.
- Main loops and the meaning of their iterations.
- Main mathematical operations.
- Output writeback locations.
- Tail blocks, masks, dynamic boundaries, and empty-input handling.
- Conditional branches that affect dataflow.

Mathematical formulas and business meaning must be supported by code, tests, or call relationships. If only naming clues exist without implementation evidence, mark them "Inferred; confirmation required."

#### 3.3 Parameter Sources and Defenses

Trace key values that affect indexing, capacity, accuracy, or dispatch:

- Shape, dtype, and stride.
- Tile/block sizes and core count.
- Loop bounds and index upper bounds.
- `num_stages`, Buffer counts, and memory levels.
- Algorithm modes, quantization configuration, and other compile-time parameters.
- Runtime scalars and `T.dynamic` values.

For every key value, record its definition, propagation, actual validation, and final use locations. A value originating from Host code, configuration, or a hardware query does not mean it has been validated. Record a value as defended only when an explicit assertion, exception, conditional branch, or call constraint is found.

#### 3.4 TileLang API and Execution-Structure Index

Record what actually appears in files formally under review:

- Decorators such as `@tilelang.jit`, `@T.prim_func`, and `@T.macro`.
- `T.Kernel`, `T.serial`, `T.Parallel`, and `T.Pipelined`.
- `T.alloc_*`, `T.copy`, `T.gemm`, reductions, and atomic operations.
- `T.call_extern`, synchronization operations, and `S.*`.
- Core, thread, or program-block mappings.
- Buffer shapes, dtypes, memory levels, and lifetimes.
- Pipeline stages and explicit data dependencies.

This step creates only an API-call index and records code facts. It does not investigate external API semantics or replace subsequent rule reviews.

#### 3.5 Performance-Structure Facts

Record only structures directly proven by the code:

- How tiles or blocks are partitioned.
- The processing range of one core or program instance.
- Whether multistage pipelines, multilevel Buffers, SIMD, SIMT, GEMM, or persistent loops exist.
- Statically computable capacity of each Buffer.
- Whether tail blocks add extra work.

Without benchmark, profiling, or explicit static evidence, do not conclude that performance is "worse" or "better."

### 4. Analyze Markdown Documents

Execute this section only for Markdown files in the formal review scope. Markdown always has code side `N/A`; do not generate Kernel, Host, Buffer, or pipeline analysis for it.

Record:

- Document purpose and main sections.
- Heading hierarchy.
- Code fences and command examples.
- File paths, relative links, and anchors.
- APIs, variables, status names, and normative-level terms.
- References to other `SKILL.md` files, `workflows/`, `steps/`, `references/`, or scripts.
- Reference relationships among multiple Markdown files.

This section only builds an index for locating rules in `references/doc-style.md`; it does not determine whether the document violates any rule.

### 5. Generate the Summary

Write the summary to `{code_summary_output_path}`. If its parent directory does not exist, create the parent directory first. Use the following concise template. Omit sections that do not apply instead of generating large empty tables.

```markdown
# File-Review Summary

## Review Scope

| File | File Type | Code Side | Role | Scope |
|---|---|---|---|---|
| {path} | TileLang/Python/Markdown | Kernel/Host/Mixed/N/A | {entrypoint/Kernel/test/document/etc.} | Formal review |
| {path} | {type} | {side} | {context role} | Context only |

Overall file type: {Python/TileLang/Markdown/Mixed}
Overall code side: {Kernel/Host/Mixed/N/A}
Operator or document name: {name}
Functional overview: {one evidence-supported sentence}

## Code Flow

> Generated only for code input.

Entrypoint and call chain: {entrypoint -> builder/wrapper -> Kernel}

Dataflow: {Host input -> Kernel parameters/Buffers -> movement and computation -> output}

Main computation: {formula or implementation semantics; explicitly mark when unconfirmed}

### Key Branches and Boundaries

| Condition | Location | Trigger Scenario | Handling Logic |
|---|---|---|---|
| {condition} | {file:line} | {scenario} | {behavior} |

### Parameter Sources and Defenses

| Parameter | Definition/Source | Validation Location | Use Location | Confirmed Constraint |
|---|---|---|---|---|
| {name} | {file:line/expression} | {file:line or not found} | {file:line} | {constraint or unconfirmed} |

### Function and Call Index

| Function | Location | Role | Direct Callers |
|---|---|---|---|
| {name} | {file:start-end} | {entrypoint/Kernel/helper/test} | {caller:line or unconfirmed} |

### TileLang API Index

| API | Location | Invocation Context |
|---|---|---|
| {API} | {file:line} | {brief parameters and purpose} |

### Buffers, Partitioning, and Pipelines

| Object/Mechanism | Location | Shape/Dtype/Level or Configuration | Purpose and Dependencies |
|---|---|---|---|
| {buffer/pipeline/loop} | {file:line} | {facts} | {facts} |

## Markdown Index

> Generated only for Markdown input.

| Document | Purpose | Section Structure | Code/Commands | Paths, Links, and References |
|---|---|---|---|---|
| {path} | {purpose} | {headings} | {locations} | {locations and targets} |

## Cross-File Relationships

| Source File | Target File | Relationship | Location | In Formal Review Scope? |
|---|---|---|---|---|
| {source} | {target} | import/call/parameter propagation/document reference | {file:line} | Yes/No |

## Unconfirmed Information

- {missing evidence, unresolved calls, or undetermined parameter constraints; write "None" when empty}
```

### 6. Return the Result

After writing the summary successfully, return to the main workflow:

```text
File type: {Python/TileLang/Markdown/Mixed}
Code side: {Kernel/Host/Mixed/N/A}
Operator name: {operator_name; for documentation-only input, use the document or directory name}
Functional overview: {one sentence}
Summary path: {code_summary_output_path}
```

## Constraints

- Read every file formally under review and list each one in the summary.
- Related files may be read as context but must not expand the formal review scope.
- Every code-side classification, call relationship, parameter constraint, and business conclusion must cite a source or document location.
- Explicitly mark information that cannot be confirmed; do not complete or guess it.
- Markdown must have code side `N/A`; do not classify it as Host or Kernel.
- Do not perform API research, design-conformance checks, rule decisions, or performance evaluations.
- Write only the specified summary file; do not modify `file_input` or context files.
