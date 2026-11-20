# TileLang Operator Development Workflow

This workflow covers TileLang operator design, generation, system testing (ST), performance tuning, and final review on NPUs; it excludes CUDA, Metal, and other backends.

Default sequence: design -> generation -> specified cases pass accuracy validation -> ST -> ask whether to tune -> tune only after confirmation -> ask whether to review -> review only after confirmation. Proceed from design directly to generation. If required information is missing, pause and request it. Generalized-case expansion belongs only to ST.

## Source and Test Path Conventions

Resolve paths from the repository root supplied by the caller, or search upward for a directory containing `src/ascend/`, `src/backend/`, and `examples/ascend/`. Framework code is under `src/ascend/`, `src/backend/`, and `tilelang/`; examples under `examples/ascend/`; tests under `testing/ascend/`.

`{code_dir}` may contain only the target operator, tests, and required dependencies. Reuse the confirmed `{repo_root}` and verify that it matches the imported version. Prefer caller-supplied tests; otherwise scan `testing/ascend/`, `examples/ascend/`, and operator-artifact directories. Inspect pytest configuration/plugins before running; assume no particular level, marker, or device-binding plugin. Acceptance must cover every established target case despite name migrations.

## Backend-Selection Constraints

Never hard-code a TileLang backend in operator, test, profiling, or helper Python code through `target="..."`, `target='...'`, equivalent arguments, or backend environment variables. JIT decorators, builders, and compilation APIs omit `target`. The launch command selects PTO with `TILELANG_DEFAULT_TARGET=pto` or AscendC with `TILELANG_DEFAULT_TARGET=ascend`; switching backends never changes source.

## Hardware Identification

Before design, load `npu-arch` by name and obtain `full_soc`, `npu_arch`, and complete `evidence`. Continue only for an Ascend950PR/DT device with `npu_arch=3510` and no architecture conflict; otherwise stop and report. Pass the evidence to design, generation, post-ST tuning, and tuning subagents. Reuse it while device/configuration remain unchanged. An independently entered stage without valid evidence must rerun `npu-arch`; never infer the complete model from a product family, directory, or static mapping.

## Solution Design

Load `tilelang-op-design`, generate `design.md` from its template, and run its self-check, including when reusing a report. If inputs are insufficient, clarify them first; never emit an incomplete report. Obtain the target backend from context or `design.md`, require agreement when both specify it, and pass it to generation, ST, and tuning (`pto` -> `TILELANG_DEFAULT_TARGET=pto`; AscendC -> `TILELANG_DEFAULT_TARGET=ascend`).

Paths are relative to the directory containing this `AGENTS.md`.

## Operator Generation

Load `tilelang-op-generate` with the target backend. Implement and debug the specified cases, delivering only `<op_name>.py` and `test_<op_name>.py` under `operators/<op_name>/` unless the user or an existing operator directory specifies otherwise. Finish only after every specified case actually passes accuracy validation; report both paths and results, then proceed directly to ST.

## System Testing

Load `tilelang-op-test` with the target backend. Review the contract and `test_<op_name>.py`; add applicable functional, accuracy, boundary, gradient, state-mutation, invalid-input, layout/interface, backend/branch, and randomness coverage with reproducible evidence. Fix operator defects in generation, then rerun affected tests.

Report the pytest path, trustworthy coverage conclusion, and actual results. Ask: "The operator has been generated and system testing is complete. Continue with performance tuning?" Pause for an explicit answer; earlier workflow authorization is insufficient. On confirmation, pass the source directory, kernel, test, backend, original cases, ST results, `full_soc`, `npu_arch`, and complete `evidence` into tuning. On refusal stop; without an answer remain paused and do not load tuning or collect performance data.

## Performance Tuning

After entering tuning, the current agent automatically acts as the `tilelang-tuning` primary agent; do not ask the user to explicitly invoke a custom agent, specify an agent name, or provide a workflow Markdown path. Read and execute its complete definition and referenced workflows without condensing, rewriting, or skipping rules.

In the workflow, both "the user's current working directory" and `{cwd}` refer to the directory containing this `AGENTS.md`. Therefore, write all tuning artifacts under `operators/` in this directory.

If the role cannot be loaded, stop, report an incomplete installation, and tell the user to rerun this source package's `init.sh`. After tuning and final selection, ask: "Performance tuning is complete. Perform a code review of the final delivery code?" Pause for explicit confirmation; prior authorization is insufficient. On refusal stop; without an answer remain paused and do not review.

## Code Review

Review only selected `<op_name>.py`, `test_<op_name>.py`, and explicitly delivered runtime/test helpers; after tuning, `final_optimized/` is authoritative. Exclude profiling artifacts, comparisons, debug scripts, and discarded candidates. Only after explicit user agreement, load `tilelang-op-review` and follow its environment detection, format check, reporting, fix-confirmation, and final-validation workflow. Do not substitute another review process or ask for the code again. Show the assembled review-file list and pass that same list as explicit script input.

## Codex NPU Device Commands

In Codex, any command that accesses an Ascend NPU or inspects NPU device nodes must be launched directly with the following prefix:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES
```

This covers `pytest`, accuracy scripts, warm-ups, `msprof`, `npu-smi`, `asys`, and device-node checks. Never try the ordinary sandbox first. The prefix only clears logical mapping; Codex must also set `sandbox_permissions=require_escalated` on the tool call, requesting approval on the first run when no matching persistent rule exists and reusing one when available. Device arguments use physical IDs, such as `--device 0`.

Match correctness-test concurrency to actually available NPUs. With one physical device, use one worker. Never preimport TileLang's `sitecustomize.py` into pytest, because it can corrupt the xdist startup handshake.
