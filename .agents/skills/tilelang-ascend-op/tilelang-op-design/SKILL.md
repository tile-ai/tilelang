---
name: tilelang-op-design
description: Generate or revise the TileLang design document design.md for Ascend 950 (A5) based on operator requirements. When the necessary input is insufficient, pause and guide the user to supplement. After the information is clear, design the implementation plan and accuracy verification method of the specified case. It is used to create new operators, migrate operators or revise the design based on implementation feedback; it is not responsible for generalized use case expansion and performance tuning.
---

# TileLang A5 operator solution design

## 1. Goals and Scope

Generate `design.md` according to the operator requirements, providing an implementation solution that can directly guide coding for subsequent models. The report should independently describe the following core decisions without relying on historical dialogue:

- **Computation definition and support scope**: input and output, attribute semantics, specified case and accuracy requirements.
- **Algorithm and execution mode**: kernel partitioning, fusion strategy, AIC/AIV division of labor and SIMD/SIMT selection.
- **API mapping and implementation basis**: key operations, parameters, applicable conditions and source code reference of corresponding versions.
- **Tiling and data flow**: task division, index mapping, storage layout, data handling and resource budgeting.
- **Scheduling and synchronization**: loop structure, pipeline series, buffer life cycle and data dependency.
- **Accuracy Verification**: Input generation, reference results and comparison criteria for specified cases.

**[design.md template](assets/design-template.md) is the structure specification of the report, which must be read completely and filled in according to the original structure. ** The Phase of this skill is a work step, not a report directory;

The report directly states the design decisions, rationales, and implementation requirements. The verification chapter explains the method and judgment criteria, and does not fill in the progress description or debugging process description such as "not compiled, not executed, no PASS"; the execution record is maintained by the implementation and testing phase. Technical assumptions or dependencies that affect the feasibility of the solution should be specified in detail, and analysis should not be replaced by a general "to be verified".

This skill only designs user-specified cases and does not automatically expand shape, dtype, exception input or general fallback. The report guides subsequent model implementation operators and verifies the accuracy of specified cases; generalized use case expansion belongs to the subsequent testing stage, and the report does not set test grading or special handover chapters.

Use a single skill, single report, without introducing multi-agent segmentation, document assembly, or additional approvals. The current stage does not generate operator code, test scripts, spec/proto, status files or independent iteration plans.

**Requirements completeness pre-check must be completed before design. When the necessary input is insufficient or there is ambiguity that affects the design, the design is suspended and the user is guided to make supplements; after receiving a valid reply and passing the pre-inspection again, the scheme design can be started again. Default values, placeholders to be added, or partial design reports may not be substituted for requirements clarification. **

## 2. Input requirements

### 2.1 Required information

The following information needs to be clear in the design.

| Information | Content Requirements |
|---|---|
| Operator identification and calculation definition | Operator name, mathematical formula or equivalent reference implementation; reduction axis, broadcast rules, scaling, epsilon, activation order and other related semantics |
| Input and output specifications | Tensor and scalar parameters, shape, dtype, necessary stride/layout, return order, and in-place update or initialization requirements |
| Specify case | The input and attribute combination of this implementation; clarify the compile-time fixed parameters and runtime parameters |
| Target backend | PTO or AscendC; asks user if not specified, deprecated or selected by default |
| Reference basis | The original operator interface, implementation or golden in the migration task; the new operator can be created by calculation definition golden |

The complete support scope of the reference operator does not automatically become the scope of this delivery. When modifying an existing operator, the existing capabilities of the original interface cannot be deleted on the grounds that this case is limited.

**Necessary inputs are determined according to operator semantics and are not limited to the above table. ** For example, Reduce needs to clarify the reduction operation, axis/dim and keepdim, Norm needs to clarify the normalization range and epsilon, and GEMM needs to clarify the logical transpose and fusion operations. Any parameters that affect calculation results, output shape, or interface behavior and cannot be determined from existing data should be clarified first.

### 2.2 Supplementary information

Priority is given to accuracy standards, operating environments, target backends, dependency constraints, and performance goals provided by users. In the absence of established accuracy standards, well-founded suggested values ​​can be put forward and marked as design suggestions; when there are no performance targets, no additional performance thresholds are set.

Implementation parameters such as tiles, threads, number of cores, pipeline and buffer configuration are determined by the model and are not required by the user; the implementation constraints specified by the user must still be adhered to.

### 2.3 Requirements completeness pre-check

Check that the information in Section 2.1 is complete and consistent in context, reference interfaces, and specified tests. When the calculation semantics, input specifications, current case or output behavior are still unclear, or there are issues such as axis out-of-bounds or specification conflicts, they must be clarified first.

The target backend is determined by the user's most recent explicit selection in this design task, and cross-round replies are also valid; if there is already a `design.md` record of a legal target backend and the user does not specify a different backend, it will be used without repeated inquiries. When the target backend is reported to be missing, fill it in if it has been explicitly specified in this round or the previous round. Otherwise, ask the user to choose PTO or AscendC and fill it in after replying. If the user selection is different from the report, clarify it first, and then revise the report and affected designs after confirming the change.

Information that can be uniquely derived is used directly without repeated inquiries; complete calculation definitions can be used to build golden without requiring additional documents. Users can determine and indicate the inputs for explicit authorization model selection by themselves, but general instructions such as "continue" do not equal authorization to complete the requirements.

The pre-inspection is passed after the demand gap is eliminated. Section 2.2 allows matters determined by the model to not block preflight; unknown API or hardware capabilities are verified according to Section 3.3.

### 2.4 Guided clarification and suspension rules

When the preflight fails, perform the following interactions:

1. Briefly describe the known needs and gaps, ask 1 to 3 necessary questions in each round, and attach a brief parameter description and filling examples. Accepts natural language, reference functions, or concrete test cases; examples are not used as defaults.
2. Pause the design after asking questions, do not enter Phase 2~5, do not generate or modify `design.md`; end the current round with a clear question, do not rely on waiting tools to keep the round running. Empty replies, timeouts, or uncommitted options do not count as replies.
3. The user's subsequent responses are regarded as continued input for the same design task; the confirmed platform, case, path and backend are retained, and solved questions are not asked again. Re-pre-inspection will only ask about the remaining gaps; after passing, it will directly continue with the design and subsequent stages authorized by the caller, without the need for re-approval.

For example, the Reduce input is known to be `(4,32,128), float32`, which can be supplemented by:

- `reduction`: reduction operation, such as `sum` (sum), `mean` (mean).
- `axis`: reduction axis, such as `-1` represents the last dimension.
- `keepdim`: whether to keep the reduction dimension. When reducing along the last dimension, `True` outputs `(4,32,1)` and `False` outputs `(4,32)`.

Example of filling in: `reduction=sum, axis=-1, keepdim=True`. Only parameters that are not yet explicit are asked; other operators organize questions according to their actual semantics.

## 3. Technical constraints and information basis

### 3.1 Target version and platform constraints

After passing the requirement pre-check, first load the Skill named `npu-arch`, and reuse the `full_soc`, `npu_arch` and complete `evidence` passed in by the caller; when the evidence is missing, incomplete or the device/configuration changes, run its own detection script in the target device environment according to the Skill's hardware evidence process. It stops when the user specifies another target machine but cannot obtain valid evidence on that machine, and the results of the current machine must not be used instead. Stops and reports when the Skill is not installed, cannot be loaded by name, or the evidence does not satisfy the combination of Ascend950PR/DT series with `npu_arch=3510`, without looking for scripts via adjacent directory paths:

- Confirm the complete `full_soc`, SocVersion/NpuArch, the actual number of available AIC/AIV cores and the UB, L1, L0A/B/C capacities involved in the solution; query L2, video memory, bandwidth and theoretical computing power on demand. The runtime interface return value is preferred, and the SKU variation parameters must match the complete model; `short_soc`, typical SKU tables, or CANN configurations of unmatched devices cannot be used as actual machine parameters.
- Write parameters, sources and capacity specifications into the existing platform and resource budget sections of `design.md`, and select the execution mode, number of cores, tiles, pipeline levels and buffer versions accordingly; distinguish the available capacity of each AIV from the entire physical capacity.
- When the query fails, the unconfirmed item is marked and processed according to Section 3.3. The actual machine value is not guessed. Exact utilization shall not be calculated when peak bandwidth or computing power is not matched to a specific SKU. Accessing the NPU follows the current device operating convention; use the `env -u ASCEND_RT_VISIBLE_DEVICES` prefix and `sandbox_permissions=require_escalated` in the Codex.

Use [A5 Backend Reference](references/a5-backend.md) to locate source code, target architecture, and compilation paths. Record the target, codegen and execution backend respectively, and check the execution domain, dtype/layout, data path, alignment and capacity constraints related to the current solution.

API assumptions from legacy 910 or GPU TileLang must not be directly inherited. The pure Vector solution only analyzes the actual storage level used; the Cube or fusion solution supplements the L1, L0A/B/C and inter-core transmission design based on the actual data path.

### 3.2 Information source priority

| Judgment content | Priority basis |
|---|---|
| The semantics and scope of this calculation | The user’s clear requirements, as well as the consistent public interfaces, formal specifications and golden; clarification in case of conflict, no override by oneself |
| API and backend capabilities | Python definitions, lowering/codegen, related tests, and applicable official information for the selected version |
| Implementation structure | Similar Ascend operators and tests in the current warehouse, indicating reusable parts and applicable conditions |
| Performance experience | There are already performance references and their verification status; as a candidate strategy, it does not replace the current version verification |

The report records the source code version, key files and functions that support the design, and cites local changes or measured evidence that affect the conclusion when necessary. The existence of source code and test code does not mean that the solution has been verified, and it does not claim that compilation or accuracy has passed.

### 3.3 Dealing with capability gaps

When the parameters, execution domain, dtype/layout or porting combination of key APIs are unclear, follow "Python API → lowering/codegen → related tests". If an unsupported combination is found, priority will be given to finding well-founded alternatives; when the backend must be expanded, it will be listed separately as an implementation dependency and will not be included in the scope of operator development by default.

When unknown capacity or unconfirmed capabilities affect the feasibility of the main plan, clarify the establishment conditions, affected designs and verification methods; provide evidence-based alternatives together, and do not write assumptions into facts.

## 4. Operator design criteria

### 4.1 Consistent semantics and clear scope

Maintain interface requirements such as input and output, return sequence, reduction axis, in-place update, and numerical processing sequence. Only the necessary implementation branches are selected for the specified case, and no general fallback is added for off-table input. The tail block and write conflicts involved in the specified case must be handled.

### 4.2 Decisions are specific and traceable

Each major choice should include "the plan, the reasons for the choice, and the key points for implementation." The report needs to give initial parameters, index formulas, data dependencies and resource calculations, and avoid only making requirements such as "reasonable tiling" and "automatic synchronization". Subsequent models can adjust parameters, but core algorithms and data flows should not be re-determined.

### 4.3 Legal resources and well-documented reuse

Peak occupancy is calculated for each AIC/AIV and each actual storage level, including physical layout, padding, alignment, multi-version buffers, simultaneous survival intermediate volume, back-end implicit allocation and necessary margin. Storage reuse needs to have a life cycle basis. Possible reuse by the compiler is not considered realized, nor is it the goal to fill the UB.

### 4.4 Numerical accuracy and correct scheduling

Explicit loading, calculation, reduce/accumulate, and store dtypes for each stage, along with associated cast, round, saturation, scaling, epsilon, and initialization locations. Prioritize the use of well-founded automatic scheduling, and explain production and consumption dependencies, buffer reuse conditions and output partitions; introduce manual synchronization when necessary.

### 4.5 Reasonable performance design

Prioritize the reuse of applicable existing structures to avoid obvious duplication and unnecessary serialization. The fusion solution needs to take into account the intermediate data capacity and parallelism; the separation solution needs to take into account workspace and additional handling. Actual rearrangement, padding or type conversion on the host side must be factored into the implementation and cost. No performance search is performed during the design phase, and speedup ratios are not promised in the absence of actual measurements.

## 5. Workflow

### Phase 1: Requirements analysis and operator feature analysis

1. Identify the semantic parameters required by the operator and pre-check the necessary input according to Sections 2.1 to 2.3. Do not use the complete set of public fields as the only criterion. In case of failure, the guided clarification in Section 2.4 must be performed and paused, and implementation of the characterization or subsequent stages cannot be continued.
2. After passing the pre-check, organize the calculation definition, input and output, attributes and specified cases, and establish a unified case ID; check the migration interface and the original reference implementation to determine golden, accuracy requirements and behaviors that must be maintained.
3. Identify the main calculation type, continuous axis, reduction axis, data reuse opportunities and parameters that need to be fixed at compile time. Determine the focus of subsequent design according to the characteristics of the operator:

| Operator type | Analysis focus |
|---|---|
| Element-by-element/broadcast | Continuous access direction, broadcast parameter index, input and output range corresponding to tile, duplicate data cache |
| Reduction / Norm / Softmax | Reduction range, single row capacity, accumulation accuracy, block result merging and final output timing |
| GEMM | M/N/K blocking, logic and on-chip layout, L0C initialization and accumulation, output post-processing |
| Cube/Vector fusion | AIC/AIV division of labor, intermediate result transmission, AIV output partition and buffer life cycle |
| Transposition/index class | Input-output index relationship, memory access continuity, block transfer or thread index, repeated write processing |

### Phase 2: Reference implementation retrieval and API verification

Entry conditions: The requirement completeness pre-check for Phase 1 has passed. If new user requirements are missing or semantic conflicts are found at any subsequent stage, return to Section 2.4 to pause and clarify; if technical capabilities are uncertain, still check according to Section 3.3.

1. First query the target machine hardware parameters according to Section 3.1, then locate similar implementations from [A5 backend reference] (references/a5-backend.md), and check their shape, dtype, layout, number of cores, and integer divisibility conditions.
2. Identify reusable algorithms or scheduling structures, as well as the parts that must be adjusted this time. When there is no complete similarity operator, components such as transfer, reduction, and GEMM are retrieved separately, and irrelevant examples are not required to be cited.
3. Verify key APIs according to Section 3 and retain parameter usage, input and output buffers, restrictions and sources. Ordinary arithmetic does not require item-by-item transcription of API tables.
4. Check the performance reference on demand without introducing its complete testing or tuning process.

### Phase 3: Implementation plan design

#### 3.1 Algorithm, execution mode and kernel division

Break down the formula into its main steps and determine single-kernel, fused, or multi-kernel solutions. Matrix calculations are mainly undertaken by AIC/Cube, and the vector part is undertaken by AIV. Within AIV, `SimdVF`, `SimtVF` or a combination can be selected.

Give a recommended solution, and list alternatives and usage conditions when there are clear trade-offs. When merging, explain the reduced handling and intermediate data capacity; when separating, explain the kernel calling sequence, intermediate result location and workspace size. Block reduction requires clear merging methods, state retention and initialization of partial results, and cannot be described as just "final reduction".

#### 3.2 Tiling, task mapping and loop structure

Specify a case for each class that gives an initial tile value or an executable selection formula and states:

- The output range of a single task, the total number of tasks, and the index mapping of task ID to batch/row/column.
- Core number selection, assignment of tasks to cores, and the conditions and corresponding case IDs of each implementation branch.
- Outer task loop, inner calculation loop, pipeline position, initial level and buffer version.

When using `Persistent`, `Pipelined` or a normal loop, use a structure that is compatible with the target backend. For small cases, it is necessary to check whether the number of tasks and loop length support the selected number of cores and pipeline depth. Cases with shared parameters can combine descriptions.

#### 3.3 Memory layout, data flow and resource budget

List the actual buffer usage, physical shape, dtype, storage location, version number and life cycle, and calculate peak occupancy according to Section 4.3. When the version dimension has been included in the shape, the version number is not multiplied repeatedly; the low-order packing type is calculated based on the actual storage bytes.

Use data flow tables or diagrams to correspond to calculation steps and buffers, describing read sources, write targets, execution units and key APIs. When reusing storage, indicate the last used location of old data; when there is no basis for reuse, the capacity budget is calculated as no reuse. [A5 Backend Reference](references/a5-backend.md) provides resource calculation and AIV partitioning examples.

#### 3.4 Numerical values, synchronization and boundary processing

- **Numerical calculation**: Provide the dtype, reduction initial value, clearing timing, accumulation sequence and output conversion of the main intermediate quantities; explain the corresponding relationship with golden.
- **Synchronization dependencies**: clarify the conditions for moving in, calculating, moving out and reusing. Distinguish between same-core pipeline, AIC/AIV collaboration and cross-kernel dependencies; list synchronization points and APIs during manual control.
- **Write Safe**: Check whether the output of task/core overlaps. When multiple tasks update the same result, clearly merge or atomic operations, initialization responsibilities and numerical impacts.
- **Tail block processing**: Give priority to the legal blocks that divide the specified case; if there are still tail blocks, clarify the valid range, fill value, calculate mask and output clipping. When the reduction reads padding, the padding value must not change the effective result.

#### 3.5 Interface integration and computing skeleton

Provides the Python calling interface, kernel builder main parameters, input and output mapping, workspace initialization responsibilities and recommended code locations. To distinguish between pure metadata view operations and actual data copying, the core calculations are completed by the TileLang kernel.

Provide short pseudocode when necessary, concatenating task indexing, buffer allocation and initialization, loops, key calculations, dependencies and writeback. Confirmed APIs use real names and key parameters; abstract steps are marked as pseudocode and do not construct unverified executable interfaces.

### Phase 4: Accuracy verification design of specified case

Clarify input generation, interface calling under test, golden calling and output-by-output determination for all specified cases. Checks shape, dtype, stride, and in-place update results when necessary; does not filter out specified cases due to blocking or alignment conditions.

For accuracy, user or existing test standards are preferred, and indicator definitions, comparison directions, specific thresholds and sources are recorded. Integer counts, indexes, and Boolean results usually compare exactly; floating point values ​​are determined by operator conventions. The basis for stating recommended values ​​when no established standards exist. You must not relax thresholds, reduce golden precision, or adjust golden based on kernel output.

Directly define comparison objects, methods and passing conditions. When boundary checking involves correctness, the checking location and expected behavior are explained, and the troubleshooting process is not described; element-by-element numerical comparison and byte-by-byte comparison should be distinguished according to operator semantics.

### Phase 5: Document generation and self-inspection

1. Completely read the [design.md template] (assets/design-template.md) and fill in the text using the text as the skeleton; retain the format of the first-level titles, the text, numbers and order of all second- and third-level titles, as well as the table column names and order. Only replaces operator names in titles, does not merge, rename, or delete chapters.
2. Delete the template comments and placeholder tips, and fill in the specific design. Simple operators can shorten the content; chapters that are not applicable retain their titles and explain why. Only the tables in Section 6.1 can be omitted if there are no technical assumptions or dependencies, and other tables should be retained. Add supplementary explanations to corresponding chapters, and add fourth-level headings if necessary.
3. Before output, compare the template item by item, self-check the title, chapter sequence, tables and fixed fields, fill in any omissions, and review the design content according to Section 6. The same self-check is performed when reusing or revising the report; if the format does not match, the valid content is retained and organized according to the template.

## 6. Quality Check

| Check items | Completion standards |
|---|---|
| Input completeness | Blocking deficiencies have been resolved, user replies are not replaced with default input or "to be supplemented"; design reports are not generated or overwritten when preflight fails |
| Template consistency | The template has been compared item by item, the title, chapter sequence, table structure and fixed fields are complete, and the prompt has been replaced |
| Requirement consistency | The calculation semantics are consistent with the input data, the specified cases have implementation paths, and the scope of the commitment is not clear |
| Solution completeness | The algorithm, kernel partitioning, API mapping, index formula and initial parameters are specific and can directly guide coding |
| Resource feasibility | Relevant level peak occupancy, capacity sources, implicit allocation, multi-version and reuse basis are complete |
| Data flow correctness | Handling scope, initialization, synchronization dependencies, output partitioning and necessary tail block processing are consistent with each other |
| Verify executability | All the inputs, golden, comparison rules and thresholds of the specified case can be used to write tests without mixing with the execution progress |
| Accuracy of evidence | Key conclusions have sources, technical assumptions and establishment conditions are clear, and there are no unsubstantiated verification conclusions |
| Implementation guidance | Clear interfaces, fixed requirements, adjustable parameters and design change conditions |

Both structure and content must meet the requirements, and length is not the basis for completion. Parameter adjustment cannot replace the core plan, and the reference path cannot replace the conclusion of this design; the plan in the report must not be described as an executed result.

## 7. Exception handling and design revision

| Scene | Processing |
|---|---|
| Missing requirements or semantic conflicts | Return to Section 2.4 to guide the user to supplement and pause the design; re-pre-inspect after a valid reply, and resume after passing |
| Insufficient API or capacity basis | Explain the conditions, impact and verification method of the plan according to Section 3.3 |
| Target combination is not supported | Evaluate evidence-based alternatives; list dependencies separately if backend must be extended |
| Revision of implementation or generalized test feedback requirements | Read old designs, failed cases, implementation and verification results after debugging, distinguish design defects from implementation errors, and only revise the affected content |

When subsequent generalization tests pass directly, there is no need to rewrite the plan. If the repair only makes the code conform to the original design, there is no need to change the correct solution; if debugging changes the algorithm, chunking, data flow, or applicable conditions, update the design based on the revised implementation and verification results. The retest failed case after repair is the same as the original specified case. Passing the test only proves the verified case and does not mean that any input is supported.

Changing the block requires checking the task mapping and tail block processing; adjusting the pipeline requires recalculating the version number and memory; changing the execution mode requires rechecking the API and synchronization.

Revision notes only record design changes and reasons, citing verification evidence when necessary, and do not copy debugging logs or remaining task lists. Fixes may not change requirements, support coverage, or accuracy standards. Changes in core algorithms, data flows, key APIs, or scope need to be reported simultaneously; parameters such as tiles, number of cores, and pipelines are allowed to be adjusted on the premise of meeting established requirements.

## 8. Output and completion report

### 8.1 File location

The design report is generated or updated only after the requirements completeness pre-check is passed. While waiting for user additions, only clarification questions will be returned and the `design.md` to be added will not be output.

Output directory priority: user-specified directory → existing operator working directory → `<project root>/operators/<op_name>/design.md`. If there is an existing design file, the original file will be used instead of creating a copy with different capitalization and capitalization.

### 8.2 Completion report

The following information is returned:

- Design document paths.
- Summary of recommended algorithms, execution modes and chunking schemes.
- Specify the design scope and accuracy verification method of the case.
- Technical assumptions or dependencies (if any) affecting implementation.

When it is necessary to explain the execution of this round, state it truthfully in the completed reply and do not write the design text. The user only requires the design to be output and then end it; when the complete development task has been authorized and the necessary requirements have been clarified, the caller can continue the implementation without the need for new design approval. Development authorization does not replace clarification of missing input.

## 9. Resource Index

| File | Usage and read timing |
|---|---|
| [references/a5-backend.md](references/a5-backend.md) | Phase 2 to check the source code and A5 constraints; Phase 3 to refer to algorithm cases, partition formulas and memory calculations on demand |
| [assets/design-template.md](assets/design-template.md) | Phase 5 generates design.md, including the implementation plan and the accuracy verification method of the specified case |
