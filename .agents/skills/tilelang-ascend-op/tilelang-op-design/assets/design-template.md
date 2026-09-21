# <Operator name>: A5 TileLang operator design

<!-- Only filled in after the requirement pre-check is passed. Keep the title format, all second-level and third-level titles and their order, table column names and their order; replace the operator name, delete this comment and placeholder tip. Chapters may not be merged, renamed or deleted. Inapplicable chapters retain their titles and explain the reasons. Only the table in this section can be omitted if there is no dependency on 6.1. Before output, self-check this template item by item and review the design content. -->

- Target platform: Ascend 950/A5; Hardware evidence: <complete full_soc, npu_arch, detection source and evidence storage location or original text>
- Target backend: <PTO or AscendC>
- Resource specifications: <Actual number of AIC/AIV cores, involved on-chip capacity; record L2, Memory, peak bandwidth and theoretical computing power as needed and their specific SKU sources, unconfirmed items are clearly marked>
- Source code environment: <Current warehouse root directory, version, local changes affecting the plan>
- Compilation configuration: <target, codegen, execution backend; unconfirmed items are clearly marked>

## 1. Operator definition and support range

### 1.1 Computing semantics and interface specifications

<Give mathematical formulas or equivalent reference definitions, and clarify input and output, attributes and return order. Special layout, storage aliasing, in-place updates, and numerical processing requirements are accounted for as needed. Mark the file path and function name when there are specifications or golden. >

<Record the confirmed operator semantic parameters, such as Reduce’s reduction operation, axis/dim, keepdim and output shape. Implementation parameters such as tiles, threads, and number of cores are designed in Section 3. >

| Parameters | shape / dtype / stride or layout | Meaning and constraints |
|---|---|---|
| <input, output or attribute> | <specific specification or reference case table> | <purpose, default value and usage restrictions> |

### 1.2 Specify case

This time only the cases in the table below are implemented, and input outside the table is not within the scope of this support. The expanded support scope in the migration reference is not automatically included in the delivery.

| case ID | Each input shape / dtype | Properties and layout | Expected output shape / dtype |
|---|---|---|---|
| C1 | <Complete input specifications> | <Specific values affecting implementation> | <All return values> |

<Distinguish between compile-time fixed parameters and run-time parameters. Note the continuity, alignment, initialization, and in-place update requirements relevant to the current implementation. When modifying an existing operator, explain how to maintain the existing capabilities of the original public interface. >

## 2. Algorithm and implementation architecture

### 2.1 Algorithm, execution mode and kernel division

<Explain the recommended algorithm, number of kernels, fusion strategy, and their applicability to the specified case. When there are clear trade-offs, alternatives and usage conditions are given. >

| Calculation phase | Subformulas and dependencies | kernel / execution unit | intermediate results |
|---|---|---|---|
| <stage> | <calculation content and preorder dependencies> | <such as SimdVF/SimtVF in AIV, or AIC/Cube> | <buffer name and purpose, or none> |

### 2.2 Numerical precision and initialization

<Explicit dtypes for inputs, intermediates, reduction/accumulation, and output, and where to cast, scale, epsilon, round, or saturate. Reduction and accumulation need to explain the initial value, clearing timing, accumulation sequence and cross-tile status. Block reduction gives a partial result merging method and explains the correspondence with golden. >

### 2.3 Calling interface and project integration

<Gives the Python calling interface, kernel builder main parameters, recommended file locations and calling relationships. Clarify workspace size, allocation and initialization responsibilities; omit inapplicable items. >

<Multi-kernel solution explains the calling sequence and input-output mapping. When there is data rearrangement, padding or type conversion, clarify the execution location and cost; distinguish between metadata views and actual data copying. >

## 3. Tiling, data flow and resource planning

### 3.1 Blocking and task mapping

| Applicable case ID | Initial tile/dispatch condition | task/core mapping | pipeline level and buffer version number |
|---|---|---|---|
| <ID> | <Numeric or executable selection formula, only covers the specified case> | <Output range responsible for each core> | <Loop and buffer configuration> |

<Gives the output range of a single task, the total number of tasks, the index formula from task ID to batch/row/column, and the basis for selecting the number of cores. Distinguish between the outer task loop and the inner calculation loop, check task coverage, repeated writing, and whether the number of tasks and loop length support the selected pipeline. >

### 3.2 Buffer layout and capacity calculation

| Buffer | Physical shape / dtype | Storage level and core | Version number / life cycle | Byte usage |
|---|---|---|---|---|
| <name> | <contains layout padding> | <such as UB for each AIV> | <write, read, multiplex periods> | <formulas and calculated values> |

<Calculate the number of bytes according to actual allocation to avoid omission or duplication in the version dimension. The low-order packed type is calculated in physical storage bytes. When using storage reuse, explain the basis for non-overlapping life cycles. >

| Storage level and its cores | Peak occupancy calculation formula and results | Available capacity and sources | Feasibility conclusion and conditions |
|---|---|---|---|
| <Actually involved UB/L1/L0, etc., calculated separately by core> | <Including temporary amount, padding and margin> | <Values with sources; if unknown, state capacity requirements and verification basis> | <Conclusion and necessary assumptions> |

<Only list the actual levels involved, including the temporary space implicitly allocated by the backend. Pure Vector does not need to calculate L0A/B/C occupancy. >

### 3.3 Data transfer and synchronization

| Data flow stage | Source buffer → Target buffer | Execution unit / key API | Data dependency and reuse conditions |
|---|---|---|---|
| <stage> | <specific buffer> | <execution domain and operation> | <before and after dependencies, automatic or manual synchronization method> |

<Describes how the main dependencies are ensured by the selected scheduling method, and lists synchronization points and APIs for manual control. AIC/AIV collaboration needs to clarify the scope of data processed by each; cross-kernel needs to clarify the readable timing of intermediate data. You must not just write "Compiler automatic synchronization". >

### 3.4 Boundary processing and calculation skeleton

<Specify case to give the divisibility relationship when all are divisible. When there is a tail block, the valid range, input padding, calculation mask, and output clipping are specified; reducing the padding value must not change the valid result. When multiple tasks update the same result, clear merge or atomic operations and initialization responsibilities. >

<Provide short pseudocode when necessary, concatenating task indexing, buffer allocation and initialization, loops, critical calculations, synchronization dependencies and writeback. Confirmed APIs use real names and key parameters, abstract steps are marked as pseudocode, and do not require direct compilation. >

## 4. API basis and performance analysis

### 4.1 Key API and reference implementation

| Design Decisions / API | Usage and Limitations | Source Files / Function or Class |
|---|---|---|
| <Key items affecting feasibility> | <Execution domain, dtype/layout, key parameters> | <Actually consulted paths and symbols> |

<Describes the reusable structure of the main reference operator, differences from the specified case, and necessary adjustments. New capabilities that depend on the backend are marked separately and are not described as currently supported. >

### 4.2 Performance considerations

<When there are performance requirements, record the comparison baseline, target, timing range and expected bottlenecks of the specified case. In the absence of actual measurement, it is marked as design expectation. If there are no performance requirements, write "No performance target specified" and do not set additional performance thresholds. >

## 5. Accuracy verification of specified case

### 5.1 Verification use cases and processes

Covers all specified cases in Section 1.2 without additionally extending use cases. When reusing existing tests, locate the corresponding case and do not run the complete test set by default.

| Specify case ID | Input generation method | Golden / Comparison rules |
|---|---|---|
| <ID of Section 1.2> | <distribution, range, seed, or generating function and parameters> | <cite definition below> |

- Golden: <Reference function location and parameter mapping, or short reference code; explains calculation dtype, output conversion and attribute processing. Created based on formulas or existing references, not adjusted based on kernel output>.
- Verification process: <Input preparation and initialization, interface to be tested and golden call, return value and in-place update result comparison, necessary shape/dtype/stride checks>.

### 5.2 Accuracy Standard

| Output / applicable case | Comparison indicators and definitions | Judgment conditions and specific thresholds | Standard sources |
|---|---|---|---|
| <Output name and case ID> | <Error formula, existing comparison function or exact comparison> | <Value and comparison direction, not replaced by "accuracy standard"> | <User requirements, existing tests, or well-founded design suggestions> |

<Different outputs can use different standards. For example, floating point results are judged according to the agreed error, and integer counts are judged according to the precise value. Complete test matrices may not be added to complete the table. >

## 6. Design constraints and implementation suggestions

### 6.1 Technical assumptions and dependencies (on demand)

<Only list the technical assumptions or external dependencies that affect the feasibility of the solution; if not, write "No additional technical assumptions or dependencies" and omit the following table, retaining the title of this section. Common compilation, running, and testing tasks are not included here. >

| Assumptions or dependencies | Impact on solutions | Verification methods or alternatives |
|---|---|---|
| <Specific conditions and basis for establishment> | <Affected steps and cases> | <Verification methods, judgment conditions or well-founded alternatives> |

### 6.2 Implementation suggestions and adjustable parameters

<Distinguish between requirements and tunable parameters that must be maintained. After adjusting tiles, core number, and pipeline depth, recalculate resources and check task mapping and tail block processing; update the design synchronously when changing core algorithms, data flows, key APIs, or support scope. >

<Give the implementation sequence as needed, for example, first complete the main calculation path, and then complete the branches and pipelines required for the specified case. This sequence does not replace the final solution, nor does it reduce the specified case, nor does it create a new iteration plan. >

### 6.3 Revision description (only filled in when revising)

<For first time design, write “initial design”. When revising, explain the design changes and reasons based on the revised implementation, cite verification evidence when necessary, and do not copy the debugging process or task status. >
