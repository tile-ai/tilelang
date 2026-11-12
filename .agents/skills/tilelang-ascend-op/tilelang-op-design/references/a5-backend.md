# A5 Backend Technical Reference

This article provides source code index, platform constraints and design examples, which can be consulted according to the current operator needs. When old skills, sample comments or experience descriptions are inconsistent with the selected version of the API or backend implementation, the current implementation shall prevail; conflicting requirements semantics will be clarified by the input processing process of the main skill.

## 1. Source code location and version information

- **Path Baseline**: All source code, example and test paths in this article are relative to the current warehouse root directory and do not depend on the warehouse directory name. The root directory passed in by the caller, or the repository containing `src/ascend/`, `src/backend/` and `examples/ascend/` located from the current workspace.
- **Framework and Examples**: For Ascend operator lowering, layout, scheduling and codegen, please check `src/ascend/` first, for backend public implementation, please check `src/backend/`; for Python API and compilation entry, please check `tilelang/`; for operator implementation examples, please check `examples/ascend/`, and for related framework testing, please check `testing/ascend/`.
- **Version and operating environment**: When generating a report, record the current warehouse version and local changes that affect the conclusion, and check whether the TileLang actually imported by Python is consistent with the source code. The existence of the source code directory does not mean that the running environment has been verified.
- **Architecture basis**: The source code basis for A5/950 corresponding to `dav-3510` is `tilelang/contrib/bisheng.py:get_npu_arch` and `testing/ascend/target/test_ascend_bisheng_arch.py`. Check with the current version when using it. This configuration basis is not equivalent to the physical device detection results.
- **Compilation path**: `ascend` and `pto` use different codegens. Check out `tilelang/ascend/target.py`, `tilelang/ascend/codegen.py`, `tilelang/ascend/execution_backend.py`, `src/ascend/codegen/` and `testing/ascend/target/test_tilelang_ascend_target.py`, and continue to `src/backend/` when you need to trace the common backend logic. Prioritizes the caller or project configuration; returns input preflight query if not specified, not recommended or selected by default. target, codegen and execution backend are recorded separately.

Use `rg --files` to locate files, `rg -n` to query functions or classes, and `git -C <root> rev-parse HEAD` to record versions. Relocation by symbol when path fails. Static source code retrieval eliminates the need to import modules that may initialize the device.

## 2. Source code and reference index

The following paths are relative to the current warehouse root directory. Before citing an example, you need to check its fixed shape, core number, dtype, and alignment conditions, and do not directly generalize to the current case.

| Design theme | Reference location |
|---|---|
| Operator semantics, golden and precision | User requirements and reference interface, `ref_program` and supporting tests of corresponding examples in `examples/ascend/`, related tests in `testing/ascend/`; track actual comparison functions, do not assume uniform use of atol/rtol |
| Element by element, quantification and fusion | `examples/ascend/example_simdvf_vecadd.py`, `examples/ascend/example_simtvf_vecadd.py`, `examples/ascend/example_simd vf_per_token_cast_to_fp8.py`, `examples/ascend/test_simdvf_vecadd.py`, `examples/ascend/test_per_token_cast_to_fp8.py` |
| Reduction and Norm | `examples/ascend/example_rmsnorm.py`, `examples/ascend/test_rmsnorm.py`, `testing/ascend/language/test_tilelang_ascend_reduce.py`; Check the example fixed core number and integer divisibility conditions |
| Backend public implementation | `src/backend/common/target_utils.cc`, `src/backend/common/codegen/`, `src/backend/common/op/`; combined with the call point of `src/ascend/` to confirm the actual implementation used by the current backend |
| GEMM and Cube/Vector data transfer | `examples/ascend/example_gemm.py`, `examples/ascend/example_gemm_mixedkernel.py`, `tilelang/contrib/ptodsl/gemm.py`, `src/ascend/op/gemm.cc` |
| Memory allocation, copy and dual_copy | `tilelang/language/copy_op.py`, `tilelang/ascend/language/allocate.py`, `tilelang/ascend/language/copy_op.py`, `src/ascend/op/copy.cc`, `testing/ascend/language/test_tilelang_ascend_copy_oob.py` |
| Kernel and execution domain | `tilelang/language/kernel.py:Kernel`, `tilelang/ascend/language/kernel.py:MixedKernel`, `tilelang/ascend/language/frame.py`, `tilelang/ascend/language/simd.py` |
| Automatic scheduling, synchronization and memory planning | `tilelang/ascend/pipeline.py`, `src/ascend/transform/`; query AutoSchedule, InsertSync, MergeUBAllocations and selected backend implementation |
| Core number and task configuration | `NUM_BLOCKS` / `N_CORES`, `Persistent` usage in `tilelang/ascend/language/kernel.py`, `tilelang/ascend/language/tile_schedule.py` and `examples/ascend/`; the actual available AIC/AIV core number follows the hardware query result of the main skill, combined with the task number selection, and does not regard the example constants as device specifications |

When you need a performance reference, load the Skill named `tilelang-performance-best-practices`; select the relevant operator family from the relative path `references/index.md` within the Skill, and check the verification status through `references/template_status.md`. Stop related references and reporting when the Skill is not installed or cannot be loaded by name, and is not read through the installation path. It only quotes experience as needed without performing its complete testing and tuning process; strategies that do not indicate applicable versions still need to be verified.

## 3. A5 design constraints

1. **Kernel and thread domain**: Currently Ascend’s `T.Kernel(N)` only accepts one-dimensional block grids and does not accept `threads=`. The thread domain is defined by `T.SimtVF(threads=N)`; the `sid` of `MixedKernel` is used for the AIV sub-core partition.
2. **Execution Mode**: For rule vector calculation, please refer to `SimdVF`, for thread-level indexing, branching or reduction, please refer to `SimtVF`, and for matrix multiplication, please refer to Cube. The API and synchronization scope of the two types of VF are different, and the CUDA warp/PTX usage cannot be directly migrated. Check the corresponding lowering for the currently selected operation.
3. **Data path**: Pure vector usually involves GM/UB/VF, GEMM involves L1, L0A/B/C, and the fusion solution may use L0C→UB, UB→L1. Confirm the legal path based on `copy.cc` and the selected codegen, and do not adopt the assumption that all data must pass through all memory layer by layer. L0A/B implicitly allocated by the helper still needs to be accounted for in the resource model.
4. **GEMM layout and numerical mode**: The current `PTOGemmL1Template` requires B to enter the corresponding transposed path in L1 with the `[N,K]` layout. Logical input B can be `[K,N]`, but a well-founded transfer transformation needs to be designed to maintain mathematical semantics. Check transpose, dtype, alignment, accumulation initialization and HF32/quantization mode according to actual lowering; the presence of parameters does not mean that all combinations are supported.
5. **dual_copy partitioning and conversion restrictions**: `tilelang/ascend/language/copy_op.py:dual_copy` determines the data partitioning of two AIVs based on the bipartite relationship of matrix dimensions. L0C→UB dual-purpose transfer does not support changing dtype at the same time; when cast is required, arrange it separately and clarify the output range of each AIV.
6. **Resource capacity**: The physical layout, DMA/SIMD access range, alignment, multi-version and temporary space are calculated separately according to the storage level and the core to which it belongs. The UB budget of each sub-core is not consolidated. The capacity basis should come from the target platform data, existing device attributes or corresponding version implementations. Do not directly apply the capacity values ​​of other platforms or infer the upper limit from a single example. When the capacity is unknown, clarify the required capacity of the plan and the basis for verification.
7. **Prerequisites for automatic scheduling**: `pipeline.py` contains passes such as automatic scheduling, synchronous insertion, and UB merging. You need to check the enabling conditions of the selected codegen. Combinations of `Persistent`, `Pipelined` and buffer versions should refer to the compatible examples and check whether the loop length can support the pipeline level. Dynamic boundaries and dependency processing capabilities are judged based on actual implementation.
8. **Boundary access**: The non-divisible parts are moved in and out according to the valid range; when calculating and reading the complete UB tile, the filling or mask of the remaining elements is specified. Processed according to the specified case design boundaries; passing the integer case does not mean that the tail block has been verified.

The design report only cites conclusions and sources related to the current operator. Ability that lacks evidence should state the establishment conditions and verification methods; static source code analysis must not be described as passing compilation, accuracy or performance verification.

## 4. Design depth example

The following examples illustrate how to extract design conclusions from source code without specifying common tiles, pipeline depth, or execution modes. The example is only based on source code analysis and does not add device testing.

### 4.1 Element-by-element fusion: task mapping, data flow and multi-version

`examples/ascend/example_simdvf_vecadd.py:vector_add` distributes the one-dimensional input to each core according to tiles, moves A/B into UB in the `T.Pipelined` loop, calculates `C = A * (A + B)` in `SimdVF` and writes it back. The corresponding `ref_program` and `examples/ascend/test_simdvf_vecadd.py` can be used to check the mathematical semantics and the accuracy comparison method of the examples.

Similar designs should clearly define the start and end locations of tiles, task allocation between cores, the loading range of each input, and the output writeback range. This example uses `begin = (iter * NUM_BLOCKS + bx) * TILE_ELEMS`, and the number of loops is calculated by integer division; when used in non-divisible cases, the tail block and coverage need to be designed separately and cannot be copied directly.

The example uses fp32 input and calculation, explicitly marking two versions for two sets of input and one set of output UB buffers. The 64-core, 8192-element tile and two-stage pipeline are example parameters and need to be reselected based on the actual number of available cores, specified cases, and capacity budget; operators involving broadcast or dtype conversion also need to be independently explained for indexing and precision processing.

### 4.2 Reduction: reduction range and intermediate states

`examples/ascend/example_rmsnorm.py:rms_norm_fwd` preloads the weights, processes the input line by line, accumulates the sum of squares in `SimtVF`, calculates the rstd, and finally writes the result and rstd. You can refer to the combination of `alloc_reducer`, initialization and `finalize_reducer`, but fixed core number and `batch // N_CORES` cannot be directly used as a general task allocation method.

The report needs to make it clear that the divisor of `rstd = rsqrt(sum(x*x) / D + eps)` is the logical length D, not the tile length after padding. When a line spans multiple tiles, it should indicate where the partial sums of squares are accumulated, when the full rstd is formed, and whether the output stage rereads the input. Blocked or online Softmax also needs to provide a method for merging partial maximum values ​​and exponential sums, and each block must not be independently normalized and then spliced ​​directly.

### 4.3 GEMM fusion: AIC/AIV division of labor and output partitioning

`examples/ascend/example_gemm_mixedkernel.py:gemm` gets `(bx, sid)` through `MixedKernel`, moves A/B into L1 in K loop and performs GEMM, initializing L0C with `clear_accum=(kt == 0)` on the first iteration. The output tile is then split between the two AIVs via `dual_copy`, each writing back the corresponding half-tile.

When pressing M to divide in two, the complete output tile is `[BM, BN]`, and the UB of each AIV is saved as `[BM/2, BN]`. The logical output range of the `sid`th AIV is:

```text
Row: [m_tile * BM + sid * (BM/2), m_tile * BM + (sid+1) * (BM/2))
Column: [n_tile * BN, n_tile * BN + BN)
```

This formula assumes that the BM is divisible and is currently a complete tile; the tail block requires additional clipping of the effective range. When post-processing requires cast, arrange it separately according to `dual_copy` restrictions. When using N partitioning or other schemes, the corresponding range should be re-derived.

### 4.4 Memory Budget: Physical Allocation and Peak Occupation

First determine the physical layout and padding of each buffer, and then calculate the number of bytes of a single version and the number of versions allocated at the same time. The low-order packed type is calculated based on actual storage bytes. The maximum value occupied by each storage tier at the same time is taken into account the back-end temporary space and necessary margin.

Assuming that each AIV's UB stores two inputs and one output, allocates 4096 bf16 elements, and each has two versions, then the occupancy of only these three sets of buffers is:

```text
UB_buffers = 3 × 4096 × 2 bytes × 2 versions = 49152 bytes = 48 KiB
```

This calculation is only an example of buffer occupancy and does not represent A5 capacity or general recommended parameters. Formal reporting is supplemented by actual alignment space, temporary volumes, and margins, and compared to sourced capacity. When the version dimension is already included in the physical shape, the number of versions must not be multiplied repeatedly; when shared storage is used, the basis for the end of the life cycle of the old data must be given.
