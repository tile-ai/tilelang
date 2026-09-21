---
name: tilelang-op-profiling
description: TileLang operator NPU performance collection and analysis, used to collect TileLang kernel performance data, locate performance bottlenecks and give optimization suggestions. Triggered when the user mentions "board performance", "operator performance test", "hardware performance verification", "NPU performance collection", "NPU profiling" and other scenarios during the TileLang operator development process.
---

# TileLang operator board performance collection and tuning

Collect TileLang kernel performance data on a real NPU, systematically interpret 8 CSV indicator files, determine whether the performance meets the standards, locate bottleneck types, and give actionable optimization suggestions.

## Backend

When `{tilelang_target}` is passed in from upstream, it is used as is; when called independently and not specified by the user, `PTO` or `AscendC` is asked first. Mapping: `PTO → pto`, `AscendC → ascend`. Prerun, msprof and retest must explicitly set the same `TILELANG_DEFAULT_TARGET`, disallowing default, fallback or mixed backends.

## Hardware evidence

Directly reuse `full_soc`, `npu_arch` and complete `evidence` under the same target device and configuration from the upstream. When it is called independently or the evidence is missing or incomplete, or the equipment or configuration changes, load the `npu-arch` Skill by name first, and the Skill will run its own detection script to obtain it again. Only accept evidence consistent with Ascend950PR/DT series with `npu_arch=3510`; stop and report if Skill is missing, detection fails, evidence conflicts, or platform is not supported.

Complete `full_soc` is used to match the number of cores, peak bandwidth and theoretical computing power of SKU changes; queryable items at runtime preferentially use the return value of the real machine. Specific specifications of current devices must not be extrapolated from `short_soc`, PR/DT product families, or typical SKUs, nor may unmatched model peaks be used in utilization calculations.

---

## Applicable scenarios

| Scene | Description |
|------|------|
| Performance acceptance after operator development is completed | Confirm that the operator reaches the expected performance level |
| Performance problem location | Pinpoint bottlenecks through CSV metrics |
| Optimization effect verification | Compare CSV data before and after optimization |
| Agent team testing phase | tester/developer call, automated performance analysis |

---

## Workflow

When using multiple launch operators, mixed AIC/AIV timing, or users requiring pytest for final acceptance, read [Whole Chain Timing and Acquisition] (references/kernel-chain-measurement.md) first. The following case kernel command is used for single-core entry; the multi-core link is adjusted according to the actual matching launch number and name set.

```
Step 1: Prepare and verify the TileLang profiling entry
    ↓
Step 2: msprof op collection indicators
    ├── Success → Continue
    └── Only DBI/instrumentation/export failure and direct adjustment is normal → Normal msprof fallback
    ↓
Step 3: Archive data + generate statistical summary
    ↓
Step 4: Read summary → judge against performance standards (read original CSV if necessary)
    ↓
Step 5: If the target is not met, consult the bottleneck optimization cheat sheet
    ↓
Step 6: Modify the code → Go back to Step 2 and collect again (the data is automatically archived for a new round)
```

### Step 1: Prepare and verify the TileLang profiling entry

Use the upstream workflow to generate an independent `profiling_file` from the latest TileLang operator source code of the current baseline or optimization solution. The file embeds all `CASES` in locked case order (Standard from `cases.csv`, Flash from user-confirmed parameters), accepts only `--device` and optionally `--warm-up`, `--launch-count`, and only calls the local target kernel within the file. The default `warm-up=0`, `launch-count=1` allows `msprof op` to be called once for each case in the same process; the normal `msprof` fallback is executed internally a specified number of times for each case by the entrance. Disable adding host timings as kernel performance results.

Before collecting, trigger TileLang JIT and confirm that the target kernel is actually running on the NPU:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<tilelang_target> \
  python <profiling_file> --device <physical_device_id>
```

Each round of baseline, each optimization plan and the source code modified plan must regenerate `profiling_file` from the latest source code in the corresponding directory; reusing old embedded implementations to measure new kernels is prohibited. If the entry directly imports the production wrapper, the import path, source code hash and call chain must also be reconfirmed. Next to the entry, record the operator source code SHA256, case list SHA256, default/explicit launch parameters and the actual parsed kernel name.

### Step 2: msprof op collection

```bash
CASE_COUNT=<selected_case_count>

# One process collects all cases
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<tilelang_target> \
  msprof op --warm-up=10 --launch-count="$CASE_COUNT" --output=<case_output> \
  --kernel-name=<expected_kernel> \
  python <profiling_file> --device <physical_device_id>
```

**Key Parameters**:

| Parameters | Description | When to use |
|------|------|---------|
| `--warm-up=N` | In-tool warmup | Use 10 to avoid DVFS affecting first run |
| `--launch-count=N` | Collect up to N matching kernel launches and will not execute them repeatedly for the application | For single core, it is the number of `CASES`; the entire chain is based on the actual total number of matching calls |
| `--output=<dir>` | Specify the output directory | Avoid results from being scattered |
| `--kernel-name=<name>` | Only collect the verified target TileLang kernel | Single-core entry must be used; the entire chain checks the filtering capability, core set and complete times according to the above reference |
| No need for `--soc-version` | Automatically detect hardware on the board | — |

Prioritize running `npu-smi info` only once before performing on-board collection. Select candidate physical devices based on health status, video memory and process information, and do not start the detection kernel on a device-by-device basis. If `npu-smi` is not installed, use `asys info -r=status -d=0` to check the device status instead. If neither command is available, skip the status pre-check and select the physical device by yourself. Do not block the collection just because of this. The msprof command must remove the inherited `ASCEND_RT_VISIBLE_DEVICES` and let the host adapter select the physical device directly; it is forbidden to drive the `msprof op` with `ASCEND_RT_VISIBLE_DEVICES=<non-zero physical ID>`. Correctness pytest can still use the visible device list according to its existing rules.

Do not wait globally just because there is any `msprof` process in the system; collection on other physical devices is not a blocking condition for this task. If `GetUserDevIdByDeviceId`, `ErrCode=107001` or `Failed to convert the driver device ID` appears in the log, it is determined as a device ID mapping error. Use a new output directory and try again after removing the visible device mapping and explicitly selecting the physical device. This must not be attributed to concurrent occupation.

**Output**: Generate `OPPROF_{timestamp}_XXX/` in the specified directory or the current directory; each case corresponds to `<resolved_kernel>/<launch_index>/`, and `launch_index` is mapped to `CASES` in order. Stop when integrity of directory count, index continuity, or general metrics 8 CSVs is not met.

### Step 2.5: Normal msprof fallback

`msprof op` is preferred. Fallback is only allowed if the following conditions are met at the same time:

1. The same latest entry is directly compiled and runs normally;
2. The failure point of `msprof op` is clearly located in DBI initialization, instrumentation or result export, rather than kernel compilation, device execution, precision, target name or case mapping;
3. Use a new output directory, keeping the device, input, case order and formal collection times unchanged.

Ordinary `msprof` does not have precise kernel-name filtering before collection, so the entry must be warm-up in a process for each case and then officially launched:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<tilelang_target> \
  msprof --output=<new_output> \
  --application="python <profiling_file> --device <physical_device_id> --warm-up 10 --launch-count 2" \
  --task-time=on --aic-mode=task-based --aic-metrics=PipeUtilization
```

After exporting, filter by the complete and actually parsed target kernel name in `mindstudio_profiler_output/task_time*.csv` (or the equivalent task/op summary in the current version). The required number of matching rows is exactly equal to:

```text
case_count * (warm_up + launch_count)
```

Matching lines are grouped in order of entry cases, the first `warm_up` lines of each group are discarded, and the last `launch_count` lines are used as the official result of the case. Stop when any of the line number, order, kernel name or source code/case SHA is inconsistent, and guess mapping is prohibited. The fallback result must indicate that the source is ordinary `msprof`; it is still the device side task/kernel time, not the total pytest time or host timing.

If ordinary `msprof` can export task-based pipe indicators, it can be used for trend and flow judgment of the same caliber; if it cannot export a certain `msprof op` exclusive CSV, it will be marked as missing as it is, and it cannot be forged or mixed with unequal fields. The final comparison between the baseline and the candidate will preferentially use the same acquisition method; if one party has to fallback, the other party will also use ordinary `msprof` to re-acquire.

Error classification is based on the location of failure instead of just looking at the last error code. For example, `507033` in the device initialization phase should first check the device health and mapping, and cannot be classified as a kernel failure; `507035` in `msprof op` can only trigger the above fallback when the direct adjustment is normal and the log proves that it is in the DBI/instrumentation stage. Always use the new output directory when retrying the same failure.

### Step 3: Archive data + generate statistical summary

```bash
# Find the latest OPPROF directory
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

#Archive case by case according to the mapping between launch_index and CASES
python3 {skill_path}/scripts/perf_summary.py \
  $OPPROF_DIR/<resolved_kernel>/<launch_index> <variant_output_dir>/<case_id> \
  --kernel-name <expected_kernel> --round-name <case_or_round_name>
```

`msprof op` results are automatically scripted:
1. Create the archive directory in `<variant_output_dir>/<case_id>/docs/perf/<case_or_round_name>/`; use auto-incrementing `round_NNN` when `--round-name` is omitted
2. Copy the 8 CSVs of the current launch and restore the timestamp suffix file name to the standard name
3. Verify that `<expected_kernel>` is uniquely matched in `OpBasicInfo.csv`; zero or multiple matches will fail immediately to avoid analyzing other kernels
4. Generate `summary.txt` statistical summary (only the target kernel is aggregated; AIC/AIV mixed kernels are counted separately, **no judgment** is made)

**Agent analysis process**:
1. **Read `summary.txt`** first — get a global overview (about 30 lines of compact text)
2. **Combined with `references/csv_fields_reference.md`** - Understand the meaning and threshold of each indicator
3. **Read the original CSV when an exception is found** — If there is an imbalance between cores, Read `PipeUtilization.csv` to view the core-by-core data
4. **According to Chapter 4.5 to calculate the actual amount of data transferred and the amount of calculation data** — Enumerate GM read and write tensors and Cube/Vector calculation amounts from the target TileLang source code (distinguish between the two) for calculation of computing power and bandwidth
5. **Combined `references/optimization_quickref.md` and the target TileLang source code** — Map the profiling bottleneck to the optimization direction of TileLang Tiling, data layout, pipeline and calculation implementation
6. **Output analysis files in Chinese** — Delivery must be in Chinese

> **Important**: `summary.txt` is only statistical aggregation and does not include analysis and judgment. All bottleneck determination and optimization suggestions are completed independently by the Agent based on the standards of Step 4 and Step 5. The original CSV file remains intact in the same directory and can be read at any time.

### Step 4: Performance standard determination

When the user specifies the case-by-case time consumption, bandwidth, or pytest acceptance criteria, the target is used to determine whether the standard is met. The following table and theoretical time consumption are used to locate the optimization space and cannot replace user acceptance; the empirical threshold in the table is not a single sufficient diagnostic condition.

#### 4.1 Overall judgment process

```
Read OpBasicInfo.csv → Get Task Duration and Block Dim
    ↓
Read PipeUtilization.csv → Find the unit with the highest proportion of each pipeline
    ↓
Calculate the actual amount of transferred data + the amount of actual operation data according to 4.5 (distinguish between Cube/Vector)
    ↓
Calculation theoretical time consumption (transportation volume/bandwidth or calculation volume/computing power)
    ↓
Compare actual time consumption vs theoretical time consumption
    ├── Gap <20% → Close to the estimated lower bound of the model used, also check the user target
    ├── Gap 20-50% → Evaluate the optimization space and check the bottleneck optimization table
    └── Gap >50% → Prioritize troubleshooting models, timing calibers and bottlenecks
```

#### 4.2 Each indicator meets the standard

| Indicators | Compliance conditions | Warning conditions | Serious problems |
|------|---------|---------|---------|
| **Inter-core load balancing** | Each core's `ai*_time(us)` difference <10% | difference 10-30% | difference >30% |
| **Block Dim** | Equal to the number of available cores (910B: 20~40 cores) | Far smaller than the number of available cores | Block Dim = 1 |
| **VEC ratio** | Matches the operator type (see 4.3) | VEC ratio >80% | VEC ratio >90% and no room for optimization |
| **MTE2 ratio** | <30% (computational operator) | 30-50% | >50% (transportation becomes a bottleneck) |
| **fixpipe_ratio** | <5% | 5-15% | >15% (check output conversion, atomic writeback and alignment, cannot directly assert misalignment) |
| **icache_miss_rate** | <5% | 5-15% | >15% (too large amount of code) |
| **bank conflict total ratio** | `aiv_vec_total_cflt_ratio` <5% | 5-15% | >15% |
| **L2 Cache total hit rate** | >80% | 50-80% | <50% |
| **Header Overhead** | <10% of total time spent | 10-30% | >30% |
| **DoubleBuffer Effect** | MTE2/VEC Overlap >30% | Overlap 10-30% | Overlap <5% |
| **Bandwidth utilization** | `bw_usage_rate` >60% | 30-60% | <30% |

#### 4.3 Expected ratio distribution of different operator types

| Operator type | Dominant pipeline | Expected ratio | Abnormal signal |
|---------|---------|-----------|---------|
| **Elementwise** (Add/Mul/Relu) | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| **Reduction** (ReduceSum/Max) | VEC | vec_ratio 40-70% | scalar_ratio >20% |
| **Activation** (Softmax/Gelu) | VEC | vec_ratio 60-85% | A large number of cast instructions |
| **MatMul** | CUBE | cube_ratio 40-70% | vec_ratio > cube_ratio |
| **Pure transportation** (Transpose/Concat) | MTE2/MTE3 | mte2+mte3 total >50% | VEC ratio >30% |

#### 4.4 Theoretical time-consuming calculation

> **Prefix**: Before calculating the theoretical time consumption, you must first calculate the **actual transferred data amount** and **actual operation data amount** of the operator according to Chapter 4.5. You cannot just apply the peak value formula. The amount of operation data must be distinguished between Cube and Vector.

**Transportation theoretical time consuming**:
**Before calculation, use the complete `full_soc` in the "Hardware Evidence" of this Skill to match the corresponding specification file, and do not select only by PR/DT product family. **
```
Theoretical time consumption (us) = actual amount of data transferred (Byte) / GM peak bandwidth
```

Confirmed specifications include: Ascend950PR 1.4 TB/s, Ascend950PR 1.6 TB/s, Ascend950DT 4 TB/s. It can be substituted only when the complete model has been matched to the corresponding gear by `npu-arch`; when the specific PR gear cannot be confirmed, the marked peak value is unknown, and the accurate bandwidth utilization is not calculated.

**Calculation theoretical time consuming**:
```
Theoretical time consumption (us) = actual calculation amount (FLOP) / theoretical computing power of the corresponding unit
```

Theoretical computing power must simultaneously match the complete model, actual core count, frequency, data type, and execution unit. DAV_3510 At 1.65 GHz, the peak value of Cube FP16/BF16 with 28/32/36 Cube Core is about 378/432/486 TFLOPS; `432 TFLOPS` is only applicable to the 32 Cube Core range and cannot be applied to all Ascend950PR/DT. The Vector instruction selects the corresponding peak value based on the actual Vector Core number, frequency and add/FMA instruction throughput; Cube, Vector and the total computing power of the entire chip must not be mixed. The specific value is obtained from the hardware parameter reference of the loaded `npu-arch` Skill; precise computing power utilization is not calculated in the absence of trusted SKU mapping.

#### 4.5 Calculation of handling volume and calculation volume (prerequisite for computing power/bandwidth analysis)

> The data on the board only has the actual time consumption. The handling volume and calculation amount must be calculated by ourselves from the operator source code. Based on this, the effective bandwidth, actual computing power and theoretical time consumption can be calculated. The amount of operation data must be distinguished between Cube and Vector.

##### 4.5.1 Calculation of actual transferred data volume (GM↔UB)

Enumerate the reading and writing of all GM tensors from the operator source code, and calculate the number of bytes one by one:

| direction | tensor | byte count |
|------|------|---------|
| Read | Each input | Total number of elements × Number of bytes per element |
| Write | Each output | Total number of elements × Number of bytes per element |

- The total number of elements = the product of all dimensions of the tensor (`prod(shape)`); the number of bytes per unit element is determined by dtype (such as fp32=4, bf16=2).
- Accumulate "total read volume/total write volume/total transfer volume".
- **Multiple input/multiple copies of data**: If a tensor exists in multiple copies in GM (such as multiple slices divided by dimensions, multiple copies of intermediate results), the handling amount must be included in all copies, not just one copy.
- Cross-validate with fields such as `read_main_memory_datas` / `GM→UB` of msprof `Memory.csv` to confirm the calculation caliber.

##### 4.5.2 Calculation of actual operation data volume (distinguish between Cube/Vector)

Determine the operator type and calculation amount from the source code:

- **Cube calculation amount**: `T.gemm` path → `2 × M × N × K` FLOP per tile; 0 without gemm.
- **Vector calculation amount**: `T.SimdVF` / `T.Parallel` operates element by element, and calculates logical FLOP according to "number of operations per element × number of elements".

Judgment basis (cross-validation with `ArithmeticUtilization.csv`):
- `aic_cube_fops > 0` / `aic_mac_ratio > 0` → with Cube calculation
- `aic_cube_fops = 0` → Pure Vector operator, Cube calculation amount = 0

##### 4.5.3 Computing power and bandwidth accounting

```
Effective bandwidth (GB/s) = total bytes transferred / measured steady-state time consumption
Bandwidth utilization = effective bandwidth / GM peak bandwidth
Actual computing power (FLOP/s) = actual calculation amount / measured steady-state time consumption
```

The ratio calculated by the source code logical bytes in the above formula is only called the "effective bandwidth ratio", not the physical HBM utilization; the latter requires the actual HBM traffic of the profiler and the precise device SKU peak. Shared cache, atomic read and write, and multiple consumption will make the two byte sizes different and cannot be mixed.

- Read-only/write-only are calculated according to the corresponding direction; when reading and writing occur at the same time, pay attention to MTE2/MTE3 bandwidth sharing.
- Computing-intensive operators use actual computing power vs. theoretical computing power to determine whether they are close to the computing upper limit.
- For small data volume operators, priority should be given to the proportion of theoretical transfer time and header overhead, rather than bandwidth utilization alone.

**Decision logic**:
- Actual MTE2 time consumption ≈ theoretical transportation time → MTE2 has reached the upper limit, and the optimization direction is pipeline orchestration
- Actual MTE2 time consumption >> Theoretical transportation time → MTE2 does not reach the upper limit, check the alignment and transportation granularity
- Actual VEC time consumption ≈ theoretical calculation time → VEC has reached the upper limit, and the optimization space is limited
- Task Duration ≈ longest running time → the running time is well arranged and other running times have been covered up

### Step 5: Bottleneck location and optimization

After confirming the bottleneck type, check `references/optimization_quickref.md` and develop specific optimization methods based on the target TileLang source code.

**Quick Search**:

| Bottleneck type | Determination conditions | Preferred optimization |
|---------|---------|---------|
| **VEC Bound** | `aiv_vec_ratio` highest | UB fusion, cast reduction, fusion command |
| **MTE2 Bound** | `ai*_mte2_ratio` Highest | Increases continuous handling granularity, checks alignment, Tile multiplexing and L2 hits |
| **CUBE Bound** | `aic_cube_ratio` highest | Adjust Tile shape to improve L0/L1 data reuse |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | Reduce dynamic branches, scalar loops and runtime index calculations |
| **Imbalance between cores** | The difference in time consumption of each core is >10% | Adjust the Tiling segmentation strategy |
| **Bank Conflict** | `vec_bank_cflt_ratio` >5% | Adjust TileLang layout, stride or padding |
| **Head overhead is large** | Head overhead accounts for >30% | Reduce the number of cores and dynamic branches, and simplify the kernel startup path |
| **DoubleBuffer does not take effect** | MTE2/VEC has no overlap | Adjust pipeline stage, check handling and calculation overlap |
| **Pipeline bubble** | Multiple units are 30-50%, no dominance | Adjust pipeline stage, Tile segmentation and asynchronous iteration |

### Step 6: Verify optimization effect

After each optimization, rerun Step 2 + Step 3. When `--round-name` is omitted, the data is automatically archived as `round_NNN+1`; when explicitly named, it must be a new directory, and the script refuses to overwrite the old archive.

**Comparison method**:
```bash
# Compare two rounds of summaries
diff <variant_output_dir>/docs/perf/round_001/summary.txt <variant_output_dir>/docs/perf/round_002/summary.txt

# Or directly read two summary.txt for comparison and analysis
```

**Comparison points**:
1. Whether Task Duration decreases?
2. Is the ratio of the bottleneck unit improved?
3. Whether the balance between cores is improved (aiv_time min/max gap)
4. Whether to introduce new bottlenecks

---

## Data directory structure

### msprof output (temporary)

```
OPPROF_{timestamp}_XXX/<resolved_kernel>/
├── 0/                          # CASES[0]
│   ├── OpBasicInfo_<timestamp>.csv
│   ├── PipeUtilization_<timestamp>.csv
│ └── ... # Total 8 indicators CSV
├── 1/                          # CASES[1]
└── ...
```

### Archive directory (persistent)

```
<variant_output_dir>/docs/perf/
├── round_001/ # First round of collection (baseline)
│ ├── OpBasicInfo.csv # Full original CSV (copied from OPPROF)
│   ├── PipeUtilization.csv
│   ├── Memory.csv
│   ├── ResourceConflictRatio.csv
│   ├── L2Cache.csv
│   ├── ArithmeticUtilization.csv
│   ├── MemoryUB.csv
│   ├── MemoryL0.csv
│ └── summary.txt # Statistical summary (min/avg/max, excluding judgment)
├── round_002/ # Second round (comparison after optimization)
│   ├── *.csv
│   └── summary.txt
└── ... # auto-increment
```

Full field descriptions for each CSV file → `references/csv_fields_reference.md`

---

## Board vs simulation selection

| Dimensions | Upper board (msprof op) | Simulation (msprof op simulator) |
|------|-----------------|---------------------------|
| NPU required | Yes | No |
| Timing accuracy | Real hardware timing | Cycle-level model estimation |
| Output | 8 CSV files | CSV + trace.json |
| Instruction level flowchart | Need to add parameters or simulate separately | Default output |
| **Resource Conflict Data** | **Yes** (ResourceConflictRatio.csv) | None |
| **L2 Cache** | **True Hit Rate** | Estimate |
| DVFS impact | Yes (warm-up required) | None |
| Suitable stage | Performance acceptance, production tuning | Early development, instruction level debugging |

**Recommendation**: Use simulation to iterate quickly during the development phase, and use boards to confirm real performance during the acceptance phase.

---

## Notes

1. **warm-up is required**: The first run is affected by DVFS and takes a long time. Always use `--warm-up=10`
2. **Frequency check**: Read the `Current Freq` and `Rated Freq` of `OpBasicInfo.csv`. If Current < Rated, it means the chip is not running at full frequency.
3. **MTE2/MTE3 bandwidth sharing**: When reading and writing GM at the same time, the total bandwidth is shared, and the theoretical time consumption should be calculated according to `(MTE2 handling volume + MTE3 handling volume) / GM bandwidth`
4. **Small data volume scenario**: When the data volume is small, the overhead ratio will be very high. This is not necessarily an operator problem, but a lack of data volume.
5. **Multi-core same address access**: Multiple cores reading the same 512B address range at the same time will be serialized, causing MTE2 to take abnormal time.

---

## Reference resources

| Document | Content | When to access |
|------|------|---------|
| `references/csv_fields_reference.md` | Complete field definitions and thresholds of 8 CSV files | Step 4 When analyzing, you need to understand the specific field meanings |
| `references/optimization_quickref.md` | profiling bottleneck to optimization mapping of TileLang Tiling, buffer, pipeline, layout and calculation implementation | Step 5 After locating the bottleneck, formulate a specific TileLang modification plan |
| `scripts/perf_summary.py` | Statistical summary generation + CSV archiving | Step 3 Automatic archiving and summary generation |

> The sum of the pipeline ratios under the same core and the same time denominator can be used as a clue to overlap. We cannot only rely on about 100% to assert complete serialization, nor can we regard 130% as a hard threshold. Combine event/version rotation of generated code with available timeline validation; the AIC/AIV ratio does not add directly. The ideal overlap estimate is only an upper bound on candidate gains, still subject to dependency, shared bandwidth, and synchronization constraints.
