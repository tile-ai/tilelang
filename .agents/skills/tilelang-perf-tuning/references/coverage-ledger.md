# 搜索覆盖账本

`optimization-search-coverage.json` 防止候选池只记录已想到的方法，却遗漏由 case 结构直接触发的高收益路线。机器校验只证明记录一致；仍需结合源码和逐 case 下界做结构审计。

## 创建时机与文件

首次修改源码前，从 `coverage-example.json` 创建 schema v3 记录；完整 RESULT 参考 `coverage-final-example.json`。示例值必须替换为当前事实。

记录包含三个 append-only 集合：

- `cases`：逐 case 结构、基线、目标和当前最佳；
- `obligations`：结构事实触发、停止前必须裁决的义务；
- `candidate_events`：候选 `CREATED` 与 `RESULT`。

每个 case 至少记录 outer/inner 独立任务、可用核、task waves、payload、kernel time、搬运/计算下界、主要 pipe、source window、用户目标和 best metrics。没有数值目标时 `performance_target` 为 `null`，不得把基线或示例阈值冒充用户目标。

## 通用义务触发器

1. outer 不足一波且 inner chunk 独立：`inner_parallel_flatten`；
2. 存在相邻工作、连续边界或显著固定成本：`work_granularity_search`；
3. 热路径逐元素/逐 lane 离散访问且窗口可界定：`physical_dataflow_routes`；
4. 同一 case 同时需要粒度和路线裁决：`granularity_dataflow_interaction`；
5. 每核迭代充足且 MTE/Compute 都可重叠：`pipeline_realization`。

触发器建立义务，不保证方法更快。最多四个 active candidates 不能删除或关闭义务。

工作粒度义务逐 case 记录 `FULL_CONTIGUOUS_UNIT`、`SIMD`、`DMA`、`CAPACITY`、`PARALLEL` 的来源值。物理路线至少裁决 `SCALAR_OR_SIMT`、`MEMORY_INDEXED_VECTOR`、`CONTIGUOUS_LOAD_REGISTER_REORDER`、`MATERIALIZED_TRANSFORM` 中与当前瓶颈相关且可实施的组合，并区分输出 writeback。

## 候选事件

`CREATED` 固定六轴 identity、逐 case 粒度、物理 route、pipeline facts、适用 case、假设和收益上界。创建下一个候选前追加同 identity 和适用 case 的 `RESULT`。

RESULT 状态使用 `PROMOTED`、`REJECTED_WITH_EVIDENCE`、`INAPPLICABLE_WITH_EVIDENCE` 或 `BLOCKED_BY_ENVIRONMENT`。实际运行修改路径的 case 使用 `TESTED_PATH` 并给出 `kernel_time_us` 和 profiling artifact。未运行、未激活或只过精度的 case 不能伪装成性能覆盖。

候选编译/执行/精度失败且健康 B0 对照正常时，可用结构化 rejection evidence 归因给候选。设备、profiler 或外部环境失效时只能记 blocked，并写恢复条件；义务保持 blocked，最终门禁拒绝收敛。

## 关闭证据

候选失败只否决精确六轴组合。关闭整个方向必须提供 `hard_infeasibility`，且 `exact_scope_only=true`：

- `CAPACITY`：完整 footprint，且证明不能继续 streaming/chunking；
- `LOWERING`：最小复现、实际查询 symbol 和编译/lowering 产物；
- `SEMANTIC`：无法保持公开语义的证明；
- `STRICT_COST_DOMINANCE`：同一有效 payload 下量化指令、访存、转换、物化和同步成本。

实现困难、源码较长、一个相邻组合失败或某次无收益都不是关闭证据。

## 校验与停止

每次创建下一候选前运行普通校验：

```bash
python3 <skill_dir>/scripts/validate_optimization_search_coverage.py <record>
```

普通校验输出 `VALID_RECORD`，允许 `PENDING`，不代表可以停止。停止前运行 `<record> --final`：全部有目标且达到时输出 `TARGET_MET`；全部无数值目标、结构闭环且有逐 case bounds 时输出 `CONVERGED_WITHOUT_NUMERIC_TARGET`；未达目标输出 `SEARCH_INCOMPLETE`。

只有全部义务、逐 case disposition、route/boundary/interaction plan、流水链和未达标 case 下界闭环时，才运行：

```bash
python3 <skill_dir>/scripts/validate_optimization_search_coverage.py \
  <record> --final --allow-unmet-convergence
```

成功输出 `UNMET_BUT_CONVERGED`。任何 pending、blocked、未运行候选或文字性推断都会失败。
