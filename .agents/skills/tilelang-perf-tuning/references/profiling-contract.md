# Profiling 合约

本文件定义性能调优的统一设备侧采集口径。CSV 字段不确定时读取实际表头，不维护易过期的字段大全。

## 入口门禁

每个 B0、B* 和 Ci 都从其最新源码重新生成独立 profiling 入口。入口：

- 按锁定顺序内嵌全部 `CASES`；
- 只接受 `--device` 和可选 `--warm-up`、`--launch-count`；
- 在创建 NPU tensor 前选择物理设备；
- 调用入口内本地目标 kernel，或经明确校验的生产 wrapper；
- 不使用 host 计时作为 kernel 性能；
- 记录算子源码 SHA、case SHA、导入路径、实际参数、kernel 名和 launch 映射。

先直调同一入口，确认编译、运行和全部 case 顺序正确：

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<target> \
  python <entry.py> --device <physical_id>
```

旧入口不得测量新源码。禁止通过第三方评测 wrapper 间接采集基线并把其 host 开销混入 kernel 口径。

## `msprof op`

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<target> \
  msprof op --warm-up=10 --launch-count=<case_count> \
  --output=<new_dir> --kernel-name=<resolved_kernel> \
  python <entry.py> --device <physical_id>
```

要求 kernel 目录数、launch index、case 顺序和八类普通 CSV 完整。每次重试使用新输出目录，不覆盖或混用旧数据。

设备健康检查只做一次，依据状态和占用选择物理设备。不要因为其他设备存在 msprof 进程而全局等待。设备 ID 映射错误先移除继承的 `ASCEND_RT_VISIBLE_DEVICES` 并显式传物理 ID，不归因为 kernel 性能失败。

## 普通 `msprof` fallback

只有以下条件同时满足时使用：同一最新入口直调正常；失败明确位于 DBI/插桩/导出而非编译、执行、精度、核名或 case 映射；使用全新输出目录。

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<target> \
  msprof --output=<new_dir> \
  --application="python <entry.py> --device <physical_id> --warm-up 10 --launch-count 2" \
  --task-time=on --aic-mode=task-based --aic-metrics=PipeUtilization
```

入口对每个 case 先 warmup 再正式 launch。导出后按完整目标 kernel 名过滤，匹配行数必须等于 `case_count × (warmup + formal)`；按 case 顺序分组并丢弃每组 warmup。核名、行数、顺序或 SHA 不一致时停止，不猜映射。

fallback 摘要标明采集方式和缺失字段。最终比较优先使用同一种采集方式；一方 fallback 时另一方也用普通 msprof 重采。

## 归档与摘要

`msprof op` 结果使用：

```bash
python3 <skill_dir>/scripts/perf_summary.py \
  <launch_dir> <variant_case_dir> \
  --kernel-name <resolved_kernel> --round-name <round>
```

脚本归档原始 CSV、标准化文件名、校验目标 kernel 唯一并生成 `summary.txt`。先读摘要；只有异常、方向不明、接近噪声或需要精确归因时再读原始 CSV/timeline。

普通 msprof 不能使用要求八 CSV 的脚本；自行归档 task/op summary、过滤结果、case 映射、warmup/formal 次数和 SHA，并生成注明缺失指标的逐 case 摘要。

## 时间口径

- AIV-only：使用目标 AIV kernel 的设备侧时间。
- AIC-only：使用目标 AIC kernel 的设备侧时间。
- 混合 AIC/AIV：依据实际协同和重叠定义关键路径，不能简单相加所有 pipe ratio。
- 多 kernel：记录每个 case 的分支、核名和调用次数；整链验收覆盖公开调用所需的全部 launch、workspace、event 和同步。逐核采集只用于归因。
- 同一版本重复 launch 按预先声明的均值/中位数/稳定值处理，异常轮必须有设备频率、并发或工具证据才能剔除。

发现漏核、漏 launch 或错误聚合时废弃错误基线并重测，不能修改公共计时器制造提升。

## 归因、比较与噪声

摘要至少报告 kernel time、核间 min/avg/max、可用 pipe、频率和异常。结合源码计算逻辑 GM bytes、Vector/Cube 工作量和下界：

```text
effective_bandwidth = logical_bytes / kernel_time
effective_compute = logical_ops / kernel_time
```

有效带宽不是物理 HBM 利用率。一个 pipe ratio 不能直接映射到某条源码操作；用生成代码、流量、timeline 或隔离候选验证。流水重叠不能由 ratio 和、`num_stages` 或 buffer 数量单独证明。

基线与候选保持同一输入、dtype、输出语义、设备、频率、warmup、正式次数、并发和 kernel 聚合口径。每个候选对全部目标 case 采集。差异接近噪声时交替复测 `B* → Ci → B* → Ci`。pytest benchmark 或 host timer 不能支持最终晋升。

profiler 插桩/导出故障记为环境/工具阻塞，不能据此淘汰候选或宣布收敛。
