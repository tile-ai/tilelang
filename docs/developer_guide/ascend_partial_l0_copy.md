# Ascend：显式 copy stride 与部分 L0 装载

## 前端合约

`tilelang.ascend.language.copy` 增加可选的 `stride` 参数，表示 L0 矩阵的
pitch，单位为 M/N 方向的 **16 元素分形数**，不是字节或分形间隙。

| 路径 | `stride` 控制 | 与普通 GEMM 配合时的值 |
|---|---|---|
| L1→L0A | 目标 pitch | `ceildiv(M, 16)` |
| L1→L0B | 目标 pitch，包括 MajorMN 转置装载 | `ceildiv(N, 16)` |
| L0C→UB/GM | 源 pitch | `ceildiv(M, 16)` |

不传 `stride` 保留既有 compact region 行为：一次 copy 搬运一个完整 GEMM
操作数时，从这次 region 推导 pitch。多次 copy 拼接一个操作数时，每次显式
传入完整 tile 的 pitch。非零矩阵目标起点要求显式 `stride`。
其他搬运路径不接受此参数。

编译器不查找 copy 对应哪个 GEMM，也不要求同一 buffer 的所有 GEMM 使用相同
pitch。调用者保证 copy 的 stride 与本次 GEMM 的有效 M/N 一致。
`stride` 可以是整数表达式，随循环或分支改变；在 copy 执行位置求值。
改变 pitch 不会重排旧数据，调用者必须重新准备相应输入和累加器。

MajorK/MajorMN 仍由既有 layout inference 决定，分配容量及 leading plane /
multibuffer slot 间距保持 allocation layout。显式 stride 仅改变 plane 内的寻址。
MX scale-factor handle 保持专用搬运合约，不在此参数的支持路径中。

## 普通装载与尾块

```python
# a 的 allocation 为 [64,64]，本次 GEMM 的有效 M=24。
T.copy(a_l1[:16, :], a[:16, :], stride=2)
T.copy(a_l1[16:32, :], a[16:32, :], stride=2)
T.gemm(a[:24, :], b, acc[:24, :], transpose_B=True, clear_accum=True)
# 完整输出 region 已经提供本次 compact pitch，无需标注。
T.copy(acc[:24, :], out[:24, :])
# 若只输出前 16 行，仍需按完整结果的 pitch 读取。
T.copy(acc[:16, :], first_rows, stride=2)
```

M=64 时，两次各搬 32 行的 copy 都传 `stride=4`。动态尾块可以传
`stride=T.ceildiv(effective_m, 16)`，region 选择包含 padding 的完整分形。

对 BF16 MajorK，逻辑坐标映射为 `[k//16, m//16, m%16, k%16]`。
分形线性位置为 `outer0 * stride + outer1`，每个分形占 512 B。
例如 allocation 为 [2,64,64]，选择 plane=1、stride=2、目标起点 [0,32]：

```text
plane base = 64 * 64 个元素
plane 内偏移 = (2 * 2 + 0) * 256 个元素
```

## V 的转置装载

```python
# v_l1 / v 的逻辑形状是 [K,N]，此处 K=64。
T.copy(v_l1[:32, :N_pad], v[:32, :N_pad], stride=T.ceildiv(N, 16))
T.copy(v_l1[32:64, :N_pad], v[32:64, :N_pad], stride=T.ceildiv(N, 16))
T.gemm(a, v[:, :N], acc[:, :N], clear_accum=True)
```

GEMM 为 V 推导 MajorMN，目标偏移按 MajorMN 和显式 stride 计算。
硬件转置位仍是 `user_transpose XOR major_mismatch`。
令 C0=256/dtype.bits，源 region 的分形步数为 (k_step,m_step)：

```text
普通实际写入 = (k_step, m_step)
转置实际写入 = (m_step*16/C0, k_step*C0/16)
```

copy 检查完整转置组、实际写入范围、目标起点和容量。
非零目标起点要求源和目标的起点、extent 均分形对齐；零起点保留既有 padded load。
实际写入必须位于声明的目标 region 内，且不能越过 stride 或 allocation。
空 copy guard 保持不变。

## Lowering 与同步

前端将 `stride` 存入 Ascend copy annotation；原有 IR 变量替换同时处理该表达式，
包括流水化迭代的变量替换。L1→L0 lowering 将它用于指令的 dst_stride，
并通过内部 access-pointer annotation 将它传给地址 remap。
LowerTileOp 保留完整 Major 映射和 allocation 间距，只修正 matrix outer0 的 stride。
L0C→UB/GM 将分形数乘以 16，得到 Fixpipe 的源行距。
GEMM lowering、布局推导不新增 pitch 传播或匹配过程。

AutoSchedule 开关不影响 stride 语义。开启时仍按 RAW/WAR/WAW 关系加同步：
每条指令的 L0 访问先表示为 plane 内的字节 Range，附带 `(repeat, stride)`，
分别表示重复次数和重复段起点间的字节距离。不同 pitch 也直接比较实际地址范围；
仅在无法确定寻址方式时退化为整个 plane。保留 leading plane / slot 分离与条件、
循环上下文。`MemoryAccessDetector` 在收集 Task 时逐次生成 `MemoryAccess`，
经 `TaskAccessInfo` 存入 `TaskNode`，依赖分析直接消费这些记录。
同一 buffer 的多个 copy 分别保留各自的 pitch 和访问范围，不先合并为一个矩形。
Task 内循环的变量范围随访问记录保存，并在比较两侧访问时分别实例化，保留
地址与动态 pitch 的关联。Task 克隆和变量替换同时处理这些记录；无需额外扫描
Task 或维护一份访问缓存。
显式 conflict hints 仍匹配原始逻辑 region，不由编译器自动生成。
这只是访问冲突判断，不给 copy 选择 GEMM 或推导 stride。
关闭 AutoSchedule 时，由调用者设置各流水线之间的同步。

多个 writer 允许顺序覆盖，不新增 writer 并集覆盖证明。
物理 lifetime 分析将 compact L0 storage 标记为 opaque，跳过跨 allocation 的重叠
复用证明，并保留所选 plane 的 may-access；其他普通 operand 保留 must-write 信息。

## 支持边界与回归

沿用 GEMM 和 L0C drain 的矩阵零起点限制。支持 L1→L0A/L0B 的分形对齐非零
目标起点、普通及显式/隐式转置、动态 M/N 尾块、K padding 和有效 K。
不插入 repack，不自动补齐非对齐 offset copy，不校验 copy 与 GEMM 的布局一致性。
FP4 packed-address 不在首批设备验证范围内。

主机回归覆盖默认 compact pitch、显式静态/动态 stride、普通/V 目标地址、
leading plane、部分 L0C drain、实际边界、不同 pitch 的同步及 allocation 复用。
NPU 回归覆盖两/三版本 ring 的部分装载、动态尾块、顺序覆盖，并覆盖关闭
AutoSchedule、手动同步、分支内不同 GEMM pitch 共用同一 buffer 的情形。
