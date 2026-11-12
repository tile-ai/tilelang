# Ascend AI Core Exception Dump 设计

本文描述社区 TileLang 仓库中 AI Core Exception Dump 的实现与维护约束。
用户操作步骤见 [Ascend Exception Dump 使用说明](ascend_exception_dump.md)。

## 1. 概述

### 1.1 问题背景与特性定位

在昇腾 NPU 上执行自定义算子时，非法内存访问或搬运错误等 AI Core 硬件异常
可能使 kernel 执行中断，并在后续同步时才向调用侧报告错误。
仅凭错误码和生成代码，通常难以判断故障任务实际使用了哪些输入数据、
访问范围是否正确，以及输出执行到了什么状态。

AI Core Exception Dump 在设备异常触发后，利用 CANN 异步回调获取故障任务的参数，
把其中可读取的 tensor buffer 保存到文件，再通过 Python 工具恢复为 NumPy 数组。
开发者可以据此核对输入、分析部分输出，并结合 kernel 源码定位问题。

### 1.2 核心能力

- **自动捕获**：预先启用配置并注册回调后，由 CANN 在设备异常时触发数据保存。
- **buffer 级数据保存**：根据 tensor 的地址、完整字节数和 dtype 保存数据，
  为分析实际数值提供依据；数据是否完整取决于异常后的内存可读性。
- **按 kernel 查找**：自定义 dump 以设备 kernel 名称为前缀，并带有时间戳，
  便于在多 kernel 或重复执行场景中查找对应记录。
- **编译配置控制**：通过 `"tl.ascend_exception_dump"` 字符串配置显式开启，默认保留原有启动 ABI。
- **端到端解析**：封装 dump 文件查找、CANN `msaicerr` 调用和 `.bin` / `.npy` 数据读取。

特性保存的是异常发生后仍可读取的 buffer 内容，并非 kernel 入口时的快照。
原地写入、部分执行或故障引起的内存变化可能已经改变输入和输出。

### 1.3 迁移来源与实现范围

本特性迁移自 `tilelang-ascend` 的 [PR #1462](https://github.com/tile-ai/tilelang-ascend/pull/1462)
和 [PR #1956](https://github.com/tile-ai/tilelang-ascend/pull/1956)。
迁移保留了“在启动参数中携带 tensor 元数据，由 CANN 异常回调保存数据”的方案，
并结合社区仓现有的 Ascend 编译流程和 TVM-FFI runtime 实现。
源仓的 Ascend 适配主要面向 A2/A3，本实现接入社区仓面向 A5 的 Ascend 框架，
设备代码生成和运行时启动均沿用社区仓的基础设施。

当前范围如下：

- 编译目标为 `target="ascend"`，仅支持 `execution_backend="tvm_ffi"`。
- 配置项为 `tl.ascend_exception_dump`，由 Ascend pass 流水线读取，默认关闭。
  使用字符串键传入 `pass_configs`，不在公共 `PassConfigKey` 中新增枚举。
- 启用时要求运行环境提供 CANN 9.2+ 的异常回调和 dump API。
- 复用社区仓的 host/device 拆分、参数打包、动态 shape 和缓存框架。
- PTO 后端不在本次实现范围内；在其他后端启用该配置会报错。

### 1.4 适用场景

| 场景 | 用途 |
| --- | --- |
| Kernel 调试 | 恢复可读取的 buffer，检查实际数值、内存范围和部分执行结果 |
| 自动化回归 | 在独立子进程中显式触发硬件异常，验证回调、dump 和数据恢复链路 |
| 运行环境排障 | 在编译和运行前启用配置，保留异常数据供后续分析 |

## 2. 技术原理

### 2.1 工作流程

启用特性并完成回调注册后，异常处理和数据恢复按以下流程进行：

```text
                  kernel 执行 ──→ AI Core 硬件异常
                                          │
                                          ▼
                              CANN 调用已注册的异常回调
                                          │
                                          ▼
                           aclrtGetArgsFromExceptionInfo()
                       获取 kernel args buffer（device 内存）
                                          │
                                          ▼
                             aclrtMemcpy()：拷贝到 Host
                                          │
                                          ▼
                               按 8 字节步长搜索 MAGIC
                            定位并校验 ExceptionDumpInfo
                                          │
                                          ▼
                             构建 acldumpTensorInfo 数组
                      （tensor 地址、字节数、元素数、数据类型）
                                          │
                                          ▼
                     acldumpSaveExceptionInfo(kernel_name, ...)
                                          │
                                          ▼
                  <ASCEND_DUMP_PATH>/extra-info/data-dump/<device>/
                          <kernel_name>.custom.<timestamp>
                                          │
                                          ▼ 事后解析
                   msaicerr.py 解析 → per-tensor .bin / .npy 文件
                                          │
                                          ▼
                 parse_exception_dump() → tensor 列表（NumPy 数组）
```

设备侧不需要主动执行 dump 指令。设备代码只增加参数声明，
元数据的构建、回调注册和文件保存均由主机侧完成。

### 2.2 CANN 异常回调机制

运行时在启用了本特性的 kernel 首次启动前，通过
`aclrtExceptionInfoCallbackRegister` 注册进程内共享的异常回调。
设备执行失败后，CANN 异步调用该回调，并提供对应任务的异常信息。

回调通过 `aclrtGetArgsFromExceptionInfo` 获取设备侧 kernel 参数缓冲区的地址和长度，
再用 `aclrtMemcpy` 将其拷回 Host。只有完成这一步，才能在主机侧安全地识别和解析元数据。
回调注册失败会阻止本次启动；回调内部的解析或保存失败不会替代应用收到的原始设备错误。

### 2.3 启动参数中的元数据与 MAGIC 定位

异常参数缓冲区含有指针和标量，但指针本身不能告诉回调 buffer 的大小与数据类型。
实现因此在编译时扩展设备函数签名，并在每次启动时追加一份按值传递的
`ExceptionDumpInfo`，记录 MAGIC、版本、kernel 名称，以及各 buffer 的设备地址、
字节数、展平元素数和 ACL dtype。

对于动态 shape，编译阶段从 host buffer map 构建元素数表达式，
通过隐藏的 `uint64` 参数传递实际大小。TVM-FFI 据此构造共享的主机/设备记录。

下图展示元数据在启动参数中的位置，以及回调识别记录后获得的信息：

```text
kernel args buffer（从 Device 拷回 Host 后）
低地址 ───────────────────────────────────────────────────────────→ 高地址
+------------------+--------------------+----------+------------------------+
| 原始指针/标量    | 隐藏 numel 参数    | 对齐填充 | ExceptionDumpInfo      |
+------------------+--------------------+----------+------------------------+

从参数缓冲区起点扫描：offset 0 → 8 → 16 → ... → MAGIC 候选
                                                  │
                                                  ▼
                                    校验版本、长度、数量与名称
                                                  │
                                                  ▼
                            ExceptionDumpInfo（末尾按值嵌入的记录）
                              ├─ magic
                              ├─ version / count / kernel_name
                              └─ tensors[i]
                                   ├─ address ──→ 设备上的 tensor 数据
                                   ├─ size        字节数
                                   ├─ elements    展平元素数
                                   └─ dtype       ACL 数据类型
```

图中的 `ExceptionDumpInfo` 是参数缓冲区内的结构体内容，按 8 字节边界放置；
其中 `tensors[i].address` 才是 tensor 的设备地址。
原始参数与隐藏元素数参数内部也可能含有对齐填充。
例如 128 个 float32 元素的描述为 `size=512`、`elements=128`，
回调无需从原始指针位置推测 tensor 的大小。

回调拷回参数后，按 8 字节步长搜索 MAGIC，再验证版本、记录长度、tensor 数量和名称。
这样可以复用同一解码器处理不同 kernel 的参数布局，并忽略未携带有效记录的异常任务。
共享 ABI 和具体参数传递方式分别见第 4～6 节。

### 2.4 Dump 生成与数据恢复

回调过滤 NULL 和空 buffer，将有效记录转换为 CANN `acldumpTensorInfo` 数组，
调用 `acldumpSaveExceptionInfo(kernel_name, "tilelang", ...)` 保存数据。
原始 dump 由 CANN 组织为一个自定义二进制文件，使用 `msaicerr.py` 解析后，
得到每个 tensor 对应的 `.bin` 或 `.npy` 文件。

#### 2.4.1 目录与文件名

下图展示默认解析目录下的文件组织；`<dump_name>` 表示原始 dump 的完整文件名：

```text
<ASCEND_DUMP_PATH>/
├── extra-info/
│   └── data-dump/
│       └── <device>/
│           └── <dump_name>                    ← CANN 生成的原始 dump
│                      │
│                      └── msaicerr.py 解析 ────────────┐
│                                                      ▼
└── parsed/                                   ← 默认解析结果目录
    └── exception-dump-<随机后缀>/              ← 每次解析独立创建
        ├── <dump_name>.input.0.float32.npy    ← 第 0 个有效 buffer
        └── <dump_name>.input.1.float32.npy    ← 第 1 个有效 buffer

原始 dump 文件名：
    <kernel_name>.custom.<timestamp>
     └─────┬────┘ └─┬──┘ └────┬────┘
           │        │         └── CANN 生成的数字时间戳
           │        └──────────── 自定义 dump 标识
           └───────────────────── 生成的设备 kernel 名称

解析后的单 tensor 文件名：
    <dump_name> . <type> . <index> . <dtype> . <ext>
        │           │         │          │        └── bin 或 npy
        │           │         │          └─────────── 数据类型，例如 float32
        │           │         └────────────────────── 同类型 tensor 的序号
        │           └──────────────────────────────── input / output / workspace
        └──────────────────────────────────────────── 原始 dump 的完整文件名
```

例如，真实故障验证生成了
`aic_fault_e2e_kernel.custom.20261106034442600`，
对应解析文件之一为
`aic_fault_e2e_kernel.custom.20261106034442600.input.0.float32.npy`。

当前回调把有效 buffer 统一标记为 `input`，其中也可能包含输出 buffer；
序号按过滤 NULL 和空 buffer 后的顺序重新排列，不一定等于原函数参数索引。
`output_dir` 可以修改解析结果目录，每次调用仍会创建独立子目录并保留结果。

#### 2.4.2 原始 dump 二进制格式

这里的“原始 bin”指 `<kernel_name>.custom.<timestamp>` 这一原始二进制 dump，
它通常没有 `.bin` 扩展名。以下格式依据本次使用的 **CANN 9.3.0** 解析实现，
并已用真实异常生成的文件核验。

原始 dump 由“头长度 + 序列化元数据头 + tensor 数据区”组成：

```text
文件偏移     0                 8                    8 + H
             │                 │                      │
             ▼                 ▼                      ▼
             ┌─────────────────┬──────────────────────┬──────────────────────┐
             │ header_length H │ Protobuf 元数据头    │ tensor 数据区        │
             │ uint64，8 字节  │ H 字节               │ 按各条目的 size 读取 │
             └─────────────────┴──────────────────────┴──────────────────────┘
                                                       │
                                                       ▼
             input[0] → input[1] → ... → output[...] → buffer[...] → space[...]
             └─ 每个条目连续占用其 size 字节；对应类别为空时不产生数据段
```

长度字段 `H` 表示后续 Protobuf 元数据头的字节数，**不包含长度字段自身的 8 字节**。
当前验证环境使用小端序；CANN 工具以 `struct.unpack("Q", ...)` 读取这一长度。
第一个 tensor 的数据从偏移 `8 + H` 开始，后续条目的起点由前面各条目的 `size` 累加得到。
这里是磁盘数据布局，tensor 数据段之间没有 TileLang kernel 参数 ABI 中的对齐槽位。

元数据头经 CANN 解析库解码后，包含以下主要信息：

| 字段 | 含义 |
| --- | --- |
| `version` | CANN dump 格式版本；本次样本为 `"2.0"` |
| `dump_time` | CANN 记录的生成时间 |
| `op_name` | dump 对应的算子/记录名称 |
| `attr` 中的 `user_tag` | 本实现调用保存 API 时传入的 `"tilelang"` |
| `input` / `output` / `buffer` / `space` | 按类别组织的 tensor 描述列表 |
| 每个描述项的 `data_type`、`format`、`shape`、`size` | 数据类型、格式、维度及对应数据段的字节数 |
| `dfx_message` 等 | CANN 附带的诊断信息 |

例如，真实样本的部分元数据解码为以下 JSON；JSON 是解析后的展示形式，
磁盘头部存储的是 Protobuf 二进制内容：

```json
{
  "version": "2.0",
  "op_name": "aic_fault_e2e_kernel.custom.20261106034442600",
  "attr": [{"name": "user_tag", "value": "tilelang"}],
  "input": [
    {"data_type": 1, "format": 2, "shape": {"dim": ["128"]}, "size": "512"},
    {"data_type": 1, "format": 2, "shape": {"dim": ["128"]}, "size": "512"}
  ]
}
```

本次样本的 `data_type=1` 在 CANN dump 类型映射中表示 float32；
启动元数据 `ExceptionDumpInfo` 使用的 ACL float32 编码则为 0，二者不应直接混用。
同样，磁盘头中的 `version="2.0"` 是 CANN dump 版本，
与 `ExceptionDumpInfo.version=1` 所标识的启动参数记录版本分别维护。

真实文件 `aic_fault_e2e_kernel.custom.20261106034442600` 的布局如下，
偏移区间采用左闭右开形式：

| 区域 | 字节偏移区间 | 长度 | 核验结果 |
| --- | --- | --- | --- |
| 头长度字段 | `[0, 8)` | 8 字节 | `b0 01 00 00 00 00 00 00`，即 `H=432` |
| Protobuf 元数据头 | `[8, 440)` | 432 字节 | 包含 2 个 input 描述和 `user_tag="tilelang"` 等信息 |
| `input[0]`，对应输入 x | `[440, 952)` | 512 字节 | 128 个 float32，数据 SHA-256 与期望输入一致 |
| `input[1]`，对应输入 y | `[952, 1464)` | 512 字节 | 128 个 float32，数据 SHA-256 与期望输入一致 |

本例总长度为 `8 + 432 + 512 + 512 = 1464` 字节。
`H=432` 是该样本的实测值，元数据内容或 tensor 数量变化时，头长度也可能变化。

`ExceptionDumpInfo` 是供回调从 kernel 参数中查找信息的内存记录；
CANN 根据回调提供的 `acldumpTensorInfo` 重新组织磁盘文件。
因此，原始 dump 的开头是长度字段 H，MAGIC 用于扫描异常参数缓冲区，
不用于定位磁盘 dump 的元数据头。

上述布局可在 CANN 的
`tools/msaicerr/ms_interface/dump_data_parser.py` 中核对：
`BigDumpDataParser._read_header_length` 读取长度并跳过头部，
`_parse_binary_to_json_data` 按类别和 `size` 顺序提取数据；
Protobuf 元数据由 `libascend_dump_parser.so` 的 `ParseDumpProtoToJson` 解码。
文件格式随 CANN 版本维护，解析应使用与运行环境匹配的工具。

#### 2.4.3 解析后的单 tensor 文件

这里的 `.bin` 和 `.npy` 都是 `msaicerr.py` 解析后生成的单 tensor 文件，
与前一节包含多个 tensor 和 CANN 元数据头的原始 dump 分别使用。
两种文件在内容上有以下区别：

```text
.bin  ┌───────────────────────────────────────────────┐
      │ tensor 原始数据字节，按文件名中的 dtype 读取  │
      └───────────────────────────────────────────────┘

.npy  ┌────────────────────────┬──────────────────────┐
      │ NumPy 文件头           │ tensor 数据字节      │
      │ dtype、shape、存储顺序 │                      │
      └────────────────────────┴──────────────────────┘
```

两种格式的用途与选择如下：

| 对比项 | `.bin`：原始数据交换 | `.npy`：带元信息的数组分析 |
| --- | --- | --- |
| 文件内容 | tensor 的原始数据字节，不携带 NumPy 文件头 | NumPy 文件头加 tensor 数据，记录 dtype、shape 和存储顺序 |
| 读取条件 | 按已知 dtype 和字节序读取；恢复多维时还需外部 shape | 可通过 `np.load` 从文件头读取 dtype（含字节序）和 shape |
| 主要用途 | 按字节核对数据、向 C/C++ 或设备侧复现程序提供原始 buffer | 在 Python 中做数值统计、数组比较、可视化和保存带类型的回归参考数据 |
| 典型场景 | 底层程序按已知字节数读入设备内存；排查字节偏移、位模式或数据传递问题 | 异常后的交互式分析；与 golden 数组比较；跨运行保存和加载参考数组 |
| 使用时关注 | dtype 或字节序选错时，字节仍可能被读出，但数值解释会出错 | 交给 NumPy 读取文件头，再使用数组数据；设备 buffer 需要的是数据区内容 |

对于相同 tensor，文件格式本身不提高或降低数值精度，区别在于如何封装和解释数据。
`.bin` 便于直接交换原始字节，`.npy` 便于把数据及其解释方式一起保存。
在本特性的常规 Python 排障和回归比对中，通常使用 `.npy` 更方便；
若下游复现程序要求裸数据 buffer，则可使用 `.bin`，并同时记录 dtype 和 shape。

读取同一个 float32 tensor 的两个文件时，可使用以下方式；路径需替换为实际解析文件：

```python
import numpy as np

# .bin：这里按当前小端 float32 数据解释，shape 来自已知的 tensor 信息。
from_bin = np.fromfile("tensor.input.0.float32.bin", dtype="<f4").reshape((128,))

# .npy：从 NumPy 文件头获得 dtype 和保存时的 shape。
from_npy = np.load("tensor.input.0.float32.npy", allow_pickle=False)
```

如果需要将 `.npy` 交给只接收裸字节的复现程序，可以先用
`np.load(..., allow_pickle=False)` 取得数组，再用数组的 `tofile(...)` 或 `tobytes()` 导出数据。
反向转换时，应先用正确 dtype 和 shape 读取 `.bin`，再通过 `np.save(...)` 保存，
不能通过修改扩展名完成格式转换。

CANN 9.3 的解析器会优先为能用 NumPy 表示的 dtype 生成 `.npy`，
无法表示时保留 `.bin`；实际输出也取决于 CANN 工具版本和已安装的 dtype 扩展。
TileLang 的 `parse_exception_dump` 同时处理这两类文件，并统一返回 `data` 数组，
一般无需调用侧手动分支选择读取函数。
对于 bfloat16，优先使用该封装完成 dtype 恢复；未支持的 raw dtype 会明确报错。

例如，128 个 float32 元素的数据区占 512 字节；`.npy` 文件还包含 NumPy 文件头。
当前 dump 元数据只记录展平元素数，因此 `.npy` 中的 shape 也对应展平后的维度，
原始多维 shape 需要调用侧结合已知信息恢复。

事后调用 `parse_exception_dump`，完成“查找对应 dump → 调用 `msaicerr.py` →
读取每个 tensor 的 `.bin` 或 `.npy` → 返回 NumPy 数组”的流程。
恢复后的数组可与故障前保存的参考数据比对，也可按已知 shape 重排后进行数值分析。

### 2.5 配置开关与兼容性

配置经 `pass_configs` 进入 TVM `PassContext`，由 Ascend pipeline 决定是否插桩，
并由设备 codegen、host 启动路径和 runtime 配合传递元数据。

默认关闭时，不添加隐藏元素数参数、共享结构体参数或生成代码中的专用头文件，
也不因本特性注册回调。原有 kernel 启动 ABI 保持不变。
开启后会增加参数传递和 host 侧元数据构建开销，并要求运行环境提供对应的 CANN API，
因此配置需要在编译前确定；运行环境中的 dump 路径与场景应在 ACL 初始化前设置。

## 3. 编译期：保留 buffer 的大小信息

### 3.1 插入位置

`AscendPassPipelineBody` 在 `SplitHostDevice` 之后、
`MakePackedAPI` 和 `LowerDeviceKernelLaunch` 之前调用
`annotate_exception_dump`。

此时既能看到 host 对设备函数的调用，也能从 host `PrimFunc.buffer_map`
获得 buffer 的 dtype、shape、strides 和元素偏移。
不能仅依赖设备函数体：`LowerAccessPtr` 等变换可能已将搬运操作降低成指针访问，
纯 DMA kernel 的函数体中也可能没有可用于恢复完整 shape 的 Buffer 节点。

### 3.2 隐藏元素数参数

插桩以 host 调用实参和 buffer 的 `data Var` 身份进行匹配，
为匹配到的 buffer 计算：

```text
numel = product(uint64(shape[i]))
```

每一维先转换为 `uint64`，再相乘，避免先做 `int32` 乘法再转换导致的溢出。
这些元素数表达式追加到 host 调用实参，设备函数则追加对应的
`__tl_dump_numel_i: uint64` 参数。

例如，两个 `(rows, 128)` 的 bfloat16 buffer 经拆分后可能得到：

```text
原设备签名：destination, source, rows
插桩后签名：destination, source, rows, numel_destination, numel_source
隐藏实参：  uint64(rows) * 128, uint64(rows) * 128
```

隐藏元素数属于普通设备参数，因此原有 host codegen 和启动 lowering 可以继续处理它们。
动态大小由每次调用时的实际 shape 求值，不在编译时固定。

### 3.3 函数属性

设备函数增加 `tl.ascend_exception_dump_tensors` 属性，每个条目包含四个整数：

| 字段 | 含义 |
| --- | --- |
| `pointer_arg_index` | buffer 指针在插桩后设备参数列表中的索引 |
| `numel_arg_index` | 对应隐藏元素数参数的索引 |
| `acl_dtype` | CANN `aclDataType` 编码 |
| `element_bits` | 每个存储元素的位宽；bool 按 8 位存储处理 |

上述例子的属性为 `[[0, 3, 27, 16], [1, 4, 27, 16]]`。
索引遵循拆分后的设备签名顺序，不能直接套用 Python 函数的参数顺序。
同一设备函数有多个调用点时，元数据布局必须一致，元素数实参可以不同。

插桩还验证连续布局、零元素偏移、dtype、tensor 数量和 kernel 名称长度。
不满足支持条件时在编译阶段报错。

## 4. 共享主机/设备 ABI

共享定义位于 `src/tl_templates/ascend/exception_dump.h`，仅依赖标准整数类型，
不依赖 CANN SDK 头文件。

### 4.1 单个 tensor 描述符

`ExceptionDumpTensor` 按 8 字节对齐，占 32 字节：

| 字段 | 偏移/字节 | 类型 | 含义 |
| --- | --- | --- | --- |
| `address` | 0 | `uint64_t` | 本次启动使用的原始设备数据地址 |
| `size` | 8 | `uint64_t` | buffer 字节数 |
| `elements` | 16 | `uint64_t` | 展平后的元素个数 |
| `dtype` | 24 | `int32_t` | CANN 数据类型编码 |
| `reserved` | 28 | `int32_t` | 保留，当前置零 |

`size` 和 `elements` 的语义必须区分。例如 128 个 float32 元素对应
`size=512`、`elements=128`，不能把字节数写入 CANN shape。

### 4.2 完整记录

`ExceptionDumpInfo` 同样按 8 字节对齐，占 656 字节：

| 字段 | 偏移/字节 | 大小/字节 | 含义 |
| --- | --- | --- | --- |
| `magic` | 0 | 8 | 固定值 `0x474e414c454c4954`，用于定位记录 |
| `version` | 8 | 4 | 当前为 1 |
| `count` | 12 | 4 | 有效描述符数量，上限为 16 |
| `kernel_name` | 16 | 128 | 以 NUL 结尾的设备 kernel 名称 |
| `tensors` | 144 | 512 | 16 个固定大小的 tensor 描述符 |

结构体采用固定宽度字段，并用 `static_assert` 约束大小和关键偏移。
两个启动路径均先清零结构体，再填写本次启动的数据。

设备 codegen 在插桩后的签名末尾追加按值传递的 `ExceptionDumpInfo`。
这里不能传主机描述符的指针：设备异常异步报告时，该主机栈对象可能已经失效，
也不属于 CANN 返回的完整设备启动参数记录。

```text
实际设备参数缓冲区：
[原始指针/标量及必要 padding][隐藏 numel 参数][8 字节对齐][ExceptionDumpInfo]
```

TVM-FFI 调用侧的 grid extent、动态 UB 大小等启动配置是另一组尾部实参，
由 `LaunchParamConfig` 提取，不作为这个字节缓冲区中的设备参数。

## 5. TVM-FFI 启动和缓存

### 5.1 元数据序列化

`ExtractFuncInfo` 将函数属性编码到 `FunctionInfo.launch_param_tags` 的尾部：

```text
[原有 grid/UB tags...,
 "tl.ascend_exception_dump.v1",
 "pointer_arg_index:numel_arg_index:acl_dtype:element_bits", ...]
```

标记之后的字符串只是 Ascend runtime 的描述信息，不代表新的 grid 参数。
当前 TVM 的 `arg_extra_tags` 是枚举数组，不能承载这个字符串格式。
复用已有字符串字段可保持 TVM 子模块和模块序列化格式不变。

`AscendModuleNode::SaveToBytes` 原有的 `[fmt][fmap][code]` 中已保存 FunctionInfo。
因此模块导出、加载和磁盘缓存恢复都可以保留这些描述信息。

### 5.2 参数打包

读取原始 FFI 参数数组之前，先校验设备参数与实际 launch 参数的总个数。
参数缺失或多余时立即报错，避免越界读取，也不会为错误调用注册回调或发起 kernel 启动。

`AscendModuleNode::GetFunction` 按版本标记分离普通 launch tags 与 dump tags。
普通 tags 交给 `LaunchParamConfig`；dump tags 用于建立 `AclArgPackPlan`。

打包计划验证字段格式、数量、参数索引和类型，并确定普通参数及结构体的偏移。
普通参数遵循现有 ACL ABI：标量至少按 4 字节对齐，指针和 uint64 按 8 字节对齐，
尾部结构体从 8 字节边界开始。

每次调用时，`PackExceptionDumpInfo` 从实际指针和隐藏元素数参数构建记录，
检查 `elements * element_bytes` 是否上溢，然后把结构体内容拷贝到启动缓冲区。
元数据不能只在函数首次调用时构建，因为动态 shape 和输入地址可能变化。
字节数检查不替代上游对 shape、总元素数和有效分配范围的验证。

开启特性的调用在实际启动前执行 `EnsureExceptionDumpCallback`，
随后通过现有 `aclrtLaunchKernelWithHostArgs` 启动路径执行。

### 5.3 开销与生命周期

若一个设备函数有 N 个被记录的 buffer，忽略对齐 padding 后，
新增设备参数占 `8 * N + 656` 字节，分别来自隐藏元素数和完整记录。
每次 host 调用还需要构建 O(N) 的描述信息。
TVM-FFI 的参数包因此会进入现有的大缓冲区动态分配分支；开启本特性有正常启动开销。
正常执行期间不为本特性复制 tensor 的数据内容，实际参数拷回和 tensor 文件保存发生在异常路径。
当前没有给出性能开销的测量结论。

描述符按每次启动构建，异常参数快照和 CANN 描述符数组均为该次回调的局部对象，
无需维护按 kernel 名称查找的可变 tensor 地址表。
跨 kernel 共享的是进程内的回调注册状态，而不是某一次启动的地址和 shape。

## 6. 执行后端限制与缓存恢复

本特性只支持 Ascend 的 TVM-FFI 执行路径。Cython 和 PTO 不支持开启
`tl.ascend_exception_dump`；关闭特性时，各执行后端沿用社区仓已有的行为。

统一缓存入口在读取内存或磁盘缓存前检查目标及执行后端，避免命中旧的 Cython dump
缓存而绕过限制。lowering 入口也做同样检查，覆盖直接调用 lowering 的场景。
这两处是公共框架中必要的配置边界检查，未增加通用插件机制或新的缓存版本字段。

host 分支、循环及表达式计算均由社区仓现有的 TVM host 编译流程处理。
本特性不再生成 Cython host wrapper，也不新增 host 控制流变换或公共表达式打印规则。
元数据只在实际调用设备函数时由原生参数打包器构造；未选中的分支和零次循环不发起设备启动。

磁盘缓存沿用既有 TVM-FFI 模块导出、加载和 FunctionInfo 序列化流程。
加载后，Ascend runtime 从 `launch_param_tags` 恢复 dump 元数据，
并在实际启动时调用 C++ `EnsureExceptionDumpCallback`，无需 Python 注册桥接入口。
共享记录的字段、656 字节大小及版本 1 保持不变。

PTO 已支持普通 TVM-FFI 执行，但不支持本特性的按值聚合记录。除配置入口检查外，
PTO codegen 也拒绝带 dump 属性的 IR，避免共享 FunctionInfo 为没有该参数的 PTO
设备函数错误追加 656 字节启动记录。关闭本特性时保持上游 PTO 行为。

缓存遵循上游多根目录规则：首根可写，其余根只读并按顺序查找。命中后直接加载
已导出的模块，不重写缓存条目。已有条目损坏时明确报错，不删除、不自动重编译，
也不跳过该条目去读取后续根目录；缺失条目才继续查找。

## 7. 异常回调

### 7.1 API 发现和注册

运行时需要以下接口：

| API | 职责 |
| --- | --- |
| `aclrtExceptionInfoCallbackRegister` | 注册异常回调，采用 CANN 9.2+ 接口 |
| `aclrtGetArgsFromExceptionInfo` | 获取故障任务的设备启动参数地址和长度 |
| `aclrtMemcpy` | 将设备启动参数拷回主机 |
| `acldumpSaveExceptionInfo` | 保存用户提供的 buffer 描述符及其数据 |

默认 stub 构建把这些符号作为可选成员加载；直接链接构建也在本特性使用时延迟解析。
关闭本特性时，不因缺少这些可选接口而阻断原有启动。
启用且接口缺失时，注册步骤报错，阻止缺少回调保护的启动。

注册使用 `std::call_once`。成功注册后复用已有回调；注册失败抛出异常，允许后续重试。
回调在进程内共享，可能收到未插桩 kernel 的异常通知，因此解码器必须能够忽略外来参数。

### 7.2 读取和识别参数

回调处理流程为：

1. 检查异常对象、设备参数地址和长度；长度必须足以容纳记录，且不超过 16 MiB。
2. 用 `aclrtMemcpy` 将参数缓冲区从设备复制到主机。
3. 以 8 字节步长查找 MAGIC；读取使用 `memcpy`，不要求主机缓冲区本身对齐。
4. 验证记录完整性、版本、`count` 范围及 kernel 名称的 NUL 结尾。
   对准备保存的非空条目，还检查 dtype、保留字段、元素数到字节数的溢出，
   以及 `size == elements × 元素存储字节数`；NULL 和空条目按第 7.3 节过滤。
5. 若候选不合法，继续查找，避免用户标量偶然等于 MAGIC 就被误认为有效记录。

回调是 `noexcept`，并捕获内部异常。解析或保存失败不能向 CANN 报告线程抛出异常，
也不能替代应用原本收到的设备错误。

### 7.3 构建 CANN 描述符

地址为 NULL、字节数为零或元素数为零的条目被过滤，其余条目压紧为 CANN 描述符数组：

| CANN 字段 | 当前取值 |
| --- | --- |
| `type` | `ACL_DUMP_TENSOR_INPUT` |
| `tensorSize` | `size`，以字节为单位 |
| `format` | ND |
| `dataType` | 元数据中的 ACL dtype |
| `tensorAddr` | 原始设备数据地址 |
| `addrType` | `ACL_DUMP_ADDR_RAW` |
| `placement` | `ACL_DUMP_PLACEMENT_DEVICE` |
| `shapeNum` / `originShapeNum` | 均为 1 |
| `shape[0]` / `originShape[0]` | 均为 `elements` |

当前没有在编译期区分输入、输出或 workspace，因而有效 buffer 均标记为 INPUT，
其中可能包含输出 buffer。压紧后的 dump 索引也不一定等于原函数的参数索引。
如果没有有效条目，不调用保存接口。

最后调用 `acldumpSaveExceptionInfo(kernel_name, "tilelang", ...)`。
文件由 CANN 写入，典型路径如下：

```text
<ASCEND_DUMP_PATH>/extra-info/data-dump/<device>/<kernel_name>.custom.<timestamp>
```

`DumpExceptionHostArgs` 内部返回值约定为：0 表示保存成功，CANN 错误码表示保存失败；
-1 表示无可识别记录，-2 表示没有可保存 tensor，-3 表示 API 缺失或内部处理失败。

## 8. 数据解析

`parse_exception_dump` 负责查找、调用 CANN 工具和恢复 NumPy 数组：

1. 默认等待 2 秒，为异步回调写文件留出时间；这不是文件完成写入的严格保证。
2. 在所有设备目录中查找指定 kernel 的 `.custom.<数字时间戳>` 文件，按时间戳数值选择最新文件。
3. 从 `ASCEND_HOME_PATH/tools/msaicerr/msaicerr.py` 找到解析器，用当前 Python 解释器调用，超时为 60 秒。
4. 每次新建独立解析子目录，并把它设为解析器工作目录，保存解析结果和工具日志，避免污染调用目录。
5. 只接收与本次 dump 文件名匹配的 `.bin` / `.npy` tensor 文件，按类型和数字索引排序。

`.bin` 按文件名中的 dtype 读取，不为未知类型静默假设 float32。
`.npy` 使用 `allow_pickle=False`，保留已存储的形状和数据类型。
bfloat16 使用 `ml-dtypes`；对于 NumPy 将其保存为两字节不透明类型的情况，以 view 恢复原始位模式。

返回列表中每个条目包含 `data`、`type`、`index`、`dtype` 和 `file`。
当前回调只记录展平后的元素数，原始多维 shape 和参数名称不在元数据中；
调用侧应按已知 shape 和参数对应关系解释数据。
解析工具不会删除已有的 dump 或解析文件。

## 9. 限制和约束说明

本节说明启用特性时需要满足的条件，以及 dump 数据能够表达和恢复的范围。

### 9.1 适用范围

- 当前仅支持 `target="ascend"` 和 `execution_backend="tvm_ffi"`。
- PTO 和其他编译后端不在当前支持范围内。
- 设备 kernel 需要由社区仓的 Ascend 编译流程生成。外部源码 kernel 无法保证与追加后的参数 ABI 一致，当前不支持开启本特性。
- 特性默认关闭，需要在编译时通过 `tl.ascend_exception_dump` 显式启用。
- 在 Cython 或 PTO 执行路径开启该配置会明确报错，且不会读取旧缓存或发起设备启动。

### 9.2 Tensor 数据类型与布局约束

支持的 buffer 元素类型为 float16/32/64、bfloat16、bool，
以及有符号和无符号的 8/16/32/64 位整数。
fp8、fp4、向量元素类型和其他未加入 dtype 映射的类型当前不支持。

参与记录的 buffer 必须满足以下条件：

| 项目 | 约束要求 |
| --- | --- |
| 存储布局 | 必须能够证明为连续布局；显式 strides 只要能够证明连续，也可以接受 |
| 元素偏移 | `elem_offset` 必须能够证明为零，避免从错误的起点读取数据 |
| 大小信息 | host 侧必须能够从 buffer 信息构造元素数表达式；动态 shape 通过隐藏的 uint64 参数传递 |
| 内存有效性 | 实际设备地址和数据范围需要有效；字节数上溢检查不能替代地址和分配范围校验 |

数据类型、连续性或元素偏移等编译期条件不满足时，编译器会明确报错。

### 9.3 数量、名称与 ABI 约束

| 项目 | 约束要求 |
| --- | --- |
| 记录数量 | 每个生成的设备 kernel 最多记录 16 个 tensor buffer |
| Kernel 名称 | 设备 kernel 名称必须少于 128 个 UTF-8 字节，为固定字段中的 NUL 结尾预留空间 |
| 共享记录 | `ExceptionDumpInfo` 当前固定为 656 字节，按 8 字节对齐，采用按值传递 |
| 启动一致性 | 设备签名、host 侧参数打包和回调解码必须遵循同一版本的共享 ABI |

开启配置后的 kernel 会增加隐藏元素数和共享结构体参数，
其启动方式必须与生成该 kernel 时的配置一致。
缓存恢复也需要保留相应元数据和回调注册逻辑，不能混用开启与关闭特性的启动 ABI。

### 9.4 运行环境与配置约束

运行环境需要 CANN 9.2+，并提供异常回调注册、故障参数获取、设备到 Host 拷贝和自定义 dump 保存接口。
缺少所需接口或回调注册失败时，启用本特性的启动会报错。

`ASCEND_DUMP_PATH` 和 `ASCEND_DUMP_SCENE=aic_err_brief_dump`
必须在导入 torch/torch_npu 或初始化 ACL 之前设置。
解析时还需要通过 `ASCEND_HOME_PATH` 定位与运行环境匹配的 `msaicerr.py`。
正常启动会承担额外的参数传递和元数据构建开销，性能影响需要结合实际算子评估。

### 9.5 数据记录与恢复能力约束

- 仅保存地址非 NULL、字节数和元素数均非零的 buffer；不满足条件的条目会被过滤。
  没有有效记录或可保存条目时，不生成对应 tensor 数据。
- 非零地址仍可能无效或在异常后不可读。数据恢复依赖 buffer 在回调处理期间仍可访问，
  已释放、已复用或损坏的内存不能保证恢复。
- 当前有效 buffer 统一标记为 `input`，其中可能包含输出 buffer；
  dump 序号按过滤后的条目顺序排列，不一定等于原函数参数索引。
- 元数据只保存展平元素数，不保留原始多维 shape 和参数名称，
  数据对应关系和原始 shape 需要调用侧结合算子信息解释。
- dump 保存的是异常发生后仍可读取的数据，不是 kernel 入口时的事务性快照。
  原地写入或部分执行可能已经修改 buffer，输出也可能只写入一部分。

因此，不能承诺每次硬件异常都能完整恢复所有参数，
也不能把恢复出的数组直接视为原始输入或完整输出。

### 9.6 文件解析约束

应使用与运行环境匹配的 CANN 工具解析原始 dump，
再读取生成的 `.bin` 或 `.npy` 单 tensor 文件。
`.bin` 需要按正确 dtype 和字节序解释；当前不支持的 raw dtype 不会被自动猜测为 float32。

回调写文件是异步过程，解析器的默认等待时间不代表文件一定已经写完。
调用侧需要确认目标 dump 已可供读取，并选择正确的设备 kernel 名称。
当 dump 缺失、工具失败或超时、dtype 不支持，或没有匹配 tensor 文件时，
Python 解析器会明确报错。

## 10. 使用与验证

### 10.1 启用方式

在导入 torch/torch_npu 或初始化 ACL 之前设置环境变量，随后编译已有的 `prim_func`：

```python
import os

os.environ["ASCEND_DUMP_PATH"] = "/tmp/tilelang_exception_dump"
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"

import tilelang

kernel = tilelang.compile(
    prim_func,
    target="ascend",
    execution_backend="tvm_ffi",
    pass_configs={"tl.ascend_exception_dump": True},
)
```

捕获设备执行或同步异常后，用生成的设备 kernel 名称调用
`parse_exception_dump`。它不一定与 Python 函数名相同。

### 10.2 自动化测试

| 测试文件（相对仓库根目录） | 覆盖内容 |
| --- | --- |
| `testing/ascend/transform/test_ascend_exception_dump.py` | 默认关闭、隐藏参数、动态 shape、连续 strides、偏移及 dtype 等限制 |
| `testing/ascend/runtime/test_exception_dump_callback.py` | 回调参数边界、记录过滤、CANN 描述符、注册重试与并发 |
| `testing/ascend/runtime/test_exception_dump_host_args.py` | fake ACL 的 LOCAL/GLOBAL 加载、混合宽度参数、grid/UB、溢出、元数据拒绝及模块导出/加载 |
| `testing/ascend/runtime/test_tvm_ffi_host_args.py` | TVM-FFI 正常执行、混合宽度标量、现有 host 控制流和动态 shape 磁盘缓存恢复 |
| `testing/ascend/runtime/test_exception_dump_parser.py` | bin/npy、bfloat16、时间戳、相对路径、目录隔离及错误处理 |
| `testing/ascend/runtime/test_exception_dump_hardware.py` | 独立子进程中的真实故障、生产回调保存及父进程精确数据恢复 |

真实硬件故障用例默认跳过，只通过以下命令显式开启：

```bash
TILELANG_TEST_ASCEND_HW_EXCEPTION=1 python -m pytest \
  testing/ascend/runtime/test_exception_dump_hardware.py -q -s
```

### 10.3 真实硬件验证

验证环境为 Ascend950DT_9574（A5）和 CANN 9.3.0。
正常测试检查混合宽度标量、动态 shape、分支/循环，以及已有 TVM-FFI 磁盘缓存恢复。

真实故障测试在独立子进程中先执行一次正常 kernel，再通过原生 TVM-FFI 设备入口传入
NULL 输出指针。生产异常回调自动保存 dump，父进程使用 CANN 解析器恢复 x/y，
核对逐元素值、完整字节和 SHA-256；随后在新进程做正常执行健康检查。
此前已撤回的 Cython 用例不计入当前支持范围和验证统计。

## 11. 主要代码与维护约束

| 代码位置 | 职责 |
| --- | --- |
| [`src/ascend/ir.cc`](../../src/ascend/ir.cc) | 注册 Ascend 专用 PassContext 配置项 |
| [`tilelang/engine/lower.py`](../../tilelang/engine/lower.py) | 编译目标范围校验 |
| [`tilelang/ascend/exception_dump.py`](../../tilelang/ascend/exception_dump.py) | 从 host 信息构建隐藏参数与设备属性 |
| [`tilelang/ascend/pipeline.py`](../../tilelang/ascend/pipeline.py) | 读取 Ascend 配置、默认关闭并选择插桩阶段 |
| [`src/ascend/codegen/codegen_ascend.cc`](../../src/ascend/codegen/codegen_ascend.cc) | 追加共享结构体参数和头文件 |
| [`src/ascend/codegen/rt_mod_ascend.cc`](../../src/ascend/codegen/rt_mod_ascend.cc) | 提取可序列化的函数信息 |
| [`src/ascend/runtime/ascend_module.cc`](../../src/ascend/runtime/ascend_module.cc) | TVM-FFI 参数打包、元数据恢复与原有 ACL 启动 |
| [`tilelang/cache/kernel_cache.py`](../../tilelang/cache/kernel_cache.py) | 在缓存读取前校验本特性的目标及执行后端 |
| [`src/tl_templates/ascend/exception_dump.h`](../../src/tl_templates/ascend/exception_dump.h) | 固定宽度、固定大小的共享 ABI |
| [`src/ascend/runtime/ascend_exception_dump.cc`](../../src/ascend/runtime/ascend_exception_dump.cc) | 注册、异常参数解码与 CANN 保存 |
| [`src/ascend/stubs/ascendcl.h`](../../src/ascend/stubs/ascendcl.h) / [`.cc`](../../src/ascend/stubs/ascendcl.cc) | CANN 描述符 ABI 及可选符号加载 |
| [`tilelang/ascend/exception_dump_bin.py`](../../tilelang/ascend/exception_dump_bin.py) | 文件选择、CANN 工具调用和数组恢复 |

维护时须保持三个约定一致：设备函数签名、TVM-FFI 启动路径写出的记录、回调解码器读取的记录。
修改结构体布局、tensor 上限或字段语义时，必须同时考虑 ABI 版本、
`tl.ascend_exception_dump.v1` 标记、旧缓存的处理策略和 TVM-FFI 启动、缓存恢复的回归验证。
扩展 dtype 时需要同时明确实际存储字节数、CANN 编码和 Python 解析方式。
扩展多维 shape 或输入/输出分类时，也应相应调整共享记录和缓存兼容方案。
