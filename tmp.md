# tilelang fat wheel (CUDA + ROCm) build — handoff

## 目标

让一份 `tilelang` wheel 同时支持 NV (CUDA) 和 AMD (ROCm/HIP) 两种 GPU。
当前 PyPI 上 `tilelang==0.1.9` 是 CUDA-only —— `libtilelang.so` 里完全没编 ROCm 后端
(`nm libtilelang.so | grep RegisterROCmCopy` 为空)，HIP target 走 lower 时
`tl::ResolveCopyImpl()` 找不到匹配实现，抛
`Check failed: (best_impl != nullptr): tl.copy requires a target-specific implementation`。

## 架构判断（已论证过，不用再纠结）

- tilelang 是 DSL / code generator，`libtilelang.so` 不参与设备执行
- `USE_CUDA` 和 `USE_ROCM` 在 CMake 里相互独立、无互斥
- 算子注册表 (`src/op/copy.cc:81-106` `CopyImplRegistry`) 是 vector，多 backend 可共存，dispatch 按 `match_target(target)`
- 已有 `TILELANG_USE_CUDA_STUBS` / `TILELANG_USE_HIP_STUBS` (默认 ON)，两套 stub 都用 dlopen，wheel 不硬依赖驱动 lib

→ 结论：`USE_CUDA=ON USE_ROCM=ON` 的 fat wheel 在原理上完全可行

## 已踩到的真坑

TVM 上游 `3rdparty/tvm/cmake/utils/FindROCM.cmake` 的 `find_rocm()` 宏 **必须** 在 `/opt/rocm`、`$ROCM_PATH` 或 `USE_ROCM=/path/to/rocm` 里找到 **真的 `libamdhip64.so`**，否则 `ROCM_FOUND=FALSE`，`cmake/modules/ROCM.cmake:32` 直接 FATAL：

```
CMake Error at 3rdparty/tvm/cmake/modules/ROCM.cmake:32 (message):
  Cannot find ROCM, USE_ROCM=ON
```

也就是 stub 设计目前只解决了 **runtime** 不依赖真 driver，没解决 **build time** 不依赖真 SDK。

## 已做的改动

分支：`fix_whl_build_support_rocm`（基于 `tile-ai/tilelang` `main` HEAD `bcb2da33`）
Commit：`f25f5819 [BugFix] Allow USE_ROCM=ON wheel builds on hosts without a ROCm runtime`
仓库 fork：`https://github.com/benenzhu/tilelang.git`

改了 2 个文件 / +43 行：

1. **`CMakeLists.txt`** —— 新增 cache 变量 `TILELANG_HIP_INCLUDE_DIR`（PATH 类型）
2. **`src/backend/rocm/CMakeLists.txt`** —— 在已有的 `find_rocm()` + stub 块之后追加 fallback：
   - 触发条件：`NOT ROCM_FOUND AND TILELANG_USE_HIP_STUBS`
   - headers 来源优先级：`TILELANG_HIP_INCLUDE_DIR` (cmake var) → `$TILELANG_HIP_INCLUDE_DIR` (env) → `/opt/rocm/include`（auto-detect）
   - 验证条件：所选目录下必须有 `hip/` 子目录
   - 满足后手动塞：`ROCM_FOUND=TRUE` / `ROCM_INCLUDE_DIRS=<headers>` / `ROCM_HIPHCC_LIBRARY=hip_stub` / `ROCM_HSA_LIBRARY=NOTFOUND`，让后续 TVM `add_subdirectory(...)` 时 `cmake/modules/ROCM.cmake` 能通过

## 这个 patch 解锁了什么、没解锁什么

✅ **只装 HIP/HSA headers**（`apt install hip-runtime-amd-dev hsa-rocr-dev`，纯头包，无需 GPU/driver/runtime libs）的机器上，`USE_CUDA=ON USE_ROCM=ON pip wheel . -v` 能跑通

❌ **完全没 ROCm 头**的机器仍然不行 —— TVM `runtime/rocm/*.cc` 直接 `#include <hip/hip_runtime_api.h>` `<hip/hip_version.h>` `<hsa/hsa.h>` 用了约 50 个 HIP/HSA 符号；tilelang vendored stub header (`src/backend/rocm/stubs/vendor/hip_runtime.h`) 是 minimal 的、且开头有 `#error` 明确禁止外用，没法替代。完整解决要 vendor 一份 HIP/HSA header 子树，工作量大且要随 ROCm 版本维护 ABI，不在本 patch 范围。

## 当前状态：在一台 NV 4090 机器上 build 仍然 FATAL

用户报告：把 patch push 到 fork、在 4090 上重跑 `pip wheel`，依然报 `Cannot find ROCM, USE_ROCM=ON`。需要在那台机器上诊断。

## 在新机器上要做的事

### Step 1：确认 build 用的就是这个 commit

```bash
cd <tilelang-build-dir>
git log -1 --oneline       # 应该看到 f25f5819
grep -n "Fallback for build hosts" src/backend/rocm/CMakeLists.txt   # 应有匹配
```

如果 commit 不对：可能 `pip wheel` 用了缓存的 sdist；或者 cibuildwheel 在隔离容器里 checkout 了别的 ref。

### Step 2：确认 HIP headers 在不在

```bash
ls -la /opt/rocm/include/hip/hip_runtime_api.h 2>&1
ls -la /opt/rocm/include/hsa/hsa.h 2>&1
ls -d /opt/rocm/include/hip 2>&1     # patch 触发的关键条件
echo "TILELANG_HIP_INCLUDE_DIR=${TILELANG_HIP_INCLUDE_DIR:-<unset>}"
apt list --installed 2>/dev/null | grep -iE "hip|rocm|hsa"
```

如果 `/opt/rocm/include/hip/` 不存在 **且** `TILELANG_HIP_INCLUDE_DIR` 没设：
fallback 不会触发，TVM 必然 FATAL。
→ 装包：`sudo apt install hip-runtime-amd-dev hsa-rocr-dev`（或等价的 dnf 包）

### Step 3：抓完整 cmake configure 日志

```bash
USE_CUDA=ON USE_ROCM=ON pip wheel . -v 2>&1 | tee /tmp/build.log
```

需要 grep 这几条 STATUS 看哪条出现哪条没出现：

```bash
grep -nE "ROCM Backend is enabled|ROCm runtime library not found|Found ROCM_|Cannot find ROCM" /tmp/build.log
```

预期：
- ✅ `ROCM Backend is enabled`（说明 USE_ROCM=ON 生效进了 `src/backend/rocm/CMakeLists.txt`）
- ✅ 二选一：`Found ROCM_HIPHCC_LIBRARY=...`（real find_rocm 成功）**或** `ROCm runtime library not found on host; using HIP headers from ...`（我的 fallback 触发）
- ❌ 不应该有 `Cannot find ROCM`

### Step 4：根据日志情况

| 现象 | 诊断 | 处理 |
|---|---|---|
| 没看到 "ROCM Backend is enabled" | `USE_ROCM=ON` 没传到 cmake | 检查 build 命令、scikit-build-core 配置 |
| 看到 "ROCM Backend is enabled" 但没看到我加的 STATUS 行 | fallback 条件没满足（headers 不在 / stubs 被关） | 看 Step 2 的 ls 结果 |
| 看到 `ROCm runtime library not found on host; using HIP headers from XXX` 但还是 FATAL | XXX 的内容有问题（路径不全 / TVM 二次 find_rocm 把变量重置了） | 把那条 STATUS 整段贴出来，要进一步改 patch |
| configure 通过但编译挂在 `<hip/...>` 找不到符号 | TVM `runtime/rocm/*.cc` 用了某个 vendored header 没覆盖的符号 | 真正在 vendor headers 上做完整覆盖 —— 本 patch 范围外，需要新方案 |

## 推荐下一步操作（顺序）

1. 在 4090 上跑 Step 1-3
2. 把日志贴回来 / 让那台机器上的 agent 解析
3. 如果是 Step 2 缺 headers：装好包再 build
4. 如果 patch 触发了但 TVM 编译挂在缺 symbol：考虑切到 **方案 A**——直接在 ROCm 物理机上的 `nvidia/cuda` docker（或反过来 NV 机上的 `rocm/dev-ubuntu` docker）里 build，避开 host header 不全的问题
5. build 出 wheel 后：分别在 NV 机器和 ROCm 机器 `pip install` 跑 smoke

## 验证 fat wheel 是否成功的快速指标

```bash
# 安装后
nm /path/to/site-packages/tilelang/lib/libtilelang.so | grep -i "RegisterROCmCopy\|RegisterCopy"
ls /path/to/site-packages/tilelang/lib/ | grep -E "stub|libhip|libcuda"
# 期望: libcuda_stub.so libcudart_stub.so libnvrtc_stub.so libhip_stub.so libhiprtc_stub.so 都在
```

## 还没解决的两个上游问题（patch 跑通后再考虑）

1. **`apache-tvm-ffi` PyPI 包是否带 ROCm**
   现 wheel 走 `pyproject.toml:34` 依赖 `apache-tvm-ffi~=0.1.0,>=0.1.2`，那个 `libtvm_ffi.so` 是上游编的。需要确认它的 `USE_ROCM` 是否开启，否则 tilelang 这边把 HIP 代码生成好之后下沉到 TVM runtime 创建 module 时还是会缺东西。
   验证：`nm site-packages/apache_tvm_ffi*/lib/libtvm*.so | grep -i hip` 看符号

2. **CI (`.github/workflows/dist.yml`) 改不改**
   现矩阵 4 个 wheel 全是 CUDA-only/Metal，没 ROCm。fat wheel 跑通且 ROCm self-hosted runner 上有 e2e smoke 之后，再决定 (a) 加 fat 矩阵还是 (b) 直接把现有 CUDA wheels 升级为 fat wheels。短期不动 CI。

## 文件位置速查

- 主 CMake：`/root/tilelang/CMakeLists.txt`
- ROCm backend CMake：`/root/tilelang/src/backend/rocm/CMakeLists.txt`
- 算子注册表：`/root/tilelang/src/op/copy.cc:81-106`
- ROCm copy 实现注册：`/root/tilelang/src/backend/rocm/op/copy.cc:194-207`
- TVM ROCm 强检查：`3rdparty/tvm/cmake/modules/ROCM.cmake:32`、`3rdparty/tvm/cmake/utils/FindROCM.cmake:35-67`
- TVM ROCm runtime（需要 HIP/HSA headers 编译）：`3rdparty/tvm/src/runtime/rocm/*.cc`
- HIP target attr 推断 (Python)：`/root/tilelang/tilelang/utils/target.py:44-89`
- Wheel CI：`/root/tilelang/.github/workflows/dist.yml`



success with: USE_CUDA=ON USE_ROCM=ON TILELANG_HIP_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/triton/backends/amd/include pip wheel . -v
