Logging in Tilelang/TVM
===================================================
<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/SiriusNEO">SiriusNEO</a>
</div>

## TVM Logging Overview

Tilelang currently utilizes the logging system from TVM. The implementation can be found in:

- [include/tvm/runtime/logging.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/logging.h): Macro definitions
- [src/runtime/logging.cc](https://github.com/apache/tvm/blob/main/src/runtime/logging.cc): Logging logic implementation

The design style is inspired by [Google's glog](https://google.github.io/glog/stable/).

## Logging Categories

There are three primary macro types:

```c++
LOG(INFO) << "aaa";
DLOG(INFO) << "aaa";
VLOG(1) << "aaa";
```

- **LOG**: Standard logging preserved in code for displaying necessary information at different levels during runtime. Most Tilelang C++ error reporting is implemented via `LOG(FATAL) << "error msg"`.
- **DLOG**: Debug logging for developer debugging output. DLOG is controlled at build time by the TVM_LOG_DEBUG environment variable and is **eliminated in Release builds through dead code elimination**.
  - The key difference between LOG(DEBUG) and DLOG is this build-time elimination. We recommend using DLOG over LOG(DEBUG), as the latter has overlapping functionality and gets compiled into the release runtime.
- **VLOG**: [Verbose logging](https://google.github.io/glog/stable/logging/#verbose-logging), primarily for debugging. Its main feature is customizable verbosity levels. For example, VLOG(n) where n can be 1, 2, 3, 4, 5, or 6, enabling complex tracing requirements. In contrast, LOG and DLOG typically use predefined verbose levels like INFO and DEBUG.
  - In practical Tilelang development, VLOG is used less frequently.
  - TVM's VLOG is implemented using DLOG, thus inheriting DLOG's characteristics.

Additional useful macros include various **CHECK** variants:

```c++
CHECK(cond) << "error msg";
DCHECK(cond) << "error msg";
ICHECK(cond) << "error msg";
```

The implementation routes errors to LogFatal:

```c++
#define CHECK(x)                                                \
  if (!(x))                                                     \
  ::tvm::runtime::detail::LogFatal(__FILE__, __LINE__).stream() \
      << "Check failed: (" #x << ") is false: "
```
- **DCHECK**: Debug mode CHECK, only compiled in debug builds
- **ICHECK**: Internal Check that should exist in Release builds. When ICHECK fails, the entire system should report an error.

## Logging Verbose Levels

TVM defines 5 levels for LOG and DLOG (adding DEBUG compared to glog):

```c++
#define TVM_LOG_LEVEL_DEBUG 0
#define TVM_LOG_LEVEL_INFO 1
#define TVM_LOG_LEVEL_WARNING 2
#define TVM_LOG_LEVEL_ERROR 3
#define TVM_LOG_LEVEL_FATAL 4
```

## Using Logging in TileLang Development

### Guidelines

For temporary debugging output in your code, there are no restrictions (you can even use std::cout). Just remember to remove it before submitting a PR.

For meaningful logging that should remain in the Tilelang codebase:

- Critical correctness checks: Use ICHECK with sufficient error messages to facilitate debugging when issues arise.
- Complex Pass debugging: For passes requiring intermediate output that may need future review (e.g., LayoutInference), use DLOG.
- General INFO/WARNING messages: Use standard LOG.

### Enabling Log Output in Tilelang

To specify current log level at runtime, we need to set the environment variable `TVM_LOG_LEVEL`. An example usage is:

```c++
TVM_LOG_DEBUG=1 python3 code.py
```

which enables all DEBUG/INFO (level <= 1) logs for all files.

#### Detailed Rules for TVM_LOG_DEBUG Specification

The parsing logic is in `logging.cc`. Reference: [HyperAI Zhihu Article](https://zhuanlan.zhihu.com/p/1933106843468665163).

Launch Python with `TVM_LOG_DEBUG=<spec>`, where `<spec>` is a comma-separated list of level assignments in the form `<file_name>=<level>`. Important notes:

- The special filename DEFAULT sets the LOG level for all files.
- `<level>` can be set to -1 to disable LOG for that file.
- `<file_name>` is the C++ source filename (e.g., .cc, not .h) relative to the `src/` directory in the TVM repository. The `src/` prefix is optional when specifying file paths.

### Enabling Debug Mode

To enable DLOG/DCHECK, developers need to first build Tilelang in Debug mode:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON
```

Tilelang's CMake logic automatically adds the `TVM_LOG_DEBUG` macro, compiling all DLOG statements:

```cmake
target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
```

Then you also need to specify the runtime environment variables. For example, to use `DLOG(INFO) << "xxx"` for debugging, run your code with INFO level (1): `TVM_LOG_DEBUG=1`.

:::{note}
   **Important**: There are two TVM_LOG_DEBUG variables. (1) Compile-time macro: Determines whether debug content (like DLOG) is compiled into the .so file. Referenced in C++ source via #ifdef TVM_LOG_DEBUG. This is automatically enabled when using Debug build mode in CMake. (2) Runtime environment variable: Controls logging level at runtime. TVM provides a specification for this variable, allowing control over per-file logging levels.

   These two should ideally have different names, but TVM uses the same name for both, which can cause confusion.
:::
