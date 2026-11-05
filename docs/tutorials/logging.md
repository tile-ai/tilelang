Logging in TileLang/TIRx
===================================================
<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/SiriusNEO">SiriusNEO</a>
</div>

## Logging Implementation

TileLang uses the logging interfaces from its fixed TIRx dependency:

- `3rdparty/tvm-tirx/include/tvm/runtime/logging.h`: `LOG`, `DLOG`, `VLOG`
  and the logging-state interfaces.
- `src/runtime/logging.cc`: TileLang's custom output format and
  `TVM_LOG_DEBUG` specification parser, linked into the matched provider.
- `src/support/check.h`: TileLang compiler-check macros using TVM FFI errors.

`CMakeLists.txt` selects `TVM_TIRX_USE_CUSTOM_LOGGING` and replaces the
default runtime logging implementation. The TIRx provider owns the resulting
runtime and compiler libraries. `TILELANG_RELEASE_BUILD` strips build-machine
paths from output; it is independent of whether debug log statements exist.

## Logging and Checks

```cpp
LOG(INFO) << "selected layout";
DLOG(INFO) << "layout candidate";
VLOG(2) << "inference details";
ICHECK(layout.defined()) << "Expected layout for tensor " << tensor;
```

`LOG` remains in ordinary builds. `DLOG` and `VLOG` require a nonzero
`TVM_LOG_DEBUG` preprocessor definition when compiling the relevant source.
When that definition is absent or zero, their macro expansion suppresses the
logging expression. `VLOG` additionally checks a per-file verbosity setting.
Use `ICHECK` for internal invariants and the appropriate typed FFI error for
user-facing validation. Fatal checks propagate an error; they are not a normal
control-flow mechanism.

## Building with Debug Logging

First follow the [source installation guide](../get_started/Installation.md)
to bootstrap the fixed FFI and development requirements. Use a fresh stage and
output directory after changing native sources:

```bash
python maint/scripts/build_native_artifacts.py stage --destination .native-debug-stage
CXXFLAGS="${CXXFLAGS} -DTVM_LOG_DEBUG=1" SKBUILD_CMAKE_BUILD_TYPE=Debug \
python maint/scripts/build_native_artifacts.py build \
  --stage .native-debug-stage --output native-debug-dist --frontend uv \
  --role provider --role frontend --wheel-only --install --jobs 4
```

This shell example uses a GCC/Clang-style definition; on MSVC use the
equivalent `/DTVM_LOG_DEBUG=1` compiler option. Check the provider's actual
compile command before expecting debug logs. Setting only the runtime
environment variable cannot enable statements that were compiled out.
`SKBUILD_CMAKE_BUILD_TYPE` selects the PEP 517 build type; a raw CMake cache
definition alone is not a substitute for that frontend setting.

## Runtime Settings

Enable DLOG statements before importing the compiler:

```bash
TVM_LOG_DEBUG=1 python code.py
```

`TVM_LOG_DEBUG=1` enables DLOG globally. A comma-separated specification also
selects VLOG verbosity by source file:

```bash
TVM_LOG_DEBUG="DEFAULT=2,transform/layout_inference/layout_inference.cc=3" python code.py
```

`DEFAULT` applies when no per-file entry exists. A level of `-1` disables VLOG
for that file. Paths are normalized relative to the last `src/` component,
with Windows separators normalized as well. An unset value or `0` disables
debug logging. These verbosity levels select debug statements, not the
severity of every ordinary `LOG` message.

The compile-time definition and runtime setting share the spelling
`TVM_LOG_DEBUG`, but serve different purposes. Verify both the build and the
loaded provider when diagnosing missing output.
