# Ascend AI Core exception dump

The Ascend backend can save kernel buffer data when an AI Core hardware
exception occurs. This helps recover inputs after illegal memory accesses and
other device execution failures. The feature is disabled by default.

For implementation details, see the [design document in Chinese](ascend_exception_dump_design_zh.md).

Enable it with `pass_configs={"tl.ascend_exception_dump": True}` when compiling for
`target="ascend"`:

```python
import os

# Set these before importing torch / torch_npu or initializing ACL.
os.environ["ASCEND_DUMP_PATH"] = "/tmp/tilelang_exception_dump"
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"

import tilelang
from tilelang.ascend.exception_dump_bin import parse_exception_dump

kernel = tilelang.compile(
    prim_func,
    target="ascend",
    execution_backend="tvm_ffi",
    pass_configs={"tl.ascend_exception_dump": True},
)
```

This requires **CANN 9.2 or newer**, with
`aclrtExceptionInfoCallbackRegister`, `aclrtGetArgsFromExceptionInfo`, and
`acldumpSaveExceptionInfo`. It applies to the Ascend backend. Enabling it adds
buffer metadata to the device launch arguments and registers an exception
callback. Disabled kernels keep their existing launch interface.

Only `execution_backend="tvm_ffi"` is supported for this feature.
Enabling dumping with Cython or PTO raises an error before cache lookup or
lowering. With dumping disabled, execution backends keep their existing behavior.

The feature accepts contiguous tensor buffers, including dynamic shapes, with at
most **16 tensor arguments per generated device kernel**. Generated kernel names
must contain **fewer than 128 UTF-8 bytes**. Non-contiguous strides, nonzero or
symbolic element offsets, unsupported dtypes and external source-code kernels
are rejected when dumping is enabled.

Host control flow and disk-cache restore use the existing TVM-FFI framework.
No Cython wrapper, shared expression-printer changes or extra cache-version
mechanism are required. The native Ascend runtime constructs a record for each
actual device launch and registers the process-local exception callback.

The callback saves valid, nonempty buffer arguments, including output buffers,
under:

```text
<ASCEND_DUMP_PATH>/extra-info/data-dump/<device>/<kernel_name>.custom.<timestamp>
```

The dump name uses the generated device kernel name. Use the name in the
generated source or dump directory when parsing; it may differ from the Python
function name. NULL pointers and zero-sized buffers are skipped. Dump entries
are labelled `input` and numbered after skipping these buffers, so their
indices need not match the original function parameter indices. Scalar
arguments are not returned as tensors.

Each tensor is described as a one-dimensional ND tensor using its flattened
element count, rather than its byte size. Original multidimensional shapes are
not preserved in the dump metadata; reshape recovered arrays using the known
input shapes.

After catching a kernel execution or stream synchronization exception, parse
the dump with CANN's `msaicerr.py`:

```python
tensors = parse_exception_dump(
    dump_path="/tmp/tilelang_exception_dump",
    kernel_name="main_kernel",  # Replace with the generated device kernel name.
)
for tensor in tensors:
    print(tensor["type"], tensor["index"], tensor["dtype"], tensor["data"])
```

Set `ASCEND_HOME_PATH` to the CANN installation containing
`tools/msaicerr/msaicerr.py`. The parser waits two seconds by default for the
asynchronous callback, then selects the newest timestamp for the requested
kernel across all device directories. Use `wait_seconds` to adjust the wait.
The function raises an error when the dump or parser is missing, parsing fails,
or no matching tensors are produced.

Both raw `.bin` and `.npy` outputs are supported; CANN 9.2 produces `.npy` files.
Each result contains `data` (a NumPy array), `type`, `index`, `dtype` (the dtype
in the filename), and `file` (the parsed file path). Raw data is flat and may be
reshaped by the caller. NumPy files preserve their stored shape and dtype.
Raw float16/32/64, signed and unsigned 8/16/32/64-bit integers, bool, and bfloat16
are supported. Bfloat16 uses `ml-dtypes`. Unsupported raw types, including
packed sub-byte formats, raise `ValueError` instead of being decoded as float32.

Every parse creates a fresh subdirectory under `output_dir`, defaulting to
`<dump_path>/parsed`. Parsed files are retained, and existing files are never
deleted. To read a raw file directly, use
`read_msaicerr_bin(file_path, dtype, shape=None, header_size=0)` from the same
module.

Hardware exceptions can leave the device context unusable. Run intentional
fault-injection tests in a separate process; ordinary parser tests require no
NPU or CANN installation.

A real fault regression is available as an explicit opt-in:

```bash
TILELANG_TEST_ASCEND_HW_EXCEPTION=1 python -m pytest \
  testing/ascend/runtime/test_exception_dump_hardware.py -q -s
```

The test uses TVM-FFI in a child process, checks one normal execution, then
intentionally passes a NULL output pointer. The parent parses the custom dump
written by the production callback and compares both input tensors against
copies saved before the fault, including exact bytes and SHA-256. Kernel source,
exception logs, original dump, parsed arrays and `report.json` remain under the
test's temporary directory. No Cython fault path is included.
