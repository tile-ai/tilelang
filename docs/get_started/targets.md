# Understanding Targets

TileLang is built on top of TVM, which relies on **targets** to describe the device you want to compile for.
The target determines which code generator is used (CUDA, HIP, Metal, LLVM, …) and allows you to pass
device-specific options such as GPU architecture flags. This page summarises how to pick and customise a target
when compiling TileLang programs.

## Common targets

TileLang ships with a small set of common target kinds. Use a bare string for the kind, or a TVM target config
dictionary when you need options such as GPU architecture or CPU model. The most frequent choices are listed below:

| Base name | Description |
| --------- | ----------- |
| `auto` | Detects CUDA → HIP → Metal in that order. Useful when running the same script across machines. |
| `cuda` | NVIDIA GPUs. Use a config dict for options such as `{"kind": "cuda", "arch": "sm_80"}`. |
| `cutedsl` | NVIDIA CUTLASS/CuTe DSL backend. Requires `nvidia-cutlass-dsl`. |
| `hip` | AMD GPUs via ROCm. Use a config dict for options such as `{"kind": "hip", "mcpu": "gfx90a"}`. |
| `metal` | Apple Silicon GPUs (arm64 Macs). |
| `llvm` | CPU execution. Use a config dict for options such as `{"kind": "llvm", "mtriple": "x86_64-linux-gnu"}`. |
| `webgpu` | Browser / WebGPU runtimes. |
| `c` | Emit plain C source for inspection or custom toolchains. |

To add options, pass a target config dictionary. For example:

```python
target = {"kind": "cuda", "arch": "sm_90"}
kernel = tilelang.compile(func, target=target, execution_backend="cython")
# or
@tilelang.jit(target=target)
def compiled_kernel(*args):
    return func(*args)
```

## Target input forms

Most TileLang APIs that accept a target, such as `tilelang.compile`, `tilelang.jit`, and the autotuner, accept the
same input forms:

```python
target = "auto"                                      # detect CUDA, HIP, or Metal
target = "cuda"                                      # bare TVM target kind
target = {"kind": "cuda", "arch": "sm_90"}           # target config dict
target = tvm.target.Target({"kind": "cuda"})         # already-built TVM Target
```

Use the bare string form for simple cases. Use a config dictionary when you need target attributes such as CUDA
`arch`, CUDA `code`, HIP `mcpu`, or LLVM CPU options. Dictionary keys must be valid attributes for that target kind;
invalid attributes are rejected when TVM constructs the target.

## Default target

If you do not pass `target=...`, TileLang reads `TILELANG_DEFAULT_TARGET`. When the environment variable is unset,
the default is `auto`.

```bash
export TILELANG_DEFAULT_TARGET=cuda
```

For target options, use a dict-like string. This is useful in scripts that rely on the default target through
`tilelang.compile(..., target=None)`, `@tilelang.jit`, or autotuning:

```bash
export TILELANG_DEFAULT_TARGET='{kind: "cuda", arch: "sm_90"}'
export TILELANG_DEFAULT_TARGET='{kind: "cuda", arch: "sm_100f", code: ["sm_100a", "sm_103a"]}'
```

In Python code, prefer passing a real dictionary instead of a string:

```python
target = {"kind": "cuda", "arch": "sm_100f", "code": ["sm_100a", "sm_103a"]}
kernel = tilelang.compile(func, target=target)
```

## CUDA `arch` and `code`

For CUDA targets, `arch` selects the SM family TileLang/TVM should compile for. TileLang accepts `sm_...` architecture
tokens for this field. In the common case, setting only `arch` is enough:

```python
target = {"kind": "cuda", "arch": "sm_90"}
```

TileLang passes this to NVCC as a single architecture target:

```bash
-arch=sm_90
```

Use `code` only when you need explicit NVCC `-gencode` behavior, for example when compiling the virtual architecture
derived from `arch` but emitting SASS for multiple GPU code instances:

```python
target = {
    "kind": "cuda",
    "arch": "sm_100f",
    "code": ["sm_100a", "sm_103a"],
}
```

This produces NVCC flags equivalent to:

```bash
-gencode arch=compute_100f,code=[sm_100a,sm_103a]
```

`code` must be a Python list of exact SM code tokens such as `sm_100a` or `sm_103a`. String forms such as
`"sm_100a,sm_103a"` or `"[sm_100a,sm_103a]"` are intentionally not supported.

When `code` contains more than one entry, TileLang emits a fat binary (`fatbin`) because NVCC does not allow
`--cubin` output with multiple GPU code instances.

### Advanced: Specify Exact Hardware

When you already know the precise GPU model, you can encode it in the target config via `arch="sm_XX"` or by
using one of TVM’s pre-defined target tags such as `nvidia/nvidia-h100`. Supplying this detail is optional for
TileLang in general use, but it becomes valuable when the TVM cost model is enabled (e.g. during autotuning).  The
cost model uses the extra attributes to make better scheduling predictions.  If you skip this step (or do not use the
cost model), generic targets like `cuda` or `auto` are perfectly fine.

All CUDA compute capabilities recognised by TVM’s target registry are listed below.  Pick the one that matches your
GPU and set it in the target config or use the corresponding target tag—for example `nvidia/nvidia-a100`.

| Architecture | GPUs (examples) |
| ------------ | ---------------- |
| `sm_20` | `nvidia/tesla-c2050`, `nvidia/tesla-c2070` |
| `sm_21` | `nvidia/nvs-5400m`, `nvidia/geforce-gt-520` |
| `sm_30` | `nvidia/quadro-k5000`, `nvidia/geforce-gtx-780m` |
| `sm_35` | `nvidia/tesla-k40`, `nvidia/quadro-k6000` |
| `sm_37` | `nvidia/tesla-k80` |
| `sm_50` | `nvidia/quadro-k2200`, `nvidia/geforce-gtx-950m` |
| `sm_52` | `nvidia/tesla-m40`, `nvidia/geforce-gtx-980` |
| `sm_53` | `nvidia/jetson-tx1`, `nvidia/jetson-nano` |
| `sm_60` | `nvidia/tesla-p100`, `nvidia/quadro-gp100` |
| `sm_61` | `nvidia/tesla-p4`, `nvidia/quadro-p6000`, `nvidia/geforce-gtx-1080` |
| `sm_62` | `nvidia/jetson-tx2` |
| `sm_70` | `nvidia/nvidia-v100`, `nvidia/quadro-gv100` |
| `sm_72` | `nvidia/jetson-agx-xavier` |
| `sm_75` | `nvidia/nvidia-t4`, `nvidia/quadro-rtx-8000`, `nvidia/geforce-rtx-2080` |
| `sm_80` | `nvidia/nvidia-a100`, `nvidia/nvidia-a30` |
| `sm_86` | `nvidia/nvidia-a40`, `nvidia/nvidia-a10`, `nvidia/geforce-rtx-3090` |
| `sm_87` | `nvidia/jetson-agx-orin-32gb`, `nvidia/jetson-agx-orin-64gb` |
| `sm_89` | `nvidia/geforce-rtx-4090` |
| `sm_90a` | `nvidia/nvidia-h100` (DPX profile) |
| `sm_100a` | `nvidia/nvidia-b100` |

Refer to NVIDIA’s [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) page or the TVM source
(`3rdparty/tvm/src/target/tag.cc`) for the latest mapping between devices and compute capabilities.

## Creating targets programmatically

TileLang exposes the helper `tilelang.backend.target.determine_target` (returns a canonical target string or config
by default, or the `Target` object when `return_object=True`):

```python
from tilelang.backend.target import determine_target

tvm_target = determine_target(
    {"kind": "cuda", "arch": "sm_100f", "code": ["sm_100a", "sm_103a"]},
    return_object=True,
)
kernel = tilelang.compile(func, target=tvm_target)
```

You can also build targets directly through TVM:

```python
from tvm.target import Target

target = Target("cuda", host="llvm")
target = target.with_host(Target({"kind": "llvm", "mcpu": "skylake"}))
```

TileLang accepts bare target strings, target config dictionaries, and `Target` inputs. For targets with options,
prefer config dictionaries over CLI-style strings.

## Troubleshooting tips

- If you see `Target {'kind': 'cuda', 'arch': 'sm_80'} is not supported`, double-check the spellings and that the option is valid for
  the target kind. Any invalid option will surface as a target-construction error.
- If CUDA `code` fails validation, make sure it is a list of exact SM code tokens, for example
  `["sm_100a", "sm_103a"]`, not a comma-separated string.
- Runtime errors such as “no kernel image is available” usually mean the `arch` value does not match the GPU you are
  running on. Try dropping the flag or switching to the correct compute capability.
- When targeting multiple environments, use `auto` for convenience and override with an explicit config only when
  you need architecture-specific tuning.
