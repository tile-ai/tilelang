# Carver: A Tile-Structure Based Hint Recommend Framework for Machine Learning Compilers

**Carver** is a lightweight framework for generating and ranking tile configurations (also known as **tiling strategies**, **blocking schemes**, or **scheduling hints**) for common GPU, CPU, and accelerator backends. It helps you explore efficient mappings of loops for operations such as matrix multiplication, elementwise transforms, and other reduction-oriented kernels.

Carver combines hardware architecture information, user-defined tile structures, and built-in heuristics to recommend tiling strategies (or "hints"). The recommended hints are easily adaptable to multiple backends, including [TVM](https://tvm.apache.org/), [triton](https://github.com/openai/triton), [tilelang](https://github.com/tile-ai/tilelang) (or other domain-specific compilers).

## Native TIRx input contract

The template constructors and `recommend_hints` entry points retain their existing
arguments. `equivalent_function()` now returns a native
`tvm_tirx.tirx.Function`; the attention template returns its existing pair of
functions. Tensor parameters are `ir.Var` objects whose `TensorType` owns shape,
dtype, scope and layout. The function body is a `SeqStmt`. Consumers of the former
`PrimFunc.buffer_map` or TVM scheduling objects must use this native contract:

```python
from tilelang import carver
from tvm_tirx import tirx

func = carver.MatmulTemplate(M=128, N=128, K=128).equivalent_function()
assert isinstance(func, tirx.Function)
tensors = [parameter for parameter in func.params if tirx.is_tensor_var(parameter)]
shapes = [tuple(parameter.ty.shape) for parameter in tensors]
```

Carver's private workload builder emits real native loads, stores, typed tensor
allocations and TileLang stage annotations. Its private `NativeSchedule` tracks
stage, loop and tensor views over those objects. It does not provide the former
TE stack or a general replacement for TVM's `Schedule`. User-supplied functions
can retain `T.sblock`, `T.axis.spatial` and `T.axis.reduce` declarations through
`import tilelang.language as T`.

Graph rewrites require independent, directly bound stages with an explicit
`tirx.noalias` contract. They preserve public tensor parameters, element types,
strides and offsets. Reindexing materializes private storage; layout changes
must be invertible, and reduction transformations preserve accumulation order.
Parallel loops, shared loop stages, pointer aliases, intervening writes and effectful producers
are rejected when the rewrite cannot prove the required ordering. A failed
candidate preserves the original function, so another hint strategy can be
tried. Successful changes invalidate old analysis handles; consumers refresh
surviving handles through the analysis that created them.

`FlashAttentionTemplate` remains a hint model for its QK and SV matrix products.
Its equivalent functions do not include softmax or causal masking. A proposed
hint is a tile configuration; compiling and numerically validating the consumer
kernel remains necessary on its selected backend.

---

## Key Features
- **Unified Tiling Framework**: Generate tile candidates for multiple backends under a unified API.
- **Architecture-Specific Modeling**: Take into account architecture constraints (e.g., CUDA `smem_cap`, warp size, CPU cache structure, etc.) when generating hints.
- **Flexible Templates**: High-level templates (like `MatmulTemplate`, `GeneralReductionTemplate`, `ElementwiseTemplate`) let you concisely specify kernel structures.
- **Extendable**: Easily add support for new backends and new operation templates.

---

## Usage Examples

### Basic Usage: General Reduction Template

Once installed tilelang, you can import Carver and start creating templates:

```python
from tilelang import carver
from tilelang.carver.arch import CUDA

# Instantiate a CUDA device object for an RTX 4090
arch = CUDA("nvidia/geforce-rtx-4090")

# Create a general reduction template for a loop nest:
# for i in Spatial(1024):
#     for j in Spatial(1024):
#         for k in Reduce(1024):
#             ...
carve_template = carver.GeneralReductionTemplate(
    structure="SSR",
    shape=[1024, 1024, 1024],
    dtype="float16",
).with_arch(arch)

# Generate top 20 tile candidates (aka scheduling hints)
hints = carve_template.recommend_hints(topk=20)
for hint in hints:
    print(hint)
```

**Example Output** (truncated):
```python
{
  'block': [1, 128],
  'thread': [1, 128],
  'rstep': [64],
  ...
},
{
  'block': [2, 64],
  'thread': [2, 64],
  'rstep': [64],
  ...
},
...
{
  'block': [1, 16],
  'thread': [1, 16],
  'rstep': [512],
  'reduce_thread': [8],
  ...
}
```

A tile structure composed of S and R can simulate various cases. For example, structure `SS` represents a 2D element-wise operation, while `SSR` can represent a general matrix multiplication.

We can specialize more advanced templates to provide finer-grained information, such as `MatmulTemplate`.

### Matmul Template

Carver also provides a specialized `MatmulTemplate` for matrix multiplication (e.g., `C = A * B`), automatically inferring common tiling strategies (thread blocks, warps, use of tensor cores, etc.).

```python
from tilelang import carver
from tilelang.carver.arch import CUDA

arch = CUDA("nvidia/geforce-rtx-4090")
carve_template = carver.MatmulTemplate(
    M=1024,
    N=1024,
    K=1024,
    in_dtype="float16",
    accum_dtype="float16",
    out_dtype="float16",
).with_arch(arch)

# Retrieve the (symbolic) function describing the matmul
func = carve_template.equivalent_function()
print("Equivalent Function:\n", func)

# Generate hints
hints = carve_template.recommend_hints(topk=20)
for hint in hints:
    print(hint)
```

**Example Output**:
```python
{
  'block': [32, 64],
  'warp': [16, 32],
  'rstep': [128],
  'use_tc': True,
  ...
},
{
  'block': [64, 32],
  'warp': [32, 16],
  'rstep': [128],
  'use_tc': True,
  ...
},
...
{
  'block': [256, 32],
  'warp': [128, 16],
  'rstep': [32],
  'use_tc': True,
  ...
}
```

---

## Supported Architectures

Carver currently provides out-of-the-box support for:
- **CUDA**: e.g., `arch = CUDA("nvidia/geforce-rtx-4090")`
- **CDNA** (AMD GPU-like backends)
- **CPU**

Adding a new architecture is as simple as implementing a new subclass of `TileDevice` or providing a custom target that describes:
- Shared/local memory capacity
- Warp (or vector) size
- Cache sizes
- Tensor instructions available

Below is an **illustrative snippet** of the CUDA backend:
```python
class CUDA(TileDevice):
    def __init__(self, target: Union[tvm_tirx.target.Target, str]):
        ...
        self.platform = "CUDA"
        # Device constraints
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        ...
        self.transaction_size = [32, 128]  # bytes
        self.bandwidth = [750, 12080]  # MB/s, approximate
        self.available_tensor_instructions = None

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = (
            TensorInstruction("mma", [16, 16]),
            TensorInstruction("wmma", [16, 16]),
        )
        return [t.shape for t in self.available_tensor_instructions]

    def __repr__(self):
        return f"CUDA({self.target})"
```

## Adapting Hints to Other Compilers

One of Carver’s main benefits is its adaptability. Here are a examples for triton lang:

Given a Carver hint like:
```python
{"block": [32, 64], "warp": [16, 32], "rstep": [128], "use_tc": True, "vectorize": {"A_reindex": 8, "B_reindex": 8}}
```
You might interpret this in **Triton** as:
- `block_m = 32, block_n = 64, block_k = 128`
- Potential warp usage = `warp_m = 16, warp_n = 32`
- `vectorize`: load data with a vector width of 8
- If `use_tc` is true, consider using Tensor Cores (TensorOps in Triton) if supported.

This helps quickly test multiple configurations without manually guessing.

## Supported Templates

Carver abstracts common loop patterns through templates:
- **`GeneralReductionTemplate`**: For general `Spatial-Spatial-Reduce` (SSR) structures or similar.
- **`FlashAttentionTemplate`**: For attention-like operations with flash memory.
- **`MatmulTemplate`**: For standard matrix multiplication `C = A * B`.
- **`GEMVTemplate`**: For `y = Ax` or `y = xA` style operations.
- **`ElementwiseTemplate`**: For elementwise transformations or pointwise ops.

You can also create your own specialized templates if you have unique loop structures or constraints. For instance, you might define specialized templates for convolution, flash attention, etc.

## TODO Items

- [ ] **Adapt to tile language**: Provide ready-made scheduling calls or wrappers for [tilelang](https://github.com/tile-ai/tilelang) to streamline end-to-end integration.
