# Shifted mHC Pre on Ascend

Forward kernels for BF16 shifted manifold-constrained hyper-connections (mHC)
on Ascend 950. One mixed Cube/Vector launch computes the projected mixing
coefficients, RMS statistics and weighted layer input. The weighted layer input
uses the **incoming** pre-mix coefficients; the newly computed coefficients are
returned for the next layer.

The operator has one public entry module and two implementations:

| File | Purpose |
| --- | --- |
| `mhc_pre.py` | Input validation, shape-based dispatch and cached native launchers |
| `mhc_pre_mk.py` | M/K partitioning and the shared SIMD preparation/Sinkhorn macros |
| `mhc_pre_large.py` | Whole-M ownership with a full K reduction in each core pair |
| `test_mhc_pre.py` | CPU routing checks and four-output device regressions |

## Inputs and outputs

All inputs must be contiguous and on the same NPU. `H` is 4096, 5120 or 7168;
the serving range covered by the original hardware tests is `N=1..8192`.

| Input | Shape | Dtype |
| --- | --- | --- |
| `x` | `[N, 4, H]` | BF16 |
| `prev_pre` | `[N, 4]` or `[N, 4, 1]` | FP32 |
| `phi` | `[24, 4*H]` | FP32 |
| `alpha` | `[3]` | FP32 |
| `bias` | `[24]` | FP32 |

Both entry points return these four tensors in order:

| Output | Shape | Dtype |
| --- | --- | --- |
| `layer_input` | `[N, H]` | BF16 |
| `next_pre_mix` | `[N, 4, 1]` | FP32 |
| `post_mix` | `[N, 4, 1]` | FP32 |
| `comb_mix` | `[N, 4, 4]` | FP32 |

Let `R = FP32(x).reshape(N, 4*H)`. The equations are:

```text
mix = (R @ phi.T) / sqrt(mean(R * R, axis=-1) + 1e-20)
next_pre_mix = sigmoid(mix[:, :4] * alpha[0] + bias[:4]) + 1e-6
post_mix = 2 * sigmoid(mix[:, 4:8] * alpha[1] + bias[4:8])
logits = (mix[:, 8:24] * alpha[2] + bias[8:24]).reshape(N, 4, 4)
layer_input = BF16(sum(FP32(x) * prev_pre, axis=stream))
```

`comb_mix` uses 20 Sinkhorn iterations. The first iteration applies row softmax,
adds `1e-6`, then normalizes columns with `1e-6` added to the denominator.
Each subsequent iteration normalizes rows and columns, adding `1e-6` to each
denominator. The projection uses HF32 nearest-zero multiplication; statistics
and coefficient operations use FP32, and layer output uses BF16 round-to-nearest-even.
This example implements forward execution without an autograd backward.

## Use

Build this repository with Ascend support; see the
[Ascend backend guide](../../../tilelang/ascend/README.md). The runtime requires
matching PyTorch/`torch_npu`, CANN/Bisheng, Ninja and a C++ compiler. The first
call compiles the selected kernel and a small host allocation helper. No vLLM
installation or external operator package is needed.

From the repository root:

```python
import torch
import torch_npu

from examples.ascend.mhc.mhc_pre import bind_mhc_pre, mhc_pre

device = torch.device("npu:0")
n, h = 33, 4096
x = torch.randn(n, 4, h, dtype=torch.bfloat16, device=device)
prev_pre = torch.rand(n, 4, 1, dtype=torch.float32, device=device)
phi = torch.randn(24, 4 * h, dtype=torch.float32, device=device) / (4 * h) ** 0.5
alpha = torch.ones(3, dtype=torch.float32, device=device)
bias = torch.zeros(24, dtype=torch.float32, device=device)

with torch.inference_mode():
    layer_input, next_pre_mix, post_mix, comb_mix = mhc_pre(x, prev_pre, phi, alpha, bias)

    # Bind stable tensor objects once for repeated execution.
    run = bind_mhc_pre(x, prev_pre, phi, alpha, bias)
    layer_input, next_pre_mix, post_mix, comb_mix = run()
```

`mhc_pre` caches compiled executors by shape, core capacity and device, and
accepts new tensor objects on each call. Cached executors retain allocation
options, not input values or output tensor storage.

`bind_mhc_pre` validates and binds the five input objects once. Their contents
may change in place between calls; their shape, dtype, strides and device must
remain fixed. Bind again when replacing an input object. Every invocation
allocates fresh outputs and, on the MK path, two fresh private workspaces.
The native launcher resolves the caller's current NPU stream on each call.

The capacity defaults are `num_aic_cores=28, num_vec_cores=56`. Pass the actual
available counts when using another capacity; this example assumes two AIVs
per active AIC and a 256 KiB UB.

## Dispatch and dataflow

On 28 AIC / 56 AIV, dispatch uses:

| Region | Implementation |
| --- | --- |
| `N <= 2304` | MK |
| `2305 <= N <= 3584`, complete original MK tile | MK |
| `2305 <= N <= 3584`, partial original MK tile | Whole-M |
| `N >= 3585` | Whole-M |

The original MK tile height is computed from the M/K partition and can be 240
or 256 in the middle region. Completeness uses that height, rather than the
whole-M implementation's tile size or `N % 128`. For other capacities, the
original capacity-based M/K partition is used; the measured middle-region
rule is specific to 28/56.

MK workers share residual loading between projection, squared-sum reduction
and weighted Apply. They publish private projection/RMS partials to GM, then
reduce the K workers and calculate the coefficients after a mixed-core barrier.
The separate read/write workspace formals intentionally alias the same storage
across that publication boundary.

Whole-M workers own complete token tiles and the full K dimension. Projection
results reach the paired Vector cores through Fixpipe, and no GM partial
workspace is needed. Full tiles and the final partial tile use distinct input
pipelines. A partial tile computes only valid rows and fills the Cube-facing
NZ padding; an empty AIV half still participates in the synchronization
protocol. Both implementations keep all Sinkhorn iterations in registers and
read residuals once.

## Tests

Run the focused suite from a built checkout:

```bash
python -m pytest examples/ascend/mhc/test_mhc_pre.py -q
```

CPU tests check route transitions, complete-MK returns and other capacities.
Device tests check all four outputs against independent CPU FP32 equations,
including zero input, small/large N, partial tiles, coefficient packs and
multiple jobs per core. They also check fresh allocations, in-place parameter
updates, current-stream execution and replacing functional inputs at the same
shape. Device tests skip when an Ascend NPU is unavailable.

Development tolerances are `atol=0.03125, rtol=1e-3` for the BF16 layer output
and `atol=5e-5, rtol=1e-3` for the FP32 outputs. These checks and the original
hardware measurements do not certify the performance of this repackaged API;
remeasure the callable boundary used by an application, including allocations
and any per-call validation.

The implementation retains its DeepSeek and Q4Kernels contributor notices and
is distributed under the repository's [MIT license](../../../LICENSE).
