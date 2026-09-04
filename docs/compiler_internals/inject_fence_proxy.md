# InjectFenceProxy Pass

`tl.InjectFenceProxy` is a TIR-level transform that keeps the GPU proxy state
consistent on NVIDIA Hopper (SM90+) by inserting `fence.proxy.async`
instructions when execution switches from **generic proxy** memory operations
to **async proxy** operations.

For the user-facing distinction between thread barriers, async completion, and
proxy ordering, see [Synchronization and Memory Ordering](../programming_guides/synchronization.md#proxy-fences).

## Why Fences Are Needed

Hopper separates memory instructions into generic and asynchronous proxy paths.
When an asynchronous instruction (for example, WGMMA, TMA, or
`cp.async.bulk`) consumes shared state written through the generic proxy, the
hardware requires a `fence.proxy.async` to order that transition. Missing the
fence can expose stale data to the async operation.

This is not a general async-completion fence. It orders prior generic-proxy
shared-memory accesses before subsequent async-proxy accesses; it neither waits
for an async operation nor makes threads rendezvous.

## What the Pass Does

- Walks statements in execution order while tracking a (may-)state of the last
  proxy kind (**generic**, **async**, or **none/reset**). Control-flow joins
  (e.g. `if`) merge states conservatively.
- Injects `fence.proxy.async` right before an async-proxy instruction whenever the preceding state can be generic.
- Treats an explicit `fence_proxy_async` as a state reset and hoists a common
  fence out of branches or loops when that is safe.

By default, unknown/external calls do **not** affect proxy state. Opaque calls
that may write into **shared memory** (for example, STMatrix or a custom call
using `tvm_access_ptr` / `address_of`) are treated as generic proxy traffic so
a later async-proxy op will still be fenced.

### Timeline View

```
generic shared store -> fence.proxy.async -> TMA / WGMMA / cp.async.bulk
     generic proxy                              async proxy
```

The proxy tracker effectively scans the program in execution order. When it
detects a possible transition from generic to async (between the store and the
async op above), it synthesizes a `fence.proxy.async` to reset the hardware
proxy state before the async path runs.

## Coverage of Intrinsics

The async side currently includes TileLang TMA load/store (including im2col,
multicast, gather4, and scatter4), WGMMA, TCGEN05 MMA, and the TVM/PTX
`cp.async.bulk` intrinsic. The generic-write side includes shared-memory
`BufferStore` statements and opaque calls whose arguments reveal a writable
shared pointer through `tvm_access_ptr` or `address_of`.

Consequently, ordinary `cp.async` and STMatrix calls are generic-write events
when their writable shared-memory operand remains visible. LDSM and other
shared-memory loads do not create a generic-to-async write dependency. Unknown
calls without a visible shared write are state-neutral; the pass cannot infer
hidden side effects from arbitrary external code. Structured blocks, branches,
`for` loops, and `while` loops are handled by conservative state propagation.

## Usage

The pass is part of the default TileLang lowering pipeline. To apply it manually:

```python
import tilelang
import tilelang.language as T
import tvm

mod = tvm.IRModule({"main": prim_func})
with tvm.transform.PassContext():
    mod = tilelang.cuda.transform.InjectFenceProxy()(mod)
```

## End-to-End Example

Before the pass:

```python
@T.prim_func
def kernel():
    with T.Kernel(1):
        desc = T.decl_buffer((1,), "uint64", scope="local.descriptor")
        smem = T.decl_buffer((128,), "float16", scope="shared")
        T.initialize_wgmma_descriptor(desc, T.uint64(0), 2, 1, 32)
        smem[0] = T.float16(0)
        T.ptx_wgmma_ss(
            "float16",
            "m64n64k16",
            T.bool(True),
            T.bool(True),
            "fp16",
            "fp16",
            "fp16",
            desc.data,
            T.int32(0),
            desc.data,
            T.int32(0),
            smem.data,
            T.int32(0),
            T.bool(True),
            1,
            1,
        )
```

After `tl.cuda.transform.InjectFenceProxy`:

```python
@T.prim_func
def kernel():
    with T.Kernel(1):
        desc = T.decl_buffer((1,), "uint64", scope="local.descriptor")
        smem = T.decl_buffer((128,), "float16", scope="shared")
        T.initialize_wgmma_descriptor(desc, T.uint64(0), 2, 1, 32)
        smem[0] = T.float16(0)
        T.fence_proxy_async()
        T.ptx_wgmma_ss(
            "float16",
            "m64n64k16",
            T.bool(True),
            T.bool(True),
            "fp16",
            "fp16",
            "fp16",
            desc.data,
            T.int32(0),
            desc.data,
            T.int32(0),
            smem.data,
            T.int32(0),
            T.bool(True),
            1,
            1,
        )
```

The only change is the `fence_proxy_async` between the generic descriptor setup
/ shared-memory write and the async `wgmma`. In larger kernels the pass performs
the same operation across nested blocks, loops, and conditional branches.

## Extending the Pass

If you introduce a new intrinsic that uses the async proxy, add it to
`IsAsyncIntrinsic` in `src/cuda/transform/inject_fence_proxy.cc`. If a custom
operation writes shared memory, preserve a writable shared pointer in its
lowered arguments so `CallMayWriteSharedMemory` can recognize it, or add an
explicit classification.

Most otherwise unknown calls are state-neutral. For custom/opaque operations
whose proxy effects cannot be represented, lower them into known intrinsics or
manually insert `fence_proxy_async` at the generic-to-async transition.
