# InjectFenceProxy Pass

`tl.InjectFenceProxy` is a TIR-level transform that keeps the GPU proxy state consistent on NVIDIA Hopper (SM90+) by inserting `fence.proxy.async` instructions when execution switches from **generic proxy** memory operations to **async proxy** operations.

## Why Fences Are Needed

Hopper separates memory instructions into generic and asynchronous proxy paths. When an asynchronous instruction (for example, `wgmma`, `tma.load`, or `cp.async.bulk`) issues after generic traffic (like `ldmatrix`, `cp.async`, or **shared-memory** buffer stores), the hardware requires a `fence.proxy.async` to guarantee ordering. Missing fences can lead to race conditions or undefined behavior.

## What the Pass Does

- Walks statements in execution order while tracking a (may-)state of the last proxy kind (**generic**, **async**, or **none/reset**). Control-flow joins (e.g. `if`) merge states conservatively.
- Normalizes `tma_store` by ensuring the required `tma_store_arrive` / `tma_store_wait` handshake exists immediately after the store.
- Injects `fence.proxy.async` right before an async-proxy instruction whenever the preceding state can be generic.

By default, unknown/external calls do **not** affect proxy state. Opaque calls that may write into **shared memory** (e.g. via `tvm_access_ptr` / `address_of`) are treated as generic proxy traffic so a later async-proxy op will still be fenced.

### Timeline View

```
generic shared-store (or ldmatrix/stmatrix/cp.async) → async op (wgmma / tma / cp.async.bulk)
                 │                                  │
                 └─ generic proxy                   └─ async proxy
                             │        fence inserted here   ↑
                             └──────────────────────────────┘
```

The proxy tracker effectively scans the program in execution order. The moment it detects a possible transition from generic to async (between the store and the async op above), it synthesizes a `fence.proxy.async` to reset the hardware proxy state before the async path runs.

## Coverage of Intrinsics

The tracker understands the TileLang intrinsics for TMA load/store, shared-memory MMA (`wgmma`), and TVM/PTX SM90 async copy intrinsics (`cp.async.bulk` family). Generic operations currently include `ldmatrix`, `stmatrix`, `cp.async`, and **shared-memory** `BufferStore` statements. Structured control flow (loops, blocks, branches) is handled by propagating and conservatively merging proxy state.

## Usage

The pass is part of the default TileLang lowering pipeline. To apply it manually:

```python
from tilelang import tl
from tvm import IRModule

mod = IRModule({"main": prim_func})
with tvm.transform.PassContext():
    mod = tl.transform.InjectFenceProxy()(mod)
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

After `tl.transform.InjectFenceProxy`:

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

The only change is the `fence_proxy_async` between the generic descriptor setup / shared-memory write and the async `wgmma`. In larger kernels the pass performs the same operation across nested blocks, loops, and conditional branches.

## Extending the Pass

If you introduce a new intrinsic that behaves like an async proxy, add it to `IsAsyncIntrinsic` in `src/transform/inject_fence_proxy.cc`. Likewise, extend `IsKnownGeneric` for additional generic operations.

Most calls default to `"none"` (no proxy-state effect). `IsNonProxyIntrinsic` exists for well-known synchronization / scheduling helpers and to document intent, but it is not required for correctness if an op is neither generic nor async.
For custom/opaque ops, you must lower them into known intrinsics (or manually insert `fence_proxy_async`) if they participate in proxy switching.
