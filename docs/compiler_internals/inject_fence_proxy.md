# InjectFenceProxy Pass

`tl.InjectFenceProxy` is a TIR-level transform that keeps the GPU proxy state consistent on NVIDIA Hopper (SM90+) by inserting `fence.proxy.async` instructions when control flow switches from generic memory operations to asynchronous proxy operations.

## Why Fences Are Needed

Hopper separates memory instructions into generic and asynchronous proxy paths. When an asynchronous instruction (for example, `cp.async` or `tma.load`) issues after generic traffic (like `ldmatrix` or plain buffer stores), the hardware requires a `fence.proxy.async` to guarantee ordering. Missing fences can lead to race conditions or undefined behaviour.

## What the Pass Does

- Walks every statement in the `PrimFunc`, tracking whether it behaves as a **generic**, **async**, or **neutral** proxy (neutral statements reset the state, such as an explicit fence).
- Automatically lowers `tma_store` intrinsics into the required `arrive`/`wait` handshake so that TMA stores participate correctly in synchronization.
- Injects an explicit `fence.proxy.async` whenever a generic statement is followed by an async statement without an intervening neutral barrier.

The pass is conservative: unknown extern calls are treated as async so that the fence is inserted rather than accidentally omitted.

## Coverage of Intrinsics

The tracker understands the TileLang intrinsics for TMA load/store, shared-memory MMA (`wgmma`), and TVM/PTX async copy intrinsics (`cp.async` variants). Generic operations currently include `ldmatrix`, `stmatrix`, and descriptor initialization. Other IR nodes (loops, blocks, attributes) receive a proxy kind derived from their bodies so that the analysis survives structured control flow.

## Usage

The pass is part of the default TileLang lowering pipeline. To apply it manually:

```python
from tilelang import tl
from tvm import IRModule

mod = IRModule({"main": prim_func})
with tvm.transform.PassContext():
    mod = tl.transform.InjectFenceProxy()(mod)
```

## Extending the Pass

If you introduce a new intrinsic that behaves like an async proxy, add it to `IsAsyncIntrinsic` in `src/transform/inject_fence_proxy.cc`. Likewise, extend `IsKnownGeneric` for additional generic operations. When adding new neutral barriers, make sure they set the proxy kind to `kNeutral` so the state resets correctly.
