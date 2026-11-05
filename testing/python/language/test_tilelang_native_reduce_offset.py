"""Offset reductions retain their owner threads through reduction and readback."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize(
    "offset,batch,clear",
    [(0, 1, True), (32, 1, True), (64, 1, True), (32, 2, True), (32, 2, False)],
)
def test_offset_scalar_and_batch_readback(offset, batch, clear):
    rows = 1 if batch == 1 else 2

    @tilelang.jit(
        out_idx=1,
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    def make_kernel():
        @T.prim_func
        def kernel(x: T.Tensor((rows, 512), "float32"), out: T.Tensor((rows,), "float32")):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((rows, 512), "float32")
                dst = T.alloc_fragment((rows,), "float32")
                T.annotate_layout(
                    {
                        src: T.Fragment((rows, 512), forward_fn=lambda i, j: (offset + j // 8, i * 8 + j % 8)),
                        dst: T.Fragment((rows,), replicate=64, forward_fn=lambda i, rep: (offset + rep, i)),
                    }
                )
                for i, j in T.Parallel(rows, 512):
                    src[i, j] = x[i, j]
                if not clear:
                    T.fill(dst, 1.0)
                T.reduce_sum(src, dst, dim=1, clear=clear, batch=batch)
                for i in T.Parallel(rows):
                    out[i] = dst[i]

        return kernel

    x = torch.arange(rows * 512, device="cuda", dtype=torch.float32).reshape(rows, 512)
    actual = make_kernel()(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, x.sum(dim=1) + (0 if clear else 1), rtol=0, atol=0)


def test_scalar_gapped_owner_image_is_rejected_during_lowering():
    # This replica order matches ComputeReducerLayout and leaves the same
    # 32-thread hole as the source layout, so rejection tests the barrier range.
    owners = {32 + (rep % 2) * 64 + rep // 2 for rep in range(64)}
    assert owners == set(range(32, 64)) | set(range(96, 128))
    assert len(owners) == 64 and max(owners) - min(owners) + 1 == 96

    @T.prim_func
    def kernel(unused: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((2, 512), "float32")
            dst = T.alloc_fragment((2,), "float32")
            T.annotate_layout(
                {
                    src: T.Fragment(
                        (2, 512),
                        forward_fn=lambda i, j: (32 + (j // 256) * 64 + (j // 8) % 32, i * 8 + j % 8),
                    ),
                    dst: T.Fragment((2,), replicate=64, forward_fn=lambda i, rep: (32 + (rep % 2) * 64 + rep // 2, i)),
                }
            )
            # No device execution: lowering must reject the incomplete
            # participating thread interval before reading this fragment.
            T.reduce_sum(src, dst, dim=1, batch=1)

    with pytest.raises(Exception, match="dense physical slots over one contiguous thread range"):
        tilelang.lower(kernel, target={"kind": "cuda", "arch": "sm_100"})
