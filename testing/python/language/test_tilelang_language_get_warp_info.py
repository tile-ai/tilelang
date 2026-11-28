import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def _all_warp_info_kernel():
    @T.prim_func
    def main(
        A0: T.Tensor((128,), T.int32),
        A1: T.Tensor((256,), T.int32),
        A2: T.Tensor((128,), T.int32),
        A3: T.Tensor((256,), T.int32),
        A4: T.Tensor((128,), T.int32),
        A5: T.Tensor((320,), T.int32),
        A6: T.Tensor((128,), T.int32),
        A7: T.Tensor((512,), T.int32),
        A8: T.Tensor((256,), T.int32),
        A9: T.Tensor((128,), T.int32),
    ):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            A0[tx] = T.get_lane_idx(None)
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            A1[tx] = T.get_lane_idx(64)
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            A2[tx] = T.get_warp_idx_sync(None)
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            A3[tx] = T.get_warp_idx_sync(16)
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            A4[tx] = T.get_warp_idx(None)
        with T.Kernel(1, threads=320):
            tx = T.get_thread_binding()
            A5[tx] = T.get_warp_idx(20)
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            A6[tx] = T.get_warp_group_idx(None, None)
        with T.Kernel(1, threads=512):
            tx = T.get_thread_binding()
            A7[tx] = T.get_warp_group_idx(32, 5)
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            A8[tx] = T.shuffle_elect(64)
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            A9[tx] = T.shuffle_elect(0)

    return main


@tilelang.testing.requires_cuda
def test_get_warp_info():
    kernel = tilelang.compile(_all_warp_info_kernel(), out_idx=list(range(10)))
    results = kernel()
    indices = [torch.arange(size, dtype=torch.int32, device="cuda") for size in (128, 256, 128, 256, 128, 320, 128, 512, 256, 128)]
    expected = [
        indices[0] % 32,
        indices[1] % 64,
        indices[2] // 32,
        indices[3] // 16,
        indices[4] // 32,
        indices[5] // 20,
        indices[6] // (32 * 4),
        indices[7] // (32 * 5),
        (indices[8] % 64 == 0).to(torch.int32),
        (indices[9] == 0).to(torch.int32),
    ]
    for result, reference in zip(results, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
