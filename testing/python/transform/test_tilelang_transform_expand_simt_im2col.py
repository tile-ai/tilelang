"""Early SIMT im2col: implementation selection, guards, and CUDA execution."""

import pytest
import torch

import tilelang
from tilelang import tvm
import tilelang.language as T
import tilelang.testing
from test_tilelang_transform_im2col_fallback import _make_im2col_kernel


def _runtime_simt_target():
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip("Execution coverage requires SM80 or newer")
    arch = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    if not tilelang.cuda.transform.UsesSIMTIm2Col(target):
        pytest.skip("This GPU selects a specialized im2col implementation")
    return target


def _expand(func, arch):
    func = func.with_attr("target", tvm.target.Target({"kind": "cuda", "arch": arch}))
    return tilelang.cuda.transform.ExpandSIMTIm2Col()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_100", "sm_120", "sm_120a"])
@tilelang.testing.requires_cuda
def test_target_gating_and_idempotence(arch):
    func = _make_im2col_kernel()
    mod = _expand(func, arch)
    assert tilelang.cuda.transform.UsesSIMTIm2Col(tvm.target.Target({"kind": "cuda", "arch": arch})) == (arch != "sm_90")
    if arch != "sm_90":
        assert "tl.tileop.im2col" not in mod.script()
        assert "im2col_m" in mod.script()
    else:
        tvm.ir.assert_structural_equal(mod["main"].without_attr("target"), func)
    tvm.ir.assert_structural_equal(tilelang.cuda.transform.ExpandSIMTIm2Col()(mod), mod)


def _extract(channels, stride, dilation, padding, offset, annotations=None, crop=False):
    n, h, w, kernel, bm, bk = 2, 7, 9, 3, 16, 32
    oh = (h + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    mt, kt = (n * oh * ow + bm - 1) // bm, (kernel * kernel * channels + bk - 1) // bk

    @T.prim_func
    def extract(data: T.Tensor((n, h, w, channels), "float16"), out: T.Tensor((mt, kt, bm, bk), "float16")):
        with T.Kernel(mt, kt, threads=128) as (mi, ki):
            shared = T.alloc_shared((bm + offset, bk), "float16")
            if crop:
                T.im2col(
                    data[:, 1:, :, :], shared[offset : offset + bm, :], mi, ki, kernel, stride, dilation, padding, annotations=annotations
                )
            else:
                T.im2col(data, shared[offset : offset + bm, :], mi, ki, kernel, stride, dilation, padding, annotations=annotations)
            T.copy(shared[offset : offset + bm, :], out[mi, ki, :, :])

    return extract, (n, h, w, channels), (oh, ow, mt, kt, bm, bk)


@pytest.mark.parametrize("annotations,crop", [({"test_preserve": 1}, False), (None, True)])
@tilelang.testing.requires_cuda
def test_unsupported_metadata_preserved(annotations, crop):
    func, _, _ = _extract(32, 1, 1, 1, 0, annotations, crop)
    mod = _expand(func, "sm_120a")
    tvm.ir.assert_structural_equal(mod["main"].without_attr("target"), func)


@tilelang.testing.requires_cuda
def test_manual_ws_schedule_preserved():
    func = _make_im2col_kernel()

    def annotate(node):
        if isinstance(node, tvm.tirx.SBlock):
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                {**node.annotations, "tl.ws_schedule": "preserve-sentinel"},
            )

    func = func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, annotate, None))
    mod = _expand(func, "sm_120a")
    tvm.ir.assert_structural_equal(mod["main"].without_attr("target"), func)


@pytest.mark.parametrize("arch", ["sm_90"])
@tilelang.testing.requires_cuda
def test_other_target_source_unchanged(arch):
    func = _make_im2col_kernel()
    target = {"kind": "cuda", "arch": arch}
    sources = []
    for enabled in (False, True):
        with tvm.target.Target(target), tvm.transform.PassContext(config={"tl.enable_early_simt_im2col": enabled}):
            sources.append(tilelang.lower(func, target=target).kernel_source)
    assert sources[0] == sources[1]
    if arch == "sm_90":
        assert "tma_load_im2col" in sources[1]


@pytest.mark.parametrize(
    "channels,stride,dilation,padding,offset", [(32, 1, 1, 1, 0), (48, 2, 2, 2, 0), (33, 1, 1, 1, 0), (65, 2, 1, 1, 3)]
)
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_simt_tail_and_destination_region(channels, stride, dilation, padding, offset):
    target = _runtime_simt_target()
    func, shape, dims = _extract(channels, stride, dilation, padding, offset)
    oh, ow, mt, kt, bm, bk = dims
    data = torch.randn(shape, device="cuda", dtype=torch.float16)
    cols = torch.nn.functional.unfold(data.permute(0, 3, 1, 2), 3, dilation=dilation, padding=padding, stride=stride)
    # torch unfolds C, KH, KW; TileLang im2col uses KH, KW, C.
    cols = cols.reshape(shape[0], channels, 3, 3, oh * ow).permute(0, 4, 2, 3, 1).reshape(-1, 9 * channels)
    padded = torch.zeros((mt * bm, kt * bk), device="cuda", dtype=torch.float16)
    padded[: cols.shape[0], : cols.shape[1]] = cols
    expected = padded.reshape(mt, bm, kt, bk).permute(0, 2, 1, 3).contiguous()
    kernel = tilelang.compile(
        func,
        out_idx=[1],
        target=target,
        pass_configs={"tl.enable_early_simt_im2col": True, "tl.disable_warp_specialized": True},
    )
    torch.testing.assert_close(kernel(data), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ws", [False, True])
@pytest.mark.parametrize("num_stages", [0, 1, 2, 3])
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_simt_convolution(ws, num_stages):
    target = _runtime_simt_target()
    if ws and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("WS requires TMA support")
    func = _make_im2col_kernel(channels=48, num_stages=num_stages)
    data = torch.randn((1, 8, 8, 48), device="cuda", dtype=torch.float16)
    weight = torch.randn((3, 3, 48, 32), device="cuda", dtype=torch.float16)
    expected = torch.nn.functional.conv2d(data.permute(0, 3, 1, 2), weight.permute(3, 2, 0, 1), padding=1).permute(0, 2, 3, 1)
    kernel = tilelang.compile(
        func,
        out_idx=[2],
        target=target,
        pass_configs={"tl.enable_early_simt_im2col": True, "tl.disable_warp_specialized": not ws},
    )
    torch.testing.assert_close(kernel(data, weight), expected, rtol=1e-2, atol=1e-2)
