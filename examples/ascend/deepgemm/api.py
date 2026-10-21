"""BF16 and FP8 tensor APIs backed by TileLang kernels.

The API suffix describes the logical expression. Physical K/MN major is
derived from tensor strides. Dimensions omitted from ``compiled_dims`` and
all outer strides stay in the runtime launch ABI.
Dense GEMMs accept ``alpha=None`` or a runtime FP32 multiplier for A @ B,
applied before output conversion and accumulation into C. Dense C must alias D
for accumulation; a separate contiguous [1, N] C selects fused bias instead.

Grouped layouts contain logical end offsets; each group starts at the previous
end rounded up to ``MK_ALIGNMENT``. M-grouped A/D must include the full aligned
row allocation. ``ensure_zero_padding`` does not clear inputs: zero output
padding requires zero input padding and valid scales.
Prepacked int16 scales are used directly, including caller-provided padding.
"""

from __future__ import annotations

import enum
import warnings

import torch

from . import epilogue as epilogue_ops
from .config import GemmDesc, GemmType, Major, ceil_div, select_gemm_config, select_transform_sf_config, execution_config
from .kernels import build_bf16_gemm, build_fp8_gemm
from .kernels.common import MK_ALIGNMENT
from .kernels.transform_sf import build_transform_sf


class BindingLayout(str, enum.Enum):
    """Logical transpose suffix used by the public GEMM bindings."""

    NT = "nt"
    NN = "nn"
    TN = "tn"
    TT = "tt"

    @property
    def trans_a(self) -> bool:
        return self.value[0] == "t"

    @property
    def trans_b(self) -> bool:
        return self.value[1] == "n"


def _check_quantized_pair(a, b):
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        # TODO: the heuristic models FP4/dequant, but those kernels are separate work.
        raise NotImplementedError("Only FP8 x FP8 kernels are implemented; FP4/dequant are TODO")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float8_e4m3fn:
        return "float8_e4m3fn"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise TypeError(f"d must be bfloat16, float32 or float8_e4m3fn, got {dtype}")


def _same_device(*tensors: torch.Tensor | None) -> None:
    devices = {tensor.device for tensor in tensors if tensor is not None}
    if len(devices) != 1:
        raise ValueError(f"all tensors must be on the same device, got {sorted(map(str, devices))}")
    device = next(iter(devices))
    if device.type != "npu":
        raise ValueError(f"DeepGEMM TileLang APIs require NPU tensors, got {device}")


def _get_major_type_ab(tensor: torch.Tensor, mn_axis: str) -> Major:
    stride_mn, stride_k = tensor.stride(-2), tensor.stride(-1)
    if stride_mn == 1 and stride_k == 1:
        raise ValueError(f"{mn_axis} operand has an ambiguous major axis, got strides={tensor.stride()}")
    if stride_k == 1:
        return Major.K
    if stride_mn == 1:
        return Major.MN
    raise ValueError(f"{mn_axis} operand requires unit stride on its logical MN or K axis, got strides={tensor.stride()}")


def _check_major_type_cd(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have unit stride along N, got {tensor.stride()}")


def _num_cores(device: torch.device) -> int:
    index = torch.npu.current_device() if device.index is None else device.index
    return int(torch.npu.get_device_properties(index).cube_core_num)


def _copy_source(d: torch.Tensor, c: torch.Tensor | None) -> None:
    if c is not None and c.data_ptr() != d.data_ptr():
        d.copy_(c)


def _gemm_early_return(m: int, n: int, k: int, d: torch.Tensor, c: torch.Tensor | None, *, with_bias=False) -> bool:
    if m == 0 or n == 0:
        return True
    if c is not None:
        _check_major_type_cd(c, "c")
        if c.data_ptr() == d.data_ptr() and c.stride() != d.stride():
            raise ValueError("aliased c and d must have identical strides")
    if k == 0:
        if c is None:
            (d.view(torch.uint8) if d.dtype == torch.float8_e4m3fn else d).zero_()
        else:
            _copy_source(d, c)
        return True
    if not with_bias:
        _copy_source(d, c)
    return False


def _get_outer_stride(tensor: torch.Tensor, major: Major) -> int:
    return tensor.stride(-2) if major == Major.K else tensor.stride(-1)


def _expected_k_from_ks(ks_cpu, default_k: int) -> int:
    if ks_cpu is not None and len(ks_cpu):
        return max(1, sum(int(value) for value in ks_cpu) // len(ks_cpu))
    # A nonempty physical K allocation can contain fewer elements than groups.
    return max(1, default_k)


def _get_gemm_desc(
    gemm_type: GemmType,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor | None = None,
    *,
    grouped_layout: torch.Tensor | None = None,
    expected_m: int | None = None,
    ks_cpu=None,
    epilogue: epilogue_ops.Epilogue,
    compiled_dims: str = "nk",
) -> GemmDesc | None:
    """Validate logical MK/NK operands and prepare the shared heuristic metadata.

    Bindings normalize logical axes before this call while preserving physical
    strides and storage.
    """
    batched = gemm_type == GemmType.Batched
    m_grouped = gemm_type == GemmType.MGroupedContiguousWithPsumLayout
    k_grouped = gemm_type == GemmType.KGroupedContiguousWithPsumLayout
    expected_ndims = (3 if batched else 2, 3 if batched or m_grouped else 2, 3 if batched or k_grouped else 2)
    if (a.ndim, b.ndim, d.ndim) != expected_ndims:
        raise ValueError(f"{gemm_type.name} a/b/d must have ranks {expected_ndims}, got {(a.ndim, b.ndim, d.ndim)}")
    _same_device(a, b, c, d, grouped_layout)
    out_dtype = _dtype_name(d.dtype)
    major_a, major_b = _get_major_type_ab(a, "a"), _get_major_type_ab(b, "b")
    _check_major_type_cd(d, "d")
    m, k = a.shape[-2:]
    n, b_k = b.shape[-2:]
    if batched:
        num_groups = a.shape[0]
    elif m_grouped:
        num_groups = b.shape[0]
    elif k_grouped:
        num_groups = d.shape[0]
    else:
        num_groups = 0
    output_shape = (num_groups, m, n) if batched or k_grouped else (m, n)
    if b_k != k or tuple(d.shape) != output_shape or (batched and b.shape[0] != num_groups):
        raise ValueError(f"incompatible {gemm_type.name} shapes: a={tuple(a.shape)}, b={tuple(b.shape)}, d={tuple(d.shape)}")
    with_bias = epilogue.kind == "bias"
    if c is not None and (tuple(c.shape) != ((1, n) if with_bias else tuple(d.shape)) or c.dtype != d.dtype):
        raise ValueError("c must have D's dtype and shape, or be a separate contiguous [1, N] bias for dense GEMM")
    if m_grouped or k_grouped:
        if (
            grouped_layout is None
            or grouped_layout.ndim != 1
            or grouped_layout.dtype is not torch.int32
            or not grouped_layout.is_contiguous()
        ):
            raise ValueError("grouped_layout must be a contiguous int32 vector")
        if grouped_layout.numel() != num_groups:
            raise ValueError(f"grouped_layout must have {num_groups} entries, got {grouped_layout.numel()}")
        if m_grouped and major_a != Major.K:
            raise ValueError("M-grouped A must be K-major")
        if k_grouped and (major_a != Major.MN or major_b != Major.MN):
            raise ValueError("K-grouped A and B must be MN-major after logical transpose")
    if (batched or m_grouped or k_grouped) and num_groups == 0:
        if m_grouped and d.numel():
            raise ValueError("nonempty M-grouped GEMM requires at least one group")
        return None
    if m_grouped and m and n and k and m % MK_ALIGNMENT:
        raise ValueError(f"M-grouped A/D must include physical padding to {MK_ALIGNMENT} rows, got M={m}")
    if _gemm_early_return(m, n, k, d, c, with_bias=with_bias):
        if k == 0 and m and n and isinstance(epilogue, epilogue_ops.FP8Quantization):
            epilogue.sfd.zero_()
        return None

    estimated_m = m // num_groups if m_grouped else m
    expected_k = _expected_k_from_ks(ks_cpu, k // num_groups) if k_grouped else k
    return GemmDesc(
        m=m,
        n=n,
        k=k,
        a_dtype=str(a.dtype).removeprefix("torch."),
        b_dtype=str(b.dtype).removeprefix("torch."),
        cd_dtype=out_dtype,
        major_a=major_a,
        major_b=major_b,
        acc=c is not None and not with_bias,
        gemm_type=gemm_type,
        num_groups=num_groups,
        expected_m=estimated_m if expected_m is None else int(expected_m),
        expected_k=expected_k,
        num_cores=_num_cores(d.device),
        epilogue_kind=epilogue.kind,
        outer_stride_a=_get_outer_stride(a, major_a),
        outer_stride_b=_get_outer_stride(b, major_b),
        outer_stride_d=d.stride(-2),
        output_alignment=d.data_ptr() % 32 if d.device.type != "meta" else 0,
        compiled_dims=compiled_dims,
        stride_batch_a=a.stride(0) if batched else 0,
        stride_batch_b=b.stride(0) if batched else 0,
    )


def _physical(logical: torch.Tensor, major: Major) -> torch.Tensor:
    return logical if major == Major.K else logical.transpose(-2, -1)


def _compiled_shape(m: int, n: int, k: int, compiled_dims: str):
    return tuple(dim if name in compiled_dims else None for name, dim in zip(("m", "n", "k"), (m, n, k)))


def _fp8_operand(operand, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(operand, (tuple, list)) or len(operand) != 2:
        raise TypeError(f"{name} must be a (data, scale) pair")
    data, scale = operand
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} data and scale must be tensors")
    if data.device != scale.device:
        raise ValueError(f"{name} data and scale must be on the same device")
    return data, scale


def _scale_recipe(recipe, name: str) -> tuple[int, int]:
    if not isinstance(recipe, (tuple, list)) or len(recipe) != 2:
        raise ValueError(f"{name} must be a (gran_mn, gran_k) pair")
    gran_mn, gran_k = (int(value) for value in recipe)
    if gran_mn <= 0 or gran_k <= 0:
        raise ValueError(f"{name} entries must be positive, got {tuple(recipe)}")
    if gran_k != 32:
        raise ValueError(f"FP8 requires gran_k=32, got {gran_k}")
    return gran_mn, gran_k


def _get_recipe(recipe, recipe_a, recipe_b) -> tuple[int, int, int]:
    if recipe is not None:
        if recipe_a is not None or recipe_b is not None:
            raise ValueError("recipe is mutually exclusive with recipe_a and recipe_b")
        if not isinstance(recipe, (tuple, list)) or len(recipe) != 3:
            raise ValueError("recipe must be a (gran_m, gran_n, gran_k) tuple")
        gran_m, gran_n, gran_k = (int(value) for value in recipe)
        _scale_recipe((gran_m, gran_k), "recipe_a")
        _scale_recipe((gran_n, gran_k), "recipe_b")
        return gran_m, gran_n, gran_k
    gran_m, gran_k_a = _scale_recipe((1, 32) if recipe_a is None else recipe_a, "recipe_a")
    gran_n, gran_k_b = _scale_recipe((1, 32) if recipe_b is None else recipe_b, "recipe_b")
    if gran_k_a != gran_k_b:
        raise ValueError("gran_k of recipe_a and recipe_b must be the same")
    return gran_m, gran_n, gran_k_a


def _check_sf_layout(
    scale: torch.Tensor,
    mn: int,
    k: int,
    gran_mn: int,
    gran_k: int,
    num_groups: int | None,
) -> bool:
    expected_ndim = 3 if num_groups is not None else 2
    if scale.dtype not in (torch.float32, torch.int16):
        raise TypeError(f"FP8 scale must be float32 or int16, got {scale.dtype}")
    if scale.ndim != expected_ndim:
        raise ValueError(f"FP8 scale must have {expected_ndim} dimensions, got shape={tuple(scale.shape)}")
    if scale.stride(-1) != 1 and scale.stride(-2) != 1:
        raise ValueError(f"FP8 scale must be contiguous in either the last or second-to-last dimension, got strides={scale.stride()}")
    if num_groups is not None and scale.shape[-3] != num_groups:
        raise ValueError(f"FP8 scale batch/group dimension must be {num_groups}, got {scale.shape[-3]}")

    expected_mn = ceil_div(mn, gran_mn)
    k_divisor = gran_k if scale.dtype is torch.float32 else gran_k * 2
    expected_tail = (expected_mn, ceil_div(k, k_divisor))
    if tuple(scale.shape[-2:]) != expected_tail:
        raise ValueError(f"FP8 scale trailing shape must be {expected_tail}, got {tuple(scale.shape[-2:])}")

    return (
        scale.dtype is torch.int16
        and gran_mn == 1
        and scale.stride(-2) == 1
        and scale.stride(-1) == scale.shape[-2]
        and (num_groups is None or scale.stride(-3) == scale.shape[-2] * scale.shape[-1])
    )


def _transform_sf_into_required_layout(
    scale: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int, int],
    *,
    is_sfa: bool,
    num_groups: int | None = None,
    disable_ue8m0_cast: bool = False,
    grouped_layout: torch.Tensor | None = None,
    k_grouped: bool = False,
) -> torch.Tensor:
    """Pack scales into MN-major int16; invalid FP32 bits trigger a device assert."""

    gran_mn, gran_k = recipe[0 if is_sfa else 1], recipe[2]
    if _check_sf_layout(scale, mn, k, gran_mn, gran_k, num_groups):
        return scale

    if gran_mn > 128 or gran_mn & (gran_mn - 1):
        raise ValueError(f"gran_mn must be a power of two in [1, 128], got {gran_mn}")
    if scale.dtype is torch.float32 and disable_ue8m0_cast:
        raise ValueError("float32 scale conversion is disabled by disable_ue8m0_cast=True")

    source = scale.unsqueeze(0) if scale.ndim == 2 else scale
    output = torch.empty((source.shape[0], ceil_div(k, 64), mn), dtype=torch.int16, device=scale.device)
    if output.numel():
        major = Major.K if source.stride(-1) == 1 else Major.MN
        is_float = scale.dtype == torch.float32
        gemm_type = GemmType.Batched if num_groups is not None else GemmType.Normal
        if grouped_layout is not None:
            gemm_type = GemmType.KGroupedContiguousWithPsumLayout if k_grouped else GemmType.MGroupedContiguousWithPsumLayout
        groups = grouped_layout if grouped_layout is not None else torch.empty(0, dtype=torch.int32, device=scale.device)
        input_stride_bytes = source.stride(-2 if major == Major.K else -1) * source.element_size()
        config = select_transform_sf_config(
            is_float, major, source.shape, gemm_type, gran_mn, 2 * _num_cores(scale.device), input_stride_bytes
        )
        kernel = build_transform_sf(is_float, major, config, gemm_type, MK_ALIGNMENT, gran_mn)
        kernel(source if major == Major.K else source.transpose(-2, -1), groups, output)
    result = output.transpose(-2, -1)
    return result.squeeze(0) if scale.ndim == 2 else result


def _prepare_scales(desc, sfa, sfb, recipe, *, grouped_layout=None, disable_ue8m0_cast=False):
    """Normalize each operand's scales, including its batch/group padding policy."""
    batched = desc.gemm_type == GemmType.Batched
    m_grouped = desc.gemm_type == GemmType.MGroupedContiguousWithPsumLayout
    k_grouped = desc.gemm_type == GemmType.KGroupedContiguousWithPsumLayout
    scales = []
    for scale, mn, is_sfa in ((sfa, desc.m, True), (sfb, desc.n, False)):
        transformed = _transform_sf_into_required_layout(
            scale,
            mn,
            desc.k,
            recipe,
            is_sfa=is_sfa,
            num_groups=desc.num_groups if batched or (m_grouped and not is_sfa) else None,
            disable_ue8m0_cast=disable_ue8m0_cast,
            grouped_layout=grouped_layout if k_grouped or (m_grouped and is_sfa) else None,
            k_grouped=k_grouped,
        )
        scales.append(transformed.transpose(-2, -1))
    return tuple(scales)


def _launch_gemm(desc, a, b, d, epilogue, *, scales=(), grouped_layout=None):
    config = execution_config(desc, select_gemm_config(desc))
    build = build_fp8_gemm if scales else build_bf16_gemm
    kernel = build(
        *_compiled_shape(desc.m, desc.n, desc.k, desc.compiled_dims),
        desc.major_a,
        desc.major_b,
        desc.cd_dtype,
        desc.acc,
        config,
        desc.epilogue_kind,
        desc.gemm_type,
    )
    args = (_physical(a, desc.major_a), _physical(b, desc.major_b), *scales, d)
    if grouped_layout is not None:
        args += (grouped_layout,)
    kernel(*args, *epilogue_ops.arguments(epilogue, dense=desc.gemm_type == GemmType.Normal))


def _make_bf16_gemm(layout: BindingLayout):
    trans_a, trans_b = layout.trans_a, layout.trans_b

    def bf16_gemm(a, b, d, c=None, compiled_dims="mn" if trans_a else "nk", alpha=None, epilogue=None) -> None:
        if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
            raise TypeError(f"a and b must be bfloat16, got {a.dtype} and {b.dtype}")
        d, resolved = epilogue_ops.resolve(d, c, alpha, epilogue, dense=True)
        logical_a = a.transpose(0, 1) if trans_a else a
        logical_b = b.transpose(0, 1) if trans_b else b
        desc = _get_gemm_desc(
            GemmType.Normal,
            logical_a,
            logical_b,
            d,
            c,
            epilogue=resolved,
            compiled_dims=compiled_dims,
        )
        if desc is None:
            return
        _launch_gemm(desc, logical_a, logical_b, d, resolved)

    bf16_gemm.__name__ = bf16_gemm.__qualname__ = f"bf16_gemm_{layout.value}"
    return bf16_gemm


bf16_gemm_nt = _make_bf16_gemm(BindingLayout.NT)
bf16_gemm_nn = _make_bf16_gemm(BindingLayout.NN)
bf16_gemm_tn = _make_bf16_gemm(BindingLayout.TN)
bf16_gemm_tt = _make_bf16_gemm(BindingLayout.TT)


def _make_fp8_gemm(layout: BindingLayout):
    trans_a, trans_b = layout.trans_a, layout.trans_b

    def fp8_gemm(
        a,
        b,
        d,
        c=None,
        recipe=None,
        recipe_a=None,
        recipe_b=None,
        compiled_dims="mn" if trans_a else "nk",
        disable_ue8m0_cast=False,
        alpha=None,
        epilogue=None,
    ) -> None:
        a_data, sfa = _fp8_operand(a, "a")
        b_data, sfb = _fp8_operand(b, "b")
        _check_quantized_pair(a_data, b_data)
        d, resolved = epilogue_ops.resolve(d, c, alpha, epilogue, dense=True)
        logical_a = a_data.transpose(0, 1) if trans_a else a_data
        logical_b = b_data.transpose(0, 1) if trans_b else b_data
        logical_sfa = sfa.transpose(0, 1) if trans_a else sfa
        logical_sfb = sfb.transpose(0, 1) if trans_b else sfb
        desc = _get_gemm_desc(
            GemmType.Normal,
            logical_a,
            logical_b,
            d,
            c,
            epilogue=resolved,
            compiled_dims=compiled_dims,
        )
        if desc is None:
            return
        recipe = _get_recipe(recipe, recipe_a, recipe_b)
        scales = _prepare_scales(desc, logical_sfa, logical_sfb, recipe, disable_ue8m0_cast=disable_ue8m0_cast)
        _launch_gemm(desc, logical_a, logical_b, d, resolved, scales=scales)

    fp8_gemm.__name__ = fp8_gemm.__qualname__ = f"fp8_gemm_{layout.value}"
    return fp8_gemm


fp8_gemm_nt = _make_fp8_gemm(BindingLayout.NT)
fp8_gemm_nn = _make_fp8_gemm(BindingLayout.NN)
fp8_gemm_tn = _make_fp8_gemm(BindingLayout.TN)
fp8_gemm_tt = _make_fp8_gemm(BindingLayout.TT)


def bf16_bmm(a, b, d, c=None, compiled_dims="nk", epilogue=None) -> None:
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(f"a and b must be bfloat16, got {a.dtype} and {b.dtype}")
    d, resolved = epilogue_ops.resolve(d, c, epilogue=epilogue)
    desc = _get_gemm_desc(
        GemmType.Batched,
        a,
        b,
        d,
        c,
        epilogue=resolved,
        compiled_dims=compiled_dims,
    )
    if desc is None:
        return
    _launch_gemm(desc, a, b, d, resolved)


def fp8_bmm(a, b, d, c=None, recipe=(1, 1, 32), compiled_dims="nk", epilogue=None) -> None:
    a_data, sfa = _fp8_operand(a, "a")
    b_data, sfb = _fp8_operand(b, "b")
    if a_data.dtype != torch.float8_e4m3fn or b_data.dtype != torch.float8_e4m3fn:
        raise TypeError("fp8_bmm supports FP8/FP8 inputs; FP4/dequant kernels are TODO")
    d, resolved = epilogue_ops.resolve(d, c, epilogue=epilogue, allow_fp8=True)
    desc = _get_gemm_desc(
        GemmType.Batched,
        a_data,
        b_data,
        d,
        c,
        epilogue=resolved,
        compiled_dims=compiled_dims,
    )
    if desc is None:
        return
    recipe = _get_recipe(recipe, None, None)
    scales = _prepare_scales(desc, sfa, sfb, recipe)
    _launch_gemm(desc, a_data, b_data, d, resolved, scales=scales)


def einsum(expr, a, b, d, c=None, use_cublaslt=False, epilogue=None) -> None:
    if use_cublaslt:
        raise RuntimeError("einsum: use_cublaslt is not implemented on Ascend")
    if expr == "bhr,hdr->bhd":
        bf16_bmm(
            a.permute(1, 0, 2),
            b,
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            "nk",
            epilogue=epilogue,
        )
    elif expr == "bhd,hdr->bhr":
        bf16_bmm(
            a.permute(1, 0, 2),
            b.permute(0, 2, 1),
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            "nk",
            epilogue=epilogue,
        )
    elif expr == "bhd,bhr->hdr":
        bf16_bmm(
            a.permute(1, 2, 0),
            b.permute(1, 2, 0),
            d,
            c,
            "mn",
            epilogue=epilogue,
        )
    else:
        raise ValueError(f"unsupported einsum expression: {expr!r}")


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 1, 32), epilogue=None) -> None:
    if isinstance(d, (tuple, list)):
        if len(d) != 2 or epilogue is not None:
            raise ValueError("The FP8 output pair and epilogue are mutually exclusive")
        d, sfd = d
        epilogue = epilogue_ops.FP8Quantization(sfd)
    a_data, sfa = _fp8_operand(a, "a")
    b_data, sfb = _fp8_operand(b, "b")

    def permute_pair(data, scale, dims):
        return data.permute(*dims), scale.permute(*dims)

    if expr == "bhr,hdr->bhd":
        fp8_bmm(
            permute_pair(a_data, sfa, (1, 0, 2)),
            (b_data, sfb),
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            recipe,
            "nk",
            epilogue=epilogue,
        )
    elif expr == "bhd,hdr->bhr":
        fp8_bmm(
            permute_pair(a_data, sfa, (1, 0, 2)),
            permute_pair(b_data, sfb, (0, 2, 1)),
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            recipe,
            "nk",
            epilogue=epilogue,
        )
    elif expr == "bhd,bhr->hdr":
        fp8_bmm(
            permute_pair(a_data, sfa, (1, 2, 0)),
            permute_pair(b_data, sfb, (1, 2, 0)),
            d,
            c,
            recipe,
            "mn",
            epilogue=epilogue,
        )
    else:
        raise ValueError(f"unsupported FP8 einsum expression: {expr!r}")


def _make_m_grouped_bf16_gemm_contiguous(layout: BindingLayout):
    trans_b = layout.trans_b

    def m_grouped_bf16_gemm_contiguous(
        a,
        b,
        d,
        grouped_layout,
        compiled_dims="nk",
        use_psum_layout=True,
        ensure_zero_padding=True,
        expected_m_for_psum_layout=None,
        epilogue=None,
    ) -> None:
        del ensure_zero_padding  # Input padding is supplied by the caller.
        if not use_psum_layout:
            raise ValueError("M-grouped BF16 GEMM requires use_psum_layout=True")
        if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
            raise TypeError("a and b must be bfloat16")
        d, resolved = epilogue_ops.resolve(d, epilogue=epilogue)
        logical_b = b.transpose(1, 2) if trans_b else b
        desc = _get_gemm_desc(
            GemmType.MGroupedContiguousWithPsumLayout,
            a,
            logical_b,
            d,
            grouped_layout=grouped_layout,
            expected_m=expected_m_for_psum_layout,
            epilogue=resolved,
            compiled_dims=compiled_dims,
        )
        if desc is None:
            return
        _launch_gemm(desc, a, logical_b, d, resolved, grouped_layout=grouped_layout)

    m_grouped_bf16_gemm_contiguous.__name__ = m_grouped_bf16_gemm_contiguous.__qualname__ = f"m_grouped_bf16_gemm_{layout.value}_contiguous"
    return m_grouped_bf16_gemm_contiguous


m_grouped_bf16_gemm_nt_contiguous = _make_m_grouped_bf16_gemm_contiguous(BindingLayout.NT)
m_grouped_bf16_gemm_nn_contiguous = _make_m_grouped_bf16_gemm_contiguous(BindingLayout.NN)


def _make_m_grouped_fp8_gemm_contiguous(layout: BindingLayout):
    trans_b = layout.trans_b

    def m_grouped_fp8_gemm_contiguous(
        a,
        b,
        d,
        grouped_layout,
        recipe=None,
        recipe_a=None,
        recipe_b=None,
        compiled_dims="nk",
        disable_ue8m0_cast=False,
        use_psum_layout=True,
        ensure_zero_padding=True,
        expected_m_for_psum_layout=None,
        epilogue=None,
    ) -> None:
        del ensure_zero_padding  # Input padding is supplied by the caller.
        if not use_psum_layout:
            raise ValueError("M-grouped FP8 GEMM requires use_psum_layout=True")
        a_data, sfa = _fp8_operand(a, "a")
        b_data, sfb = _fp8_operand(b, "b")
        _check_quantized_pair(a_data, b_data)
        d, resolved = epilogue_ops.resolve(d, epilogue=epilogue)
        logical_b = b_data.transpose(1, 2) if trans_b else b_data
        logical_sfb = sfb.transpose(1, 2) if trans_b else sfb
        desc = _get_gemm_desc(
            GemmType.MGroupedContiguousWithPsumLayout,
            a_data,
            logical_b,
            d,
            grouped_layout=grouped_layout,
            expected_m=0 if expected_m_for_psum_layout is None else expected_m_for_psum_layout,
            epilogue=resolved,
            compiled_dims=compiled_dims,
        )
        if desc is None:
            return
        recipe = _get_recipe(recipe, recipe_a, recipe_b)
        scales = _prepare_scales(desc, sfa, logical_sfb, recipe, grouped_layout=grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast)
        _launch_gemm(desc, a_data, logical_b, d, resolved, scales=scales, grouped_layout=grouped_layout)

    m_grouped_fp8_gemm_contiguous.__name__ = m_grouped_fp8_gemm_contiguous.__qualname__ = f"m_grouped_fp8_gemm_{layout.value}_contiguous"
    return m_grouped_fp8_gemm_contiguous


m_grouped_fp8_gemm_nt_contiguous = _make_m_grouped_fp8_gemm_contiguous(BindingLayout.NT)
m_grouped_fp8_gemm_nn_contiguous = _make_m_grouped_fp8_gemm_contiguous(BindingLayout.NN)


def k_grouped_bf16_gemm_tn_contiguous(
    a, b, d, ks_cpu, grouped_layout, c=None, compiled_dims="mn", use_psum_layout=True, epilogue=None
) -> None:
    if not use_psum_layout:
        raise ValueError("K-grouped BF16 GEMM requires use_psum_layout=True")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError("a and b must be bfloat16")
    d, resolved = epilogue_ops.resolve(d, c, epilogue=epilogue)
    desc = _get_gemm_desc(
        GemmType.KGroupedContiguousWithPsumLayout,
        a.transpose(0, 1),
        b.transpose(0, 1),
        d,
        c,
        grouped_layout=grouped_layout,
        ks_cpu=ks_cpu,
        epilogue=resolved,
        compiled_dims=compiled_dims,
    )
    if desc is None:
        return
    _launch_gemm(desc, a.transpose(0, 1), b.transpose(0, 1), d, resolved, grouped_layout=grouped_layout)


def k_grouped_fp8_gemm_tn_contiguous(
    a,
    b,
    d,
    ks_cpu,
    grouped_layout,
    c=None,
    recipe=(1, 1, 32),
    compiled_dims="mn",
    use_psum_layout=True,
    epilogue=None,
) -> None:
    if not use_psum_layout:
        raise ValueError("K-grouped FP8 GEMM requires use_psum_layout=True")
    a_data, sfa = _fp8_operand(a, "a")
    b_data, sfb = _fp8_operand(b, "b")
    _check_quantized_pair(a_data, b_data)
    d, resolved = epilogue_ops.resolve(d, c, epilogue=epilogue)
    desc = _get_gemm_desc(
        GemmType.KGroupedContiguousWithPsumLayout,
        a_data.transpose(0, 1),
        b_data.transpose(0, 1),
        d,
        c,
        grouped_layout=grouped_layout,
        ks_cpu=ks_cpu,
        epilogue=resolved,
        compiled_dims=compiled_dims,
    )
    if desc is None:
        return
    recipe = _get_recipe(recipe, None, None)
    scales = _prepare_scales(desc, sfa.transpose(0, 1), sfb.transpose(0, 1), recipe, grouped_layout=grouped_layout)
    _launch_gemm(desc, a_data.transpose(0, 1), b_data.transpose(0, 1), d, resolved, scales=scales, grouped_layout=grouped_layout)


def k_grouped_fp8_gemm_nt_contiguous(
    a,
    b,
    d,
    ks_cpu,
    grouped_layout,
    c=None,
    recipe=(1, 1, 32),
    compiled_dims="mn",
    use_psum_layout=True,
    epilogue=None,
) -> None:
    warnings.warn(
        "Calling k_grouped_fp8_gemm_nt_contiguous is deprecated; transpose a and b "
        "to [K, M]/[K, N] and call k_grouped_fp8_gemm_tn_contiguous instead",
        DeprecationWarning,
        stacklevel=2,
    )
    a_data, sfa = _fp8_operand(a, "a")
    b_data, sfb = _fp8_operand(b, "b")
    return k_grouped_fp8_gemm_tn_contiguous(
        (a_data.transpose(0, 1), sfa.transpose(0, 1)),
        (b_data.transpose(0, 1), sfb.transpose(0, 1)),
        d,
        ks_cpu,
        grouped_layout,
        c,
        recipe,
        compiled_dims,
        use_psum_layout,
        epilogue,
    )


fp8_fp4_gemm_nt = fp8_gemm_nt
fp8_fp4_gemm_nn = fp8_gemm_nn
fp8_fp4_gemm_tn = fp8_gemm_tn
fp8_fp4_gemm_tt = fp8_gemm_tt
m_grouped_fp8_fp4_gemm_nt_contiguous = m_grouped_fp8_gemm_nt_contiguous
m_grouped_fp8_fp4_gemm_nn_contiguous = m_grouped_fp8_gemm_nn_contiguous
epilogue = epilogue_ops


__all__ = [
    "epilogue",
    "fp8_fp4_gemm_nt",
    "fp8_fp4_gemm_nn",
    "fp8_fp4_gemm_tn",
    "fp8_fp4_gemm_tt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nn_contiguous",
    "bf16_bmm",
    "bf16_gemm_nn",
    "bf16_gemm_nt",
    "bf16_gemm_tn",
    "bf16_gemm_tt",
    "einsum",
    "fp8_bmm",
    "fp8_einsum",
    "fp8_gemm_nn",
    "fp8_gemm_nt",
    "fp8_gemm_tn",
    "fp8_gemm_tt",
    "k_grouped_bf16_gemm_tn_contiguous",
    "k_grouped_fp8_gemm_nt_contiguous",
    "k_grouped_fp8_gemm_tn_contiguous",
    "m_grouped_bf16_gemm_nn_contiguous",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nn_contiguous",
    "m_grouped_fp8_gemm_nt_contiguous",
]
