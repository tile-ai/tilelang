"""BF16 and FP8 tensor APIs backed by TileLang kernels.

The API suffix describes the logical expression. Physical K/MN major is
derived from tensor strides. Dimensions omitted from ``compiled_dims`` and
all outer strides stay in the runtime launch ABI.
Dense GEMMs accept ``alpha=None`` or a runtime FP32 multiplier for A @ B,
applied before output conversion and accumulation into C.

Set ``TILELANG_DEEPGEMM_LOOP_ORDER`` to ``mnk`` or ``kmn`` before the first
import; changing it after import does not switch the kernel builders.

Grouped layouts contain logical end offsets; each group starts at the previous
end rounded up to ``MK_ALIGNMENT``. M-grouped A/D must include the full aligned
row allocation. ``ensure_zero_padding`` does not clear inputs: zero output
padding requires zero input padding and valid scales.
Prepacked int16 scales are used directly, including caller-provided padding.
"""

from __future__ import annotations

import enum
import os
import warnings

import torch

from .config import GemmDesc, GemmType, Major, ceil_div, select_gemm_config, select_transform_sf_config
from .kernels import load_kernels
from .kernels.common import MK_ALIGNMENT
from .kernels.transform_sf import build_transform_sf


LOOP_ORDER = os.environ.get("TILELANG_DEEPGEMM_LOOP_ORDER", "mnk").lower()
_kernels = load_kernels(LOOP_ORDER)
build_bf16_gemm = _kernels.build_bf16_gemm
build_bf16_batched_gemm = _kernels.build_bf16_batched_gemm
build_bf16_m_grouped_gemm = _kernels.build_bf16_m_grouped_gemm
build_bf16_k_grouped_gemm = _kernels.build_bf16_k_grouped_gemm
build_fp8_gemm = _kernels.build_fp8_gemm
build_fp8_batched_gemm = _kernels.build_fp8_batched_gemm
build_fp8_m_grouped_gemm = _kernels.build_fp8_m_grouped_gemm
build_fp8_k_grouped_gemm = _kernels.build_fp8_k_grouped_gemm


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


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise TypeError(f"d must be bfloat16 or float32, got {dtype}")


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
    if tensor.stride(-2) == 1 and tensor.stride(-1) == 1:
        raise ValueError(f"{name} has an ambiguous major axis, got strides={tensor.stride()}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have unit stride along N, got {tensor.stride()}")


def _num_cores(device: torch.device) -> int:
    index = torch.npu.current_device() if device.index is None else device.index
    return int(torch.npu.get_device_properties(index).cube_core_num)


def _copy_source(d: torch.Tensor, c: torch.Tensor | None) -> None:
    if c is not None and c.data_ptr() != d.data_ptr():
        d.copy_(c)


def _gemm_early_return(m: int, n: int, k: int, d: torch.Tensor, c: torch.Tensor | None) -> bool:
    if m == 0 or n == 0:
        return True
    if c is not None:
        _check_major_type_cd(c, "c")
        if c.data_ptr() == d.data_ptr() and c.stride() != d.stride():
            raise ValueError("aliased c and d must have identical strides")
    if k == 0:
        if c is None:
            d.zero_()
        else:
            _copy_source(d, c)
        return True
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
    with_alpha: bool = False,
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
    if c is not None and (c.shape != d.shape or c.dtype != d.dtype):
        raise ValueError("c and d must have identical shape, dtype, and device")
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
        if k_grouped:
            if major_a != Major.MN or major_b != Major.MN:
                raise ValueError("K-grouped A and B must be MN-major after logical transpose")
            if c is None:
                raise ValueError("K-grouped GEMM requires accumulation into c")
    if (batched or m_grouped or k_grouped) and num_groups == 0:
        if m_grouped and d.numel():
            raise ValueError("nonempty M-grouped GEMM requires at least one group")
        return None
    if m_grouped and m and n and k and m % MK_ALIGNMENT:
        raise ValueError(f"M-grouped A/D must include physical padding to {MK_ALIGNMENT} rows, got M={m}")
    if _gemm_early_return(m, n, k, d, c):
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
        acc=c is not None,
        gemm_type=gemm_type,
        num_groups=num_groups,
        expected_m=estimated_m if expected_m is None else int(expected_m),
        expected_k=expected_k,
        num_cores=_num_cores(d.device),
        with_alpha=with_alpha,
        outer_stride_a=_get_outer_stride(a, major_a),
        outer_stride_b=_get_outer_stride(b, major_b),
    )


def _physical(logical: torch.Tensor, major: Major) -> torch.Tensor:
    return logical if major == Major.K else logical.transpose(-2, -1)


def _compiled_shape(m: int, n: int, k: int, compiled_dims: str):
    return tuple(dim if name in compiled_dims else None for name, dim in zip("mnk", (m, n, k)))


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


def _physical_scale(scale: torch.Tensor) -> torch.Tensor:
    """View MN-major logical scales as the kernel's physical K-pair rows."""

    return scale.transpose(-2, -1)


def _make_bf16_gemm(layout: BindingLayout):
    trans_a, trans_b = layout.trans_a, layout.trans_b

    def bf16_gemm(a, b, d, c=None, compiled_dims="mn" if trans_a else "nk", alpha=None) -> None:
        if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
            raise TypeError(f"a and b must be bfloat16, got {a.dtype} and {b.dtype}")
        logical_a = a.transpose(0, 1) if trans_a else a
        logical_b = b.transpose(0, 1) if trans_b else b
        desc = _get_gemm_desc(GemmType.Normal, logical_a, logical_b, d, c, with_alpha=alpha is not None)
        if desc is None:
            return
        config = select_gemm_config(desc)
        m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
        kernel = build_bf16_gemm(
            m_static,
            n_static,
            k_static,
            desc.major_a,
            desc.major_b,
            desc.cd_dtype,
            desc.acc,
            config,
            with_alpha=desc.with_alpha,
        )
        kernel(_physical(logical_a, desc.major_a), _physical(logical_b, desc.major_b), d, 1.0 if alpha is None else float(alpha))

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
    ) -> None:
        a_data, sfa = _fp8_operand(a, "a")
        b_data, sfb = _fp8_operand(b, "b")
        if a_data.dtype is not torch.float8_e4m3fn or b_data.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"a and b data must be float8_e4m3fn, got {a_data.dtype} and {b_data.dtype}")
        logical_a = a_data.transpose(0, 1) if trans_a else a_data
        logical_b = b_data.transpose(0, 1) if trans_b else b_data
        logical_sfa = sfa.transpose(0, 1) if trans_a else sfa
        logical_sfb = sfb.transpose(0, 1) if trans_b else sfb
        desc = _get_gemm_desc(GemmType.Normal, logical_a, logical_b, d, c, with_alpha=alpha is not None)
        if desc is None:
            return
        recipe = _get_recipe(recipe, recipe_a, recipe_b)
        transformed_sfa = _transform_sf_into_required_layout(
            logical_sfa,
            desc.m,
            desc.k,
            recipe,
            is_sfa=True,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )
        transformed_sfb = _transform_sf_into_required_layout(
            logical_sfb,
            desc.n,
            desc.k,
            recipe,
            is_sfa=False,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )
        config = select_gemm_config(desc)
        m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
        kernel = build_fp8_gemm(
            m_static,
            n_static,
            k_static,
            desc.major_a,
            desc.major_b,
            desc.cd_dtype,
            desc.acc,
            config,
            with_alpha=desc.with_alpha,
        )
        kernel(
            _physical(logical_a, desc.major_a),
            _physical(logical_b, desc.major_b),
            _physical_scale(transformed_sfa),
            _physical_scale(transformed_sfb),
            d,
            1.0 if alpha is None else float(alpha),
        )

    fp8_gemm.__name__ = fp8_gemm.__qualname__ = f"fp8_gemm_{layout.value}"
    return fp8_gemm


fp8_gemm_nt = _make_fp8_gemm(BindingLayout.NT)
fp8_gemm_nn = _make_fp8_gemm(BindingLayout.NN)
fp8_gemm_tn = _make_fp8_gemm(BindingLayout.TN)
fp8_gemm_tt = _make_fp8_gemm(BindingLayout.TT)


def bf16_bmm(a, b, d, c=None, compiled_dims="nk") -> None:
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(f"a and b must be bfloat16, got {a.dtype} and {b.dtype}")
    desc = _get_gemm_desc(GemmType.Batched, a, b, d, c)
    if desc is None:
        return
    config = select_gemm_config(desc)
    m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
    kernel = build_bf16_batched_gemm(
        m_static,
        n_static,
        k_static,
        desc.major_a,
        desc.major_b,
        desc.cd_dtype,
        desc.acc,
        config,
    )
    kernel(_physical(a, desc.major_a), _physical(b, desc.major_b), d)


def fp8_bmm(a, b, d, c=None, recipe=(1, 1, 32), compiled_dims="nk") -> None:
    a_data, sfa = _fp8_operand(a, "a")
    b_data, sfb = _fp8_operand(b, "b")
    if a_data.dtype is not torch.float8_e4m3fn or b_data.dtype is not torch.float8_e4m3fn:
        raise TypeError("a and b data must be float8_e4m3fn")
    desc = _get_gemm_desc(GemmType.Batched, a_data, b_data, d, c)
    if desc is None:
        return
    recipe = _get_recipe(recipe, None, None)
    transformed_sfa = _transform_sf_into_required_layout(sfa, desc.m, desc.k, recipe, is_sfa=True, num_groups=desc.num_groups)
    transformed_sfb = _transform_sf_into_required_layout(sfb, desc.n, desc.k, recipe, is_sfa=False, num_groups=desc.num_groups)
    config = select_gemm_config(desc)
    m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
    kernel = build_fp8_batched_gemm(
        m_static,
        n_static,
        k_static,
        desc.major_a,
        desc.major_b,
        desc.cd_dtype,
        desc.acc,
        config,
    )
    kernel(
        _physical(a_data, desc.major_a),
        _physical(b_data, desc.major_b),
        _physical_scale(transformed_sfa),
        _physical_scale(transformed_sfb),
        d,
    )


def einsum(expr, a, b, d, c=None, use_cublaslt=False) -> None:
    if use_cublaslt:
        raise RuntimeError("einsum: use_cublaslt is not implemented on Ascend")
    if expr == "bhr,hdr->bhd":
        bf16_bmm(
            a.permute(1, 0, 2),
            b,
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            "nk",
        )
    elif expr == "bhd,hdr->bhr":
        bf16_bmm(
            a.permute(1, 0, 2),
            b.permute(0, 2, 1),
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            "nk",
        )
    elif expr == "bhd,bhr->hdr":
        bf16_bmm(
            a.permute(1, 2, 0),
            b.permute(1, 2, 0),
            d,
            c,
            "mn",
        )
    else:
        raise ValueError(f"unsupported einsum expression: {expr!r}")


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 1, 32)) -> None:
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
        )
    elif expr == "bhd,hdr->bhr":
        fp8_bmm(
            permute_pair(a_data, sfa, (1, 0, 2)),
            permute_pair(b_data, sfb, (0, 2, 1)),
            d.permute(1, 0, 2),
            c.permute(1, 0, 2) if c is not None else None,
            recipe,
            "nk",
        )
    elif expr == "bhd,bhr->hdr":
        fp8_bmm(
            permute_pair(a_data, sfa, (1, 2, 0)),
            permute_pair(b_data, sfb, (1, 2, 0)),
            d,
            c,
            recipe,
            "mn",
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
    ) -> None:
        del ensure_zero_padding  # Input padding is supplied by the caller.
        if not use_psum_layout:
            raise ValueError("M-grouped BF16 GEMM requires use_psum_layout=True")
        if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
            raise TypeError("a and b must be bfloat16")
        logical_b = b.transpose(1, 2) if trans_b else b
        desc = _get_gemm_desc(
            GemmType.MGroupedContiguousWithPsumLayout,
            a,
            logical_b,
            d,
            grouped_layout=grouped_layout,
            expected_m=expected_m_for_psum_layout,
        )
        if desc is None:
            return
        config = select_gemm_config(desc)
        m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
        kernel = build_bf16_m_grouped_gemm(m_static, n_static, k_static, desc.major_b, desc.cd_dtype, config)
        kernel(a, _physical(logical_b, desc.major_b), d, grouped_layout)

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
    ) -> None:
        del ensure_zero_padding  # Input padding is supplied by the caller.
        if not use_psum_layout:
            raise ValueError("M-grouped FP8 GEMM requires use_psum_layout=True")
        a_data, sfa = _fp8_operand(a, "a")
        b_data, sfb = _fp8_operand(b, "b")
        if a_data.dtype is not torch.float8_e4m3fn or b_data.dtype is not torch.float8_e4m3fn:
            raise TypeError("a and b data must be float8_e4m3fn")
        logical_b = b_data.transpose(1, 2) if trans_b else b_data
        logical_sfb = sfb.transpose(1, 2) if trans_b else sfb
        desc = _get_gemm_desc(
            GemmType.MGroupedContiguousWithPsumLayout,
            a_data,
            logical_b,
            d,
            grouped_layout=grouped_layout,
            expected_m=0 if expected_m_for_psum_layout is None else expected_m_for_psum_layout,
        )
        if desc is None:
            return
        recipe = _get_recipe(recipe, recipe_a, recipe_b)
        transformed_sfa = _transform_sf_into_required_layout(
            sfa,
            desc.m,
            desc.k,
            recipe,
            is_sfa=True,
            disable_ue8m0_cast=disable_ue8m0_cast,
            grouped_layout=grouped_layout,
        )
        transformed_sfb = _transform_sf_into_required_layout(
            logical_sfb,
            desc.n,
            desc.k,
            recipe,
            is_sfa=False,
            num_groups=desc.num_groups,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )
        config = select_gemm_config(desc)
        m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
        kernel = build_fp8_m_grouped_gemm(m_static, n_static, k_static, desc.major_b, desc.cd_dtype, config)
        kernel(
            a_data,
            _physical(logical_b, desc.major_b),
            _physical_scale(transformed_sfa),
            _physical_scale(transformed_sfb),
            d,
            grouped_layout,
        )

    m_grouped_fp8_gemm_contiguous.__name__ = m_grouped_fp8_gemm_contiguous.__qualname__ = f"m_grouped_fp8_gemm_{layout.value}_contiguous"
    return m_grouped_fp8_gemm_contiguous


m_grouped_fp8_gemm_nt_contiguous = _make_m_grouped_fp8_gemm_contiguous(BindingLayout.NT)
m_grouped_fp8_gemm_nn_contiguous = _make_m_grouped_fp8_gemm_contiguous(BindingLayout.NN)


def k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, compiled_dims="mn", use_psum_layout=True) -> None:
    if not use_psum_layout:
        raise ValueError("K-grouped BF16 GEMM requires use_psum_layout=True")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError("a and b must be bfloat16")
    desc = _get_gemm_desc(
        GemmType.KGroupedContiguousWithPsumLayout,
        a.transpose(0, 1),
        b.transpose(0, 1),
        d,
        c,
        grouped_layout=grouped_layout,
        ks_cpu=ks_cpu,
    )
    if desc is None:
        return
    config = select_gemm_config(desc)
    m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
    kernel = build_bf16_k_grouped_gemm(m_static, n_static, k_static, desc.cd_dtype, config)
    kernel(a, b, d, grouped_layout)


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
) -> None:
    if not use_psum_layout:
        raise ValueError("K-grouped FP8 GEMM requires use_psum_layout=True")
    a_data, sfa = _fp8_operand(a, "a")
    b_data, sfb = _fp8_operand(b, "b")
    if a_data.dtype is not torch.float8_e4m3fn or b_data.dtype is not torch.float8_e4m3fn:
        raise TypeError("a and b data must be float8_e4m3fn")
    desc = _get_gemm_desc(
        GemmType.KGroupedContiguousWithPsumLayout,
        a_data.transpose(0, 1),
        b_data.transpose(0, 1),
        d,
        c,
        grouped_layout=grouped_layout,
        ks_cpu=ks_cpu,
    )
    if desc is None:
        return
    recipe = _get_recipe(recipe, None, None)
    transformed_sfa = _transform_sf_into_required_layout(
        sfa.transpose(0, 1),
        desc.m,
        desc.k,
        recipe,
        is_sfa=True,
        grouped_layout=grouped_layout,
        k_grouped=True,
    )
    transformed_sfb = _transform_sf_into_required_layout(
        sfb.transpose(0, 1),
        desc.n,
        desc.k,
        recipe,
        is_sfa=False,
        grouped_layout=grouped_layout,
        k_grouped=True,
    )
    config = select_gemm_config(desc)
    m_static, n_static, k_static = _compiled_shape(desc.m, desc.n, desc.k, compiled_dims)
    kernel = build_fp8_k_grouped_gemm(m_static, n_static, k_static, desc.cd_dtype, config)
    kernel(a_data, b_data, _physical_scale(transformed_sfa), _physical_scale(transformed_sfb), d, grouped_layout)


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
    )


__all__ = [
    "LOOP_ORDER",
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
