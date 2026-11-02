from __future__ import annotations

import re
import shlex

from tvm import tirx

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import hgcc
from tilelang.env import env
from tilelang.transform import PassConfigKey

from . import codegen, execution_backend, pipeline

_GLOBAL_KERNEL_PATTERN = re.compile(r'(?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+)')


def _raise_on_pdl(node):
    if isinstance(node, tirx.Call) and node.op.name in ("tl.pdl_trigger", "tl.pdl_sync"):
        raise ValueError("PDL is not supported on PPU backend")


def tilelang_callback_ppu_validate(device_mod, target):
    support_pdl = hgcc.have_pdl(target)
    for _, base_func in device_mod.functions.items():
        if not isinstance(base_func, tirx.PrimFunc):
            continue
        if not support_pdl:
            tirx.stmt_functor.post_order_visit(base_func.body, _raise_on_pdl)
        if not base_func.attrs:
            continue
        source = base_func.attrs.get("code_block_source")
        if source is None:
            continue
        global_symbol = base_func.attrs.get("global_symbol")
        if global_symbol is None:
            raise ValueError("CodeGenTileLangHGGC expects source-kernel PrimFunc to have the global_symbol attribute")
        expected_name = str(global_symbol)
        entry_name = base_func.attrs.get("code_block_entry_name")
        if entry_name is not None and str(entry_name) != expected_name:
            raise ValueError("T.HGGCSourceCodeKernel expects the lowered device global_symbol to match entry_name")
        kernel_names = list(dict.fromkeys(_GLOBAL_KERNEL_PATTERN.findall(str(source))))
        if not kernel_names:
            raise ValueError("T.HGGCSourceCodeKernel expects external HGGC source to declare at least one __global__ kernel")
        if expected_name not in kernel_names:
            raise ValueError(
                f"T.HGGCSourceCodeKernel expected device global_symbol `{expected_name}` to match a __global__ kernel "
                f"in the provided HGGC source. Available entries: {', '.join(kernel_names)}"
            )


def tilelang_callback_ppu_compile(code, target, pass_config=None):
    cfg = pass_config or {}
    options: list[str] = []
    if cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False):
        options.append("--use_fast_math")
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS)
    if isinstance(extra_flags, str):
        options.extend(shlex.split(extra_flags))
    elif extra_flags:
        for flag in extra_flags:
            options.extend(shlex.split(flag) if isinstance(flag, str) else [str(flag)])
    return hgcc.compile_ppu(code, target=target, options=options, verbose=env.get_default_verbose())


BACKEND = register_backend(
    BackendModule(
        name="ppu",
        target_kinds=("ppu",),
        pipelines={"ppu": pipeline.PPU_PIPELINE},
        device_codegens={
            "ppu": DeviceCodegen(
                "ppu",
                build=codegen.build_ppu,
                build_without_compile=codegen.build_ppu_without_compile,
            )
        },
        execution_backends=execution_backend.PPU_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={
            "tilelang_callback_ppu_validate": tilelang_callback_ppu_validate,
            "tilelang_callback_ppu_compile": tilelang_callback_ppu_compile,
        },
    )
)
