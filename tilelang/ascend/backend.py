"""Ascend backend manifests (plain AscendC and PTO)."""

from __future__ import annotations

from tilelang.ascend.target import target_is_plain_ascend, target_is_pto
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import bisheng
from tilelang.env import TILELANG_TEMPLATE_PATH, env
from tilelang.transform import PassConfigKey

from . import codegen, execution_backend, pipeline


def tilelang_callback_ascend_compile(code, target, pass_config=None):
    """Compile AscendC source into a cached, executable CCE ELF."""
    cfg = pass_config or {}
    extra_flags = bisheng.normalize_options(cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS))
    options = ["-I" + TILELANG_TEMPLATE_PATH, *extra_flags]

    target_arch = bisheng.get_target_npu_arch(target)

    compile_options = bisheng.get_bisheng_compile_options(target_arch, options)
    linker_options = bisheng.get_aibin_linker_options()

    from tilelang.cache.ascend_binary_cache import AscendBinaryCache

    cache_key = AscendBinaryCache.make_key(
        code=code,
        target_kind=target.kind.name,
        target_arch=target_arch,
        compile_format=AscendBinaryCache.binary_format,
        options=compile_options,
        linker_options=linker_options,
    )
    cached_binary = AscendBinaryCache.load(cache_key)
    if cached_binary is not None:
        return bytearray(cached_binary)

    aibin = bisheng.compile_ascend(
        code,
        target_format="aibin",
        npu_arch=target_arch,
        options=options,
        verbose=env.get_default_verbose(),
    )
    AscendBinaryCache.save(cache_key, aibin)
    return aibin


# Plain Ascend and PTO share the "ascend" target kind, so each is its own
# BackendModule with a mutually exclusive `supports_target` predicate; the
# resolver then selects exactly one. Both reuse the same lowering pipeline.
BACKEND = register_backend(
    BackendModule(
        name="ascend",
        target_kinds=("ascend",),
        supports_target=target_is_plain_ascend,
        pipelines={"ascend": pipeline.ascend_pipeline},
        device_codegens={"ascend": codegen.ASCEND_CODEGEN},
        execution_backends=execution_backend.ASCEND_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_ascend_compile": tilelang_callback_ascend_compile},
    )
)

PTO_BACKEND = register_backend(
    BackendModule(
        name="pto",
        target_kinds=("ascend",),
        supports_target=target_is_pto,
        pipelines={"ascend": pipeline.ascend_pipeline},
        device_codegens={"ascend": codegen.PTO_CODEGEN},
        execution_backends=execution_backend.PTO_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
