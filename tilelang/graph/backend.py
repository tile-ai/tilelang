"""TileLang torch.compile backend entry point."""

import logging
from collections.abc import Callable

import torch
from torch import fx

from tilelang import tvm as tvm
from tvm.target import Target

from tilelang.graph import cache as graph_cache
from tilelang.graph.converter import fx_to_relax
from tilelang.graph.utils import validate_cuda_tensors
from tilelang.backend.target import determine_target

logger = logging.getLogger(__name__)


class _BackendConfig:
    """Mutable configuration for the ``"tilelang"`` torch.compile backend.

    Set attributes on the singleton :data:`tilelang.graph.backend_config`
    **before** calling ``torch.compile``::

        import tilelang
        from tilelang.graph import backend_config

        backend_config.extern_dispatch = my_dispatch
        compiled = torch.compile(model, backend="tilelang")
    """

    def __init__(self):
        self.extern_dispatch: Callable[..., bool] | None = None
        self.vm_clone_output: bool = True  # Clone VM outputs (False for benchmarking)
        self.use_cuda_graph: bool = False  # Capture static regions as CUDA graphs (WIP)
        self.auto_fp32_promote: bool = True  # Match Inductor: run math ops at fp32 for fp16/bf16 inputs

    def reset(self):
        """Restore all options to defaults."""
        self.extern_dispatch = None
        self.vm_clone_output = True
        self.use_cuda_graph = False
        self.auto_fp32_promote = True

    def cache_key(self) -> tuple:
        """Return options that affect conversion, lowering, or execution."""
        dispatch_key = None
        if self.extern_dispatch is not None:
            dispatch_key = (
                getattr(self.extern_dispatch, "__module__", None),
                getattr(self.extern_dispatch, "__qualname__", None),
                id(self.extern_dispatch),
            )
        return (
            dispatch_key,
            self.vm_clone_output,
            self.use_cuda_graph,
            self.auto_fp32_promote,
        )


backend_config = _BackendConfig()


def _detect_target(device: torch.device) -> Target:
    with torch.cuda.device(device):
        return determine_target("auto", return_object=True)


def tilelang_backend(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> Callable:
    """torch.compile backend that compiles FX graphs using TileLang.

    Converts FX graph → Relax IR → optimized TIR → Relax VM executable.
    Unsupported or explicitly dispatched ops run through registered TVM
    packed functions at VM runtime.
    """
    tensor_inputs = [inp for inp in example_inputs if isinstance(inp, torch.Tensor)]
    device = validate_cuda_tensors(tensor_inputs)
    dispatch = backend_config.extern_dispatch
    key = graph_cache.graph_cache_key(gm, example_inputs, backend_config.cache_key())

    # In-memory cache
    cached = graph_cache.get_memory_cached(key)
    if cached is not None:
        return cached

    # Cold compile
    try:
        target = _detect_target(device)
        relax_mod, fallback_calls = fx_to_relax(
            gm,
            tensor_inputs,
            extern_dispatch=dispatch,
            fallback_namespace=key,
        )

        from tilelang.graph.vm_build import build_vm_runner

        wrapper = build_vm_runner(
            relax_mod,
            target,
            fallback_calls=fallback_calls,
            clone_output=backend_config.vm_clone_output,
            use_cuda_graph=backend_config.use_cuda_graph,
            device_index=device.index,
        )
    except Exception:
        logger.error("tilelang_backend compilation failed", exc_info=True)
        raise

    graph_cache.put_memory_cached(key, wrapper)
    return wrapper
