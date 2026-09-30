"""TVM-FFI execution with Torch's current NPU stream."""

from __future__ import annotations

import torch

from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


def _install_torch_stream_exchange() -> None:
    from tilelang.ascend.torch_exchange import (
        install_torch_npu_stream_exchange,
    )

    install_torch_npu_stream_exchange()


class AscendTVMFFIKernelAdapter(TVMFFIKernelAdapter):
    """Install the NPU DLPack stream callback before the first invocation."""

    _torch_npu_stream_exchange_installed: bool = False

    def _prepare_torch_device(self, device: torch.device) -> None:
        if device.type == "npu" and not self._torch_npu_stream_exchange_installed:
            _install_torch_stream_exchange()
            self._torch_npu_stream_exchange_installed = True
