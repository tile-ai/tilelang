"""NVRTC Backend for TileLang.

This module provides runtime compilation support using NVIDIA's NVRTC API.
"""

import logging

__all__ = ["NVRTCKernelAdapter", "TLNVRTCSourceWrapper", "NVRTCLibraryGenerator", "is_nvrtc_available", "check_nvrtc_available"]

logger = logging.getLogger(__name__)

# Check if cuda-python is available
is_nvrtc_available = False
NVRTC_UNAVAILABLE_MESSAGE = (
    "cuda-python is not available, NVRTC backend cannot be used. "
    "Please install cuda-python via `pip install cuda-python` "
    "if you want to use the NVRTC backend."
)

try:
    import cuda.bindings.driver as cuda  # noqa: F401
    import cuda.bindings.nvrtc as nvrtc  # noqa: F401

    is_nvrtc_available = True
except ImportError as e:
    logger.debug(f"cuda-python import failed: {e}")


def check_nvrtc_available():
    """Check if NVRTC backend is available.

    Raises
    ------
    ImportError
        If cuda-python is not installed or cannot be imported
    """
    if not is_nvrtc_available:
        raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)


# Conditionally import the adapter
if is_nvrtc_available:
    from .adapter import NVRTCKernelAdapter
    from .wrapper import TLNVRTCSourceWrapper
    from .libgen import NVRTCLibraryGenerator
else:
    # Provide a dummy class that raises error on instantiation
    class NVRTCKernelAdapter:
        """Dummy NVRTCKernelAdapter that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)

        @classmethod
        def from_database(cls, *args, **kwargs):
            raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)

    class TLNVRTCSourceWrapper:
        """Dummy TLNVRTCSourceWrapper that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)

    class NVRTCLibraryGenerator:
        """Dummy NVRTCLibraryGenerator that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)
