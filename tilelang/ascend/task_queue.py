"""Build and register the ABI-isolated torch_npu task queue adapter."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import sys
import sysconfig
import threading
from pathlib import Path

from tilelang.env import Environment

_SUBMIT_SYMBOL = "tilelang_torch_npu_submit"
_adapter_lock = threading.Lock()
_adapter_module: object | None = None
# Set once a build/registration attempt fails so later kernels in the same
# process fall back to direct ACL launches instead of retrying the build.
_adapter_failed = False

# torch.utils.cpp_extension compiles C++ sources without any -O flag by
# default (i.e. -O0); opt the adapter into optimized code explicitly. These
# flags are part of the extension fingerprint below, so changing them forces
# a fresh adapter build instead of reusing a stale cached .so.
_ADAPTER_CFLAGS = (
    "-O3",
    "-DNDEBUG",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
)


def is_task_queue_enabled() -> bool:
    value = os.getenv("TASK_QUEUE_ENABLE")
    if value is None:
        return True
    match = re.match(r"^\s*[+-]?\d+", value)
    return match is not None and int(match.group()) != 0


def _adapter_source() -> Path:
    package_dir = Path(__file__).resolve().parents[1]
    candidates = (
        package_dir / "src" / "ascend" / "adapter" / "task_queue_adapter.cc",
        package_dir.parent / "src" / "ascend" / "adapter" / "task_queue_adapter.cc",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("TileLang installation is missing src/ascend/adapter/task_queue_adapter.cc")


def _extension_name(source: Path, torch, torch_npu) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(source.read_bytes())
    fingerprint.update(str(torch.__version__).encode())
    fingerprint.update(str(getattr(torch_npu, "__version__", "unknown")).encode())
    fingerprint.update(
        str(
            getattr(
                getattr(torch, "_C", None),
                "_GLIBCXX_USE_CXX11_ABI",
                "unknown",
            )
        ).encode()
    )
    fingerprint.update(str(Path(torch_npu.__file__).resolve().parent).encode())
    fingerprint.update(str(_ADAPTER_CFLAGS).encode())
    fingerprint.update(str(sysconfig.get_config_var("SOABI")).encode())
    fingerprint.update(str(sys.platform).encode())
    fingerprint.update(str(platform.machine()).encode())
    torch_npu_library = Path(torch_npu.__file__).resolve().parent / "lib" / "libtorch_npu.so"
    try:
        library_stat = torch_npu_library.stat()
        library_identity = (library_stat.st_size, library_stat.st_mtime)
    except OSError:
        library_identity = "missing"
    fingerprint.update(str(library_identity).encode())
    return f"tilelang_torch_npu_adapter_{fingerprint.hexdigest()[:16]}"


def _build_adapter():
    try:
        import torch
        import torch_npu
        from torch.utils.cpp_extension import load
    except ImportError as error:
        raise RuntimeError("Ascend task queue support requires torch and torch_npu at runtime.") from error

    source = _adapter_source()
    name = _extension_name(source, torch, torch_npu)
    torch_npu_dir = Path(torch_npu.__file__).resolve().parent
    include_dir = torch_npu_dir / "include"
    library_dir = torch_npu_dir / "lib"
    op_command_header = include_dir / "torch_npu" / "csrc" / "framework" / "OpCommand.h"
    torch_npu_library = library_dir / "libtorch_npu.so"
    if not op_command_header.is_file():
        raise RuntimeError(f"torch_npu installation is missing {op_command_header}")
    if not torch_npu_library.is_file():
        raise RuntimeError(f"torch_npu installation is missing {torch_npu_library}")

    # Keep the build directory fixed under TileLang's own cache tree so the
    # ninja artifacts survive process restarts. Every new process goes through
    # load(), but with unchanged inputs ninja re-checks dependencies instead of
    # recompiling, and torch's FileBaton serializes concurrent first-time
    # builds across processes.
    build_dir = Path(Environment.TILELANG_CACHE_DIR).expanduser() / "torch_extension" / name
    build_dir.mkdir(parents=True, exist_ok=True)

    adapter = load(
        name=name,
        sources=[str(source)],
        extra_include_paths=[
            str(include_dir),
            str(include_dir / "third_party" / "acl" / "inc"),
            str(include_dir / "third_party" / "hccl" / "inc"),
        ],
        extra_cflags=list(_ADAPTER_CFLAGS),
        extra_ldflags=[
            str(torch_npu_library),
            f"-Wl,-rpath,{library_dir}",
        ],
        with_cuda=False,
        build_directory=str(build_dir),
        verbose=os.getenv("TILELANG_TASK_QUEUE_BUILD_VERBOSE", "0") == "1",
    )

    if not hasattr(adapter, "submit_address"):
        raise RuntimeError("torch extension builder returned an invalid adapter module")
    return adapter


def ensure_task_queue_adapter() -> bool:
    """Build once against the active torch_npu installation and register it.

    Returns True when the adapter is registered and kernels run through the
    task queue. Returns False after a failed attempt: the adapter stays
    disabled for the rest of the process so callers can fall back to direct
    ACL launches without retrying the (potentially slow) build. The first
    failed attempt raises.
    """
    global _adapter_module, _adapter_failed
    if _adapter_module is not None:
        return True
    if _adapter_failed:
        return False

    with _adapter_lock:
        if _adapter_module is not None:
            return True
        if _adapter_failed:
            return False

        try:
            adapter = _build_adapter()
            address = adapter.submit_address()
            if not isinstance(address, int) or address == 0:
                raise RuntimeError(f"{_SUBMIT_SYMBOL} has a null address")

            from tilelang import tvm

            tvm.get_global_func("tl.ascend.SetTaskQueueSubmitFn")(address)
            _adapter_module = adapter
        except Exception:
            _adapter_failed = True
            raise
    return True
