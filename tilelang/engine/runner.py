"""Standalone runner for TileLang generated host/CUDA source files."""

from __future__ import annotations

import ctypes
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tvm.target import Target

from tilelang import tvm
from tilelang.contrib.nvcc import get_nvcc_compiler, get_target_arch, get_target_compute_version
from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.utils.target import determine_target

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class CUDARunner:
    """Compile and execute TileLang generated CUDA kernels from source files.

    Parameters
    ----------
    host_code_path : str | Path
        Path to TileLang-generated host source file.
    cuda_source_path : str | Path
        Path to TileLang-generated CUDA source file.
    target : str | Target
        CUDA compilation target.
    lib_path : str | Path | None
        Output shared library path. If None, a temp path is used.
    compile_flags : list[str] | None
        Extra NVCC compile flags.
    verbose : bool
        Print compile command/output if True.
    apply_postproc : bool
        Whether to invoke TileLang post-processing callbacks before compile.
    keep_temporary_sources : bool
        Keep transformed temp source files.
    auto_compile : bool
        Compile and load during initialization.
    """

    def __init__(
        self,
        host_code_path: str | Path,
        cuda_source_path: str | Path,
        target: str | Target = "cuda",
        lib_path: str | Path | None = None,
        compile_flags: list[str] | None = None,
        verbose: bool = False,
        apply_postproc: bool = True,
        keep_temporary_sources: bool = False,
        auto_compile: bool = True,
    ):
        self.host_code_path = Path(host_code_path).expanduser().resolve()
        self.cuda_source_path = Path(cuda_source_path).expanduser().resolve()
        self.verbose = verbose
        self.apply_postproc = apply_postproc
        self.keep_temporary_sources = keep_temporary_sources
        self.compile_flags = compile_flags or []
        self.target = Target.canon_target(determine_target(target, return_object=True))
        if self.target.kind.name != "cuda":
            raise ValueError(f"CUDARunner only supports CUDA target, got {self.target}")

        if not self.host_code_path.is_file():
            raise FileNotFoundError(f"Host source file not found: {self.host_code_path}")
        if not self.cuda_source_path.is_file():
            raise FileNotFoundError(f"CUDA source file not found: {self.cuda_source_path}")

        if lib_path is None:
            self.lib_path: Path | None = None
        else:
            self.lib_path = Path(lib_path).expanduser().resolve()

        self._lib: ctypes.CDLL | None = None
        self._temp_sources: list[Path] = []
        self._processed_host_source: str | None = None
        self._processed_cuda_source: str | None = None
        self.last_compile_command: list[str] | None = None

        if auto_compile:
            self.compile()
            self.load()

    def _get_global_func(self, names: list[str]) -> Any | None:
        for name in names:
            try:
                return tvm.ffi.get_global_func(name)
            except Exception:
                continue
        return None

    def _read_sources(self) -> tuple[str, str]:
        host_source = self.host_code_path.read_text(encoding="utf-8")
        cuda_source = self.cuda_source_path.read_text(encoding="utf-8")
        return host_source, cuda_source

    def _apply_postprocess(self, host_source: str, cuda_source: str) -> tuple[str, str]:
        if not self.apply_postproc:
            return host_source, cuda_source

        cuda_postproc = self._get_global_func(["tilelang_callback_cuda_postproc"])
        if cuda_postproc is not None:
            cuda_source = str(cuda_postproc(cuda_source, self.target))

        host_postproc = self._get_global_func(
            [
                "tilelang_callback_host_c_postproc",
                "tilelang_callback_c_host_postproc",
            ]
        )
        if host_postproc is not None:
            host_source = str(host_postproc(host_source, self.target))

        return host_source, cuda_source

    def _get_target_arch(self) -> str:
        compute_version = get_target_compute_version(self.target)
        return get_target_arch(compute_version)

    def _split_flags(self, flags: list[str]) -> list[str]:
        tokens: list[str] = []
        for flag in flags:
            if isinstance(flag, str):
                tokens.extend(shlex.split(flag))
            else:
                tokens.append(str(flag))
        return tokens

    def _make_output_lib_path(self) -> Path:
        if self.lib_path is not None:
            self.lib_path.parent.mkdir(parents=True, exist_ok=True)
            return self.lib_path
        tmp = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
        tmp.close()
        return Path(tmp.name)

    def _write_temp_sources(self, host_source: str, cuda_source: str) -> tuple[Path, Path]:
        host_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
        cuda_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
        host_tmp.write(host_source)
        cuda_tmp.write(cuda_source)
        host_tmp.close()
        cuda_tmp.close()
        host_path = Path(host_tmp.name).resolve()
        cuda_path = Path(cuda_tmp.name).resolve()
        self._temp_sources = [host_path, cuda_path]
        return host_path, cuda_path

    def _cleanup_temp_sources(self):
        if self.keep_temporary_sources:
            return
        for path in self._temp_sources:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        self._temp_sources.clear()

    def _build_compile_command(self, host_tmp: Path, cuda_tmp: Path, output_lib: Path) -> list[str]:
        target_arch = self._get_target_arch()
        command = [
            get_nvcc_compiler(),
            "-std=c++17",
            "-w",
            "-Xcudafe",
            "--diag_suppress=177",
            "--compiler-options",
            "-fPIC",
            "-lineinfo",
            "--shared",
            str(host_tmp),
            str(cuda_tmp),
            "-lcuda",
            "-gencode",
            f"arch=compute_{target_arch},code=sm_{target_arch}",
            "-I" + TILELANG_TEMPLATE_PATH,
        ]

        if CUTLASS_INCLUDE_DIR:
            command += ["-I" + CUTLASS_INCLUDE_DIR]
        if CUDA_HOME:
            command += ["-I" + str(Path(CUDA_HOME) / "include")]

        command += self._split_flags(self.compile_flags)
        command += ["-o", str(output_lib)]
        return command

    def compile(self, timeout: float | None = None) -> Path:
        """Compile host/CUDA sources into a shared object and return its path."""
        host_source, cuda_source = self._read_sources()
        host_source, cuda_source = self._apply_postprocess(host_source, cuda_source)
        self._processed_host_source = host_source
        self._processed_cuda_source = cuda_source

        output_lib = self._make_output_lib_path()
        host_tmp, cuda_tmp = self._write_temp_sources(host_source, cuda_source)
        command = self._build_compile_command(host_tmp, cuda_tmp, output_lib)
        self.last_compile_command = command

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
        finally:
            self._cleanup_temp_sources()

        if self.verbose and result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(f"CUDARunner failed to compile source files.\nCommand: {' '.join(command)}\nOutput:\n{result.stdout}")

        self.lib_path = output_lib.resolve()
        return self.lib_path

    def load(self, lib_path: str | Path | None = None):
        """Load compiled shared object and initialize runtime state."""
        if lib_path is not None:
            self.lib_path = Path(lib_path).expanduser().resolve()
        if self.lib_path is None:
            raise ValueError("No library path provided. Call compile() first or pass lib_path to load().")
        if not self.lib_path.is_file():
            raise FileNotFoundError(f"Compiled library file not found: {self.lib_path}")

        self._lib = ctypes.CDLL(str(self.lib_path))
        if hasattr(self._lib, "get_last_error"):
            self._lib.get_last_error.restype = ctypes.c_char_p
        if not hasattr(self._lib, "call"):
            raise RuntimeError(f"Loaded library has no `call` symbol: {self.lib_path}")

        self._lib.call.restype = ctypes.c_int
        if hasattr(self._lib, "init"):
            self._lib.init.restype = ctypes.c_int
            ret = int(self._lib.init())
            if ret != 0:
                msg = self.get_last_error() or f"Initialization failed with return code {ret}"
                raise RuntimeError(msg)

    def get_last_error(self) -> str | None:
        """Return last runtime error string from the shared library, if available."""
        if self._lib is None or not hasattr(self._lib, "get_last_error"):
            return None
        err = self._lib.get_last_error()
        if not err:
            return None
        if isinstance(err, bytes):
            return err.decode("utf-8", errors="replace")
        return str(err)

    def _resolve_stream(self, stream: int | Any | None) -> int:
        if stream is not None:
            if hasattr(stream, "cuda_stream"):
                return int(stream.cuda_stream)
            return int(stream)
        if torch is not None and torch.cuda.is_available():
            return int(torch.cuda.current_stream().cuda_stream)
        return 0

    @staticmethod
    def _to_c_arg(arg: Any):
        if arg is None:
            return ctypes.c_void_p(0)
        if isinstance(arg, ctypes._SimpleCData):  # pylint: disable=protected-access
            return arg
        if hasattr(arg, "data_ptr") and callable(arg.data_ptr):
            return ctypes.c_void_p(int(arg.data_ptr()))
        if hasattr(arg, "item") and callable(arg.item):
            arg = arg.item()
        if isinstance(arg, bool):
            return int(arg)
        if isinstance(arg, (int, float)):
            return arg
        raise TypeError(f"Unsupported argument type for kernel launch: {type(arg)}")

    def run(self, *args, stream: int | Any | None = None) -> int:
        """Invoke the compiled `call` entry with tensor/scalar arguments."""
        if self._lib is None:
            if self.lib_path is None:
                self.compile()
            self.load()

        assert self._lib is not None
        call_args = [self._to_c_arg(arg) for arg in args]
        stream_ptr = self._resolve_stream(stream)
        call_args.append(ctypes.c_void_p(stream_ptr))
        ret = int(self._lib.call(*call_args))
        if ret != 0:
            msg = self.get_last_error() or f"Kernel execution failed with return code {ret}"
            raise RuntimeError(msg)
        return ret

    def __call__(self, *args, stream: int | Any | None = None) -> int:
        return self.run(*args, stream=stream)
