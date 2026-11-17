"""Compile PTODSL kernels and their host launcher into a shared library."""

from __future__ import annotations

import logging
import os
import subprocess

from tvm.contrib import utils
from tvm.target import Target

from tilelang.env import env
from tilelang.jit.adapter.libgen import LibraryGenerator

logger = logging.getLogger(__name__)


class PTOLibraryGenerator(LibraryGenerator):
    def __init__(self, target: Target, verbose: bool = False):
        super().__init__(target, verbose)
        self.pto_kernel_source: str | None = None
        self.pto_kernel_names: list[str] | None = None
        self._pto_temp_dir = None
        self._keep_pto_temp_files = False

    def update_pto_kernels(self, pto_kernel_source: str, pto_kernel_names: list[str] | None):
        if not pto_kernel_source.strip():
            raise RuntimeError("PTO compilation requires a non-empty PTODSL kernel source.")
        if not pto_kernel_names:
            raise RuntimeError("PTO compilation requires at least one kernel name.")
        if any(not isinstance(name, str) or not name for name in pto_kernel_names):
            raise RuntimeError(f"Invalid PTO kernel names: {pto_kernel_names}")
        if len(pto_kernel_names) != len(set(pto_kernel_names)):
            raise RuntimeError(f"PTO kernel names must be unique: {pto_kernel_names}")
        self.pto_kernel_source = pto_kernel_source
        self.pto_kernel_names = list(pto_kernel_names)

    def compile_lib(self, timeout: float = None):
        try:
            self._compile_pto_lib()
        except Exception:
            retained_dir = None
            if self._keep_pto_temp_files and self._pto_temp_dir is not None:
                retained_dir = self._pto_temp_dir.temp_dir
            self._cleanup_pto_temp_files()
            if retained_dir is not None:
                logger.warning("PTO temporary directory retained after failure: %s", retained_dir)
            raise

    def _compile_pto_lib(self):
        if self.pto_kernel_source is None:
            raise RuntimeError("PTO compilation requires a PTODSL kernel source.")
        if not self.pto_kernel_names:
            raise RuntimeError("PTO compilation requires at least one kernel name.")
        if self.lib_code is None:
            raise RuntimeError("PTO compilation requires a host launch source.")

        out_dir = self._create_pto_temp_dir()
        ptodsl_path = os.path.join(out_dir, "kernel.ptodsl.py")
        pto_path = os.path.join(out_dir, "kernel.pto")
        fatobj_path = os.path.join(out_dir, "kernel.fatobj.o")
        launch_cpp = os.path.join(out_dir, "launch.cpp")
        launch_obj = os.path.join(out_dir, "launch.o")
        libpath = os.path.join(out_dir, "lib_kernel.so")

        from tilelang.contrib.pto_toolchain import compile_ptodsl_to_pto, resolve_pto_options, run_ptoas

        compile_ptodsl_to_pto(self.pto_kernel_source, self.pto_kernel_names, ptodsl_path, pto_path, self.verbose)
        with open(launch_cpp, "w", encoding="utf-8") as file:
            file.write(self.lib_code)

        pto_arch, pto_flags = resolve_pto_options()
        run_ptoas(pto_path, fatobj_path, pto_arch, pto_flags, self.verbose)

        from tilelang.contrib.bisheng import find_bisheng_path

        bisheng_bin = find_bisheng_path()
        aicore_arch = os.environ.get("PTO_AICORE_ARCH", "dav-c310")

        compile_cmd = [
            bisheng_bin,
            "-c",
            "-fPIC",
            "-O2",
            "-xcce",
            "-Xhost-start",
            "-Xhost-end",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=true",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            f"--cce-aicore-arch={aicore_arch}",
            "-std=c++17",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            launch_cpp,
            "-o",
            launch_obj,
        ]
        link_cmd = [
            bisheng_bin,
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            "-o",
            libpath,
            fatobj_path,
            launch_obj,
            "-Wl,--no-as-needed",
        ]

        for cmd in (compile_cmd, link_cmd):
            if self.verbose:
                print(f"bisheng command: {' '.join(cmd)}")
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"PTO host link failed.\nCommand: {' '.join(cmd)}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}")

        self.srcpath = launch_cpp
        self.libpath = libpath

    def _create_pto_temp_dir(self) -> str:
        if self._pto_temp_dir is not None:
            return self._pto_temp_dir.temp_dir
        self._keep_pto_temp_files = not env.should_cleanup_temp_files()
        self._pto_temp_dir = utils.tempdir(
            keep_for_debug=self._keep_pto_temp_files,
        )
        return self._pto_temp_dir.temp_dir

    def _cleanup_pto_temp_files(self):
        """Release PTO build files during failed compilation cleanup."""
        temp_dir = self._pto_temp_dir
        if temp_dir is None or temp_dir.temp_dir is None:
            return
        if self._keep_pto_temp_files:
            logger.debug("Keeping PTO temporary directory for debugging: %s", temp_dir.temp_dir)
            return

        work_dir = os.path.abspath(temp_dir.temp_dir)
        temp_dir.remove()
        self._pto_temp_dir = None
        self._keep_pto_temp_files = False
        for attr in ("srcpath", "libpath"):
            path = getattr(self, attr, None)
            if path and os.path.abspath(path).startswith(work_dir + os.sep):
                setattr(self, attr, None)
