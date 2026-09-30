"""Compile PTODSL kernels and their host launcher into a shared library."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

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

    @staticmethod
    def _compile_ptodsl_source_to_pto(
        ptodsl_source: str,
        kernel_names: list[str],
        src_path: str | Path,
        out_path: str | Path,
    ) -> None:
        src_path = Path(src_path)
        out_path = Path(out_path)
        src_path.write_text(ptodsl_source, encoding="utf-8")

        script = textwrap.dedent(
            """
            import importlib.util
            import json
            import pathlib
            import re
            import sys
            import traceback
            import types

            src_path = pathlib.Path(sys.argv[1])
            kernel_names = json.loads(sys.argv[2])
            out_path = pathlib.Path(sys.argv[3])
            module_name = "_tilelang_ptodsl_compile"

            try:
                # Importing ``tilelang.contrib.ptodsl`` normally executes the
                # top-level tilelang and contrib package initializers.  Those
                # initializers load the full TileLang runtime (and, in this
                # environment, Triton/LLVM libraries), which conflicts with
                # PTOAS's LLVM libraries in this compiler-only subprocess.
                # Expose only namespace packages so the PTODSL helper modules
                # can be imported without running either initializer.
                tilelang_spec = importlib.util.find_spec("tilelang")
                if tilelang_spec is None or tilelang_spec.origin is None:
                    raise ImportError("Unable to locate the tilelang package for PTODSL helpers")
                tilelang_root = pathlib.Path(tilelang_spec.origin).parent

                tilelang_pkg = types.ModuleType("tilelang")
                tilelang_pkg.__path__ = [str(tilelang_root)]
                sys.modules["tilelang"] = tilelang_pkg

                contrib_pkg = types.ModuleType("tilelang.contrib")
                contrib_pkg.__path__ = [str(tilelang_root / "contrib")]
                sys.modules["tilelang.contrib"] = contrib_pkg

                spec = importlib.util.spec_from_file_location(module_name, src_path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
            except Exception:
                traceback.print_exc()
                sys.exit(2)

            try:
                kernels = []
                for kernel_name in kernel_names:
                    kernel = getattr(module, kernel_name)
                    if not callable(getattr(kernel, "build", None)) or not callable(getattr(kernel, "compile", None)):
                        raise TypeError(f"PTODSL entry `{kernel_name}` is not a compilable PTO kernel handle")
                    kernels.append(kernel)

                if len(kernels) == 1:
                    pto_text = kernels[0].compile().mlir_text()
                else:
                    from ptodsl import pto

                    merge_jit_modules = getattr(pto, "merge_jit_modules", None)
                    if not callable(merge_jit_modules):
                        raise RuntimeError(
                            "PTO multi-kernel JIT requires ptodsl.pto.merge_jit_modules "
                            "from ptoas-vmi 0.1.4 or newer"
                        )
                    pto_text = str(merge_jit_modules(*kernels))

                    # TODO: Remove this temporary workaround once the duplicate helper
                    # symbol issue is fixed in PTOAS. PTOAS may emit identical global
                    # symbols for helpers (notably ``pto.init_core``) in each nested kernel
                    # module. Give duplicate module-local helpers distinct symbols so the
                    # fatobj linker can combine the entries without changing semantics.
                    modules = re.split(r"(?m)(?=^  module attributes \\{pto\\.backend = )", pto_text)
                    seen_helpers = set()
                    rewritten = [modules[0]]
                    for module_index, module_text in enumerate(modules[1:], 1):
                        helper_names = re.findall(
                            r"func\\.func @([A-Za-z_][A-Za-z0-9_]*)\\([^\\n]*\\)"
                            r' attributes \\{pto\\.ptodsl\\.callable_kind = "func"',
                            module_text,
                        )
                        for helper_name in helper_names:
                            if helper_name in seen_helpers:
                                unique_name = f"{helper_name}__module_{module_index}"
                                module_text = re.sub(
                                    rf"(?<![A-Za-z0-9_]){re.escape(helper_name)}(?![A-Za-z0-9_])",
                                    unique_name,
                                    module_text,
                                )
                            seen_helpers.add(helper_name)
                        rewritten.append(module_text)
                    pto_text = "".join(rewritten)
                out_path.write_text(pto_text, encoding="utf-8")
            except Exception:
                traceback.print_exc()
                sys.exit(3)
            """
        )

        python_bin = sys.executable
        result = subprocess.run(
            [python_bin, "-c", script, str(src_path), json.dumps(kernel_names), str(out_path)],
            text=True,
            capture_output=True,
        )
        if result.returncode == 2:
            raise RuntimeError(
                f"PTODSL compiler frontend is unavailable in the current environment.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                "PTODSL compile-only lowering failed.\n"
                f"Kernels: {kernel_names}\n"
                f"Command: {python_bin} -c <ptodsl-compile-script> {src_path} <kernel-names-json> {out_path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

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

        self._compile_ptodsl_source_to_pto(self.pto_kernel_source, self.pto_kernel_names, ptodsl_path, pto_path)
        with open(launch_cpp, "w", encoding="utf-8") as file:
            file.write(self.lib_code)

        pto_arch = os.environ.get("PTO_ARCH", "a5").strip().lower()
        if pto_arch != "a5":
            raise RuntimeError(f"Unsupported PTO_ARCH for PTO JIT: {pto_arch}")
        pto_flags = shlex.split(os.environ.get("PTO_FLAGS", ""))
        if not any(flag == "--pto-backend" or flag.startswith("--pto-backend=") for flag in pto_flags):
            pto_flags.append("--pto-backend=vpto")

        pto_cmd = ["ptoas", f"--pto-arch={pto_arch}", *pto_flags, pto_path, "-o", fatobj_path]
        if self.verbose:
            print(f"PTO compile command: {' '.join(pto_cmd)}")
        result = subprocess.run(pto_cmd, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"PTO lowering failed for kernels {self.pto_kernel_names}.\n"
                f"Command: {' '.join(pto_cmd)}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}"
            )

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
