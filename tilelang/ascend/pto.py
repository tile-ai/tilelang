"""PTO device compilation; host execution is owned by TVM-FFI Host IR."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import textwrap

from tvm.contrib import utils

from tilelang.contrib import bisheng
from tilelang.env import env
from tilelang.transform import PassConfigKey


def _find_msobjdump() -> str:
    tool = shutil.which("msobjdump")
    if tool:
        return tool
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.environ.get(name)
        if root:
            for suffix in ("tools/msobjdump/msobjdump", "tools/msobjdump", "bin/msobjdump"):
                candidate = Path(root) / suffix
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
    raise RuntimeError("PTO TVM-FFI requires CANN's msobjdump to extract the device ELF. Source CANN's set_env.sh.")


def _run(command: list[str], stage: str) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if env.get_default_verbose():
        print(shlex.join(command))
    if result.returncode:
        raise RuntimeError(f"PTO {stage} failed.\nCommand: {shlex.join(command)}\n{result.stdout}\n{result.stderr}")


def _read_device_binary(directory: Path) -> bytearray:
    # msobjdump also writes metadata files. Only accept a linked ELF, never a
    # relocatable fat object or a host shared library (ET_DYN).
    binaries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if data[:6] == b"\x7fELF\x02\x01" and len(data) >= 64 and int.from_bytes(data[16:18], "little") == 2:
                binaries.append((path, data))
    if len(binaries) != 1:
        raise RuntimeError(
            f"PTO requires one executable device ELF per entry, got {len(binaries)} in {directory}. "
            "Check the PTOAS/CANN toolchain and msobjdump output."
        )
    return bytearray(binaries[0][1])


def compile_pto(code, kernel_name, target, pass_config=None):
    """Return an executable device ELF for one PTO entry, without a host launcher.

    PTOAS currently emits a CCE fat object. Bisheng performs its final device
    link; msobjdump extracts the executable consumed by AscendModuleCreate.
    The temporary host container is never loaded or used to execute kernels.
    """
    from tilelang.cache.ascend_binary_cache import AscendBinaryCache

    arch = os.environ.get("PTO_ARCH", "a5").strip().lower()
    if arch != "a5":
        raise ValueError(f"PTO TVM-FFI only supports PTO_ARCH=a5, got {arch!r}")
    flags = shlex.split(os.environ.get("PTO_FLAGS", ""))
    flags += bisheng.normalize_options((pass_config or {}).get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS))
    if not any(flag == "--pto-backend" or flag.startswith("--pto-backend=") for flag in flags):
        flags.append("--pto-backend=vpto")
    key = AscendBinaryCache.make_key(
        code=code,
        target_kind="pto",
        target_arch=str(target),
        compile_format="aibin",
        options=[str(kernel_name), arch, *flags],
        linker_options=["--cce-fatobj-link"],
    )
    cached = AscendBinaryCache.load(key, "aibin")
    if cached is not None:
        return bytearray(cached)
    compiler = bisheng.find_bisheng_path()
    extractor = _find_msobjdump()
    temp = utils.tempdir(keep_for_debug=not env.should_cleanup_temp_files())
    root = Path(temp.temp_dir)
    pto = root / "kernel.pto"
    fatobj = root / "kernel.fatobj.o"
    container = root / "kernel.so"
    extracted = root / "device"
    extracted.mkdir()
    compile_ptodsl_source_to_pto(code, str(kernel_name), root / "kernel.py", pto)
    _run(["ptoas", f"--pto-arch={arch}", *flags, str(pto), "-o", str(fatobj)], "device compilation")
    _run([compiler, "-fPIC", "-shared", "--cce-fatobj-link", str(fatobj), "-o", str(container)], "device link")
    _run([extractor, "--extract-elf", str(container), "--out-dir", str(extracted)], "device ELF extraction")
    binary = _read_device_binary(extracted)
    AscendBinaryCache.save(key, "aibin", binary)
    return binary


def compile_ptodsl_source_to_pto(
    ptodsl_source: str,
    kernel_name: str,
    src_path: str | Path,
    out_path: str | Path,
) -> None:
    src_path = Path(src_path)
    out_path = Path(out_path)
    src_path.write_text(ptodsl_source, encoding="utf-8")

    script = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import sys
        import traceback

        src_path = pathlib.Path(sys.argv[1])
        kernel_name = sys.argv[2]
        out_path = pathlib.Path(sys.argv[3])
        module_name = "_tilelang_ptodsl_compile"

        try:
            spec = importlib.util.spec_from_file_location(module_name, src_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception:
            traceback.print_exc()
            sys.exit(2)

        try:
            kernel = getattr(module, kernel_name)
            if not callable(getattr(kernel, "compile", None)):
                raise TypeError(f"PTODSL entry `{kernel_name}` is not a compilable PTO kernel handle")
            pto_text = kernel.compile().mlir_text()
            out_path.write_text(pto_text, encoding="utf-8")
        except Exception:
            traceback.print_exc()
            sys.exit(3)
        """
    )

    python_bin = sys.executable
    result = subprocess.run(
        [python_bin, "-c", script, str(src_path), kernel_name, str(out_path)],
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
            f"Kernel: {kernel_name}\n"
            f"Command: {python_bin} -c <ptodsl-compile-script> {src_path} <kernel-name> {out_path}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
