"""Lower PTODSL kernels to MLIR text.

Usage: python <tilelang>/contrib/ptodsl_compile.py <src.ptodsl.py> <kernel-names.json> <out.pto>

Also runnable as ``python -m tilelang.contrib.ptodsl_compile``.
"""

import importlib.util
import json
import pathlib
import sys
import traceback
import types

# A script run puts this file's directory (tilelang/contrib) first on sys.path,
# where the contrib/ptodsl would shadow the ``ptodsl`` package imported by the generated source.
# Drop it again so module resolution still finds the real ``ptodsl`` package;
# with ``-m`` sys.path is already clean and the guard is a no-op.
_script_dir = str(pathlib.Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _script_dir:
    sys.path.pop(0)

src_path = pathlib.Path(sys.argv[1])
kernel_names = json.loads(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])
module_name = "_tilelang_ptodsl_compile"

# The generated source runs ``from ptodsl import pto`` and ``import tilelang.contrib.ptodsl as tl``.
# ``tilelang.contrib.ptodsl`` depends on ptodsl only.
# Preload ``tilelang`` and ``tilelang.contrib`` with ``__path__`` only, no ``__init__`` so that
# import skips ``tilelang/__init__.py`` and its torch/tvm import chain (which is time-consuming).
# Already-imported state (``-m``) is left untouched.
_tilelang_spec = importlib.util.find_spec("tilelang")
if "tilelang" not in sys.modules and _tilelang_spec is not None:
    _tilelang_root = pathlib.Path(_tilelang_spec.submodule_search_locations[0])
    _tilelang_pkg = types.ModuleType("tilelang")
    _tilelang_pkg.__path__ = [str(_tilelang_root)]
    _tilelang_contrib = types.ModuleType("tilelang.contrib")
    _tilelang_contrib.__path__ = [str(_tilelang_root / "contrib")]
    sys.modules["tilelang"] = _tilelang_pkg
    sys.modules["tilelang.contrib"] = _tilelang_contrib

try:
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
            raise RuntimeError("PTO multi-kernel JIT requires ptodsl.pto.merge_jit_modules from ptoas-vmi 0.1.4 or newer")
        pto_text = str(merge_jit_modules(*kernels))
    out_path.write_text(pto_text, encoding="utf-8")
except Exception:
    traceback.print_exc()
    sys.exit(3)
