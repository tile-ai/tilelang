"""Lower PTODSL kernels to MLIR text.

Usage: python -m tilelang.contrib.ptodsl_compile <src.ptodsl.py> <kernel-names.json> <out.pto>
"""

import importlib.util
import json
import pathlib
import sys
import traceback

src_path = pathlib.Path(sys.argv[1])
kernel_names = json.loads(sys.argv[2])
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
