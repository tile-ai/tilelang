import itertools
import re

import tvm_ffi

from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


def test_cuda_functions_have_deterministic_declaration_and_definition_order():
    reference = None
    names = ["alpha", "middle", "omega"]
    for order in itertools.permutations(names):
        functions = {}
        for name in order:
            function = tirx.Function([], ir.Evaluate(ir.const(0)))
            function = function.with_attr("global_symbol", name).with_attr("calling_conv", 2)
            functions[name] = function.with_attr("tirx.kernel_launch_params", [])
        module = ir.IRModule(functions).with_attr("codegen.source_only", True)
        source = tvm_ffi.get_global_func("tl.codegen.build_cuda")(module, Target({"kind": "cuda", "arch": "sm_80"})).inspect_source()
        declarations = re.findall(r"void\s+(alpha|middle|omega)\s*\([^)]*\)\s*;", source)
        definitions = re.findall(r"void\s+(alpha|middle|omega)\s*\([^)]*\)\s*\{", source)
        assert declarations == names
        assert definitions == names
        if reference is None:
            reference = source
        else:
            assert source == reference
