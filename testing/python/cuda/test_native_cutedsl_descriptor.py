"""Native descriptor carriers preserve MMA offsets and CTA-group selection."""

import ast

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang  # noqa: F401 -- register TileLang code generators


def _call(name, args):
    args = [ir.StringImm(arg) if isinstance(arg, str) else arg if isinstance(arg, ir.Expr) else ir.const(arg) for arg in args]
    return ir.Call(name, args, ty="void")


def _source(variant, use_2cta, alias):
    scope = "local.descriptor.wgmma" if variant.startswith("wgmma") else "local.descriptor.tcgen05_smem"
    descriptors = [ir.Var(name, tirx.TensorType([1], "uint64", scope=scope)) for name in ("desc_a", "desc_b")]
    body = [ir.Bind(var, ir.Call("tirx.alloc_tensor", [], ty_args=[var.ty])) for var in descriptors]
    operands = [ir.Call("tirx.tensor_data_ptr", [var]) for var in descriptors]
    if alias:
        aliases = [ir.Var(name, expr.ty) for name, expr in zip(("alias_a", "alias_b"), operands)]
        body.extend(ir.Bind(var, expr) for var, expr in zip(aliases, operands))
        operands = aliases
    a = ir.Var("tmem_a", ir.PtrType("uint32", "shared"))
    c = ir.Var("tmem_c", ir.PtrType("uint32", "shared"))
    da, db = operands
    if variant == "ts":
        args = ["bf16", a, 8, db, 64, c, 4, 136316048, 1, 0, 0, 0, 0, use_2cta]
        name, helper, descriptor_positions = "tl.ptx_tcgen05_mma_ts", "tcgen05mma_ts", [2]
    elif variant in ("ss", "ws"):
        args = ["bf16", da, 32, db, 64, c, 4, 136316048, 1, 0, 0, 0, 0, variant == "ws", use_2cta]
        name = "tl.ptx_tcgen05_mma_ss"
        helper, descriptor_positions = "tcgen05mma_ws_ss" if variant == "ws" else "tcgen05mma_ss", [1, 2]
    elif variant == "blockscaled":
        args = ["e4m3", da, 32, db, 64, c, 4, 136316048, 1, a, 0, a, 0, 0, 0, use_2cta]
        name, helper, descriptor_positions = "tl.ptx_tcgen05_mma_blockscaled_ss", "tcgen05mma_blockscaled_ss", [1, 2]
    elif variant == "wgmma_ss":
        args = ["m64n128k16", True, True, "bf16", "bf16", "fp32", da, 32, db, 64, c, 4, 1, True, True]
        name, helper, descriptor_positions = "tl.ptx_wgmma_ss", "wgmma_ss", [10, 11]
    else:
        assert variant == "wgmma_rs"
        args = ["m64n128k16", True, "bf16", "bf16", "fp32", a, 0, db, 64, c, 4, 1, True, True]
        name, helper, descriptor_positions = "tl.ptx_wgmma_rs", "wgmma_rs", [10]
    body.append(ir.Evaluate(_call(name, args)))
    function = tirx.Function(
        [a, c],
        body,
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "descriptor_carrier", "calling_conv": 2}),
    )
    module = tvm_ffi.get_global_func("target.build.tilelang_cutedsl_without_compile")(
        ir.IRModule({"descriptor_carrier": function}), Target({"kind": "cuda", "arch": "sm_100a"})
    )
    return module.inspect_source("cutedsl"), helper, descriptor_positions


@pytest.mark.parametrize("alias", [False, True], ids=["data", "alias"])
@pytest.mark.parametrize(
    "variant,use_2cta",
    [
        ("ts", False),
        ("ts", True),
        ("ss", False),
        ("ss", True),
        ("ws", False),
        ("blockscaled", False),
        ("blockscaled", True),
        ("wgmma_ss", False),
        ("wgmma_rs", False),
    ],
)
def test_native_mma_descriptor_carrier(variant, use_2cta, alias, tmp_path):
    source, helper, positions = _source(variant, use_2cta, alias)
    (tmp_path / "descriptor.py").write_text(source)
    tree = ast.parse(source)
    # A descriptor is an object with offset arithmetic; it is not a CuTe Tensor.
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
        and node.value.id.startswith(("desc_", "alias_"))
        for node in ast.walk(tree)
    )
    if alias:
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        for suffix in ("a", "b"):
            assert isinstance(assignments["alias_" + suffix], ast.Name)
            assert assignments["alias_" + suffix].id == "desc_" + suffix
    emitted = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == helper
    )
    expected_names = ["b"] if len(positions) == 1 else ["a", "b"]
    for position, suffix in zip(positions, expected_names):
        value = emitted.args[position]
        assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
        assert isinstance(value.left, ast.Name)
        assert value.left.id == ("alias_" if alias else "desc_") + suffix
        assert isinstance(value.right, ast.Constant) and value.right.value == (32 if suffix == "a" else 64)
    if variant in ("ts", "ss", "blockscaled"):
        group = next(keyword.value for keyword in emitted.keywords if keyword.arg == "use_2cta")
        assert isinstance(group, ast.Constant) and group.value is use_2cta
