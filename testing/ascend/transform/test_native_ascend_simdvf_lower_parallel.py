"""Native SIMD_VF lowering, checked against independent scalar memory oracles.

The interpreter covers the instruction contracts used by this pass. It does not
stand in for CANN compilation or NPU execution; those require a matching runner.
Input seed, exact integer oracles and float tolerances are fixed below.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import Axis
from testing.ascend._block import attribute as _scope_attribute, attribute_fields as _scope_fields


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape, dtype="float32", layout=None):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope="shared", layout=layout))


def region(value, starts=None, extents=None):
    starts = starts if starts is not None else [0] * len(value.ty.shape)
    extents = extents if extents is not None else value.ty.shape
    return ir.TensorRegion(value, [ir.Range.from_min_extent(a, b) for a, b in zip(starts, extents)], ir.TensorRegionType())


def attribute(key, value, body):
    return _scope_attribute(integer(0), key, value, body)


def vf(body, simd=True):
    return attribute("tl.sblock", {"name": "SIMD_VF" if simd else "SIMT_VF", "no_realize": True, "annotations": {}}, body)


def tileop(name, *operands, metadata=None):
    return ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"), ir.Op.get("tl.tileop." + name), list(operands), metadata or {}, ir.UnknownLoc()
        )
    )


def parallel(var, begin, end, body, step=1):
    return ir.For(var, integer(begin, str(var.ty)), integer(end, str(var.ty)), integer(step, str(var.ty)), ir.ForKind.PARALLEL, body)


def objects(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


def calls(value, name):
    return [expr for expr in objects(value, ir.Call) if isinstance(expr.op, ir.Op) and expr.op.name == name]


def apply(params, body, tmp_path):
    before = tirx.Function(params, body).with_attr("global_symbol", "simdvf_native")
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    operation = tvm_ffi.get_global_func("tl.transform.AscendSimdVFLowerParallel")()
    initial = tvm_tirx.IRModule({"simdvf_native": before})
    result = operation(initial)
    ir.assert_structural_equal(operation(ir.load_json(ir.save_json(initial))), result)
    assert tirx.analysis.check_well_formed(result)
    for value in objects(result, ir.Call):
        value.validate()
    ir.assert_structural_equal(operation(result), result)
    assert ir.save_json(before) == saved
    (tmp_path / "before.json").write_text(saved)
    (tmp_path / "after.json").write_text(ir.save_json(result["simdvf_native"]))
    return result["simdvf_native"]


@dataclass
class Address:
    tensor: object
    offset: int
    extent: int
    mask: int


class InstructionOracle:
    """Execute vector instruction semantics while checking every memory span."""

    def __init__(self, memory, scalars=None):
        self.memory = {key: value.reshape(-1) for key, value in memory.items()}
        self.values = dict(scalars or {})
        self.trace = []

    def offset(self, tensor_value, indices):
        index = [int(self.expr(i)) for i in indices]
        shape = [int(self.expr(i)) for i in tensor_value.ty.shape]
        if tensor_value.ty.layout is None:
            assert len(index) == len(shape)
            offset = 0
            for i, extent in zip(index, shape):
                assert 0 <= i < extent, (i, extent)
                offset = offset * extent + i
            return offset
        layout = tensor_value.ty.layout
        strides = [self.expr(it.stride) for it in layout.shard]
        offset = self.expr(layout.offset.get(Axis.get("m"), integer(0)))
        return int(offset + sum(i * s for i, s in zip(index, strides)))

    def expr(self, value):
        if isinstance(value, ir.Var):
            return self.values[value]
        if isinstance(value, (ir.IntImm, ir.FloatImm, ir.StringImm)):
            return value.value
        if isinstance(value, ir.TensorLoad):
            return self.memory[value.source][self.offset(value.source, value.indices)]
        if isinstance(value, ir.prim.Cast):
            return np.asarray(self.expr(value.value), dtype=str(value.ty)).item()
        for name, operation in (
            ("Add", np.add),
            ("Sub", np.subtract),
            ("Mul", np.multiply),
            ("Div", np.divide),
            ("FloorDiv", np.floor_divide),
            ("FloorMod", np.mod),
            ("Min", np.minimum),
            ("Max", np.maximum),
            ("LT", np.less),
            ("EQ", np.equal),
        ):
            if type(value).__name__ == name:
                return operation(self.expr(value.a), self.expr(value.b))
        if isinstance(value, ir.Call):
            name = value.op.name
            operands = value.args
            if value.attrs is not None and "operand_count" in value.attrs:
                operands = operands[: int(value.attrs["operand_count"])]
            if name == "tl.access_ptr":
                load, extent, mask = operands
                return Address(load.source, self.offset(load.source, load.indices), int(self.expr(extent)), int(self.expr(mask)))
            args = [self.expr(arg) for arg in operands]
            if name == "tl.simd.pset":
                assert args[1] == "PAT_ALL"
                return int(args[0])
            if name == "tl.simd.vld":
                address, distribution = args
                lanes = value.ty.dtype.lanes
                count = 1 if distribution.startswith("BRC_B") else lanes
                self.check(address, count, 1)
                loaded = self.memory[address.tensor][address.offset : address.offset + count].copy()
                return np.repeat(loaded, lanes) if count == 1 else loaded
            if name == "tl.simd.vdup":
                return np.full(value.ty.dtype.lanes, args[0], dtype=str(value.ty).split("x")[0])
            if name == "tl.simd.vcvt":
                assert args[2:] == ["ROUND_R", "RS_ENABLE", "PART_EVEN", "MODE_ZEROING"]
                output = np.zeros(value.ty.dtype.lanes, dtype=str(value.ty).split("x")[0])
                output[::2] = args[0]
                return output
            if name == "tl.simd.vsts":
                address, vector, _, distribution = args
                if distribution == "PK_B32":
                    vector = vector[::2]
                elif distribution.startswith("ONEPT"):
                    vector = vector[:1]
                else:
                    assert distribution.startswith("NORM")
                self.check(address, len(vector), 2)
                self.memory[address.tensor][address.offset : address.offset + len(vector)] = vector
                return None
            binary = {
                "vadd": np.add,
                "vsub": np.subtract,
                "vmul": np.multiply,
                "vdiv": np.divide,
                "vmax": np.maximum,
                "vmin": np.minimum,
                "vand": np.bitwise_and,
                "vor": np.bitwise_or,
                "vxor": np.bitwise_xor,
            }
            op = name.removeprefix("tl.simd.")
            if op in binary:
                return binary[op](args[0], args[1])
            unary = {"vabs": np.abs, "vneg": np.negative, "vexp": np.exp, "vln": np.log, "vsqrt": np.sqrt}
            if op in unary:
                return unary[op](args[0])
            if op == "vpack":
                source = args[0]
                dtype = np.dtype(str(value.ty).split("x")[0])
                assert args[1] == "LOWER" and source.dtype.kind in "iu" and dtype.kind in "iu"
                assert source.dtype.itemsize == 2 * dtype.itemsize
                # LOWER packs truncated source lanes into the low half. Only
                # lane zero is observed by the horizontal reduction store.
                output = np.zeros(value.ty.dtype.lanes, dtype=dtype)
                output[: source.size] = source.astype(dtype)
                return output
            if op in ("vcadd", "vcmax", "vcmin"):
                source = args[0]
                output = np.zeros(value.ty.dtype.lanes, dtype=str(value.ty).split("x")[0])
                output[0] = np.sum(source, dtype=output.dtype) if op == "vcadd" else np.max(source) if op == "vcmax" else np.min(source)
                return output
        raise AssertionError(f"Unmodeled instruction expression: {value}")

    def check(self, address, count, access):
        assert address.mask & access
        assert count <= address.extent, (count, address.extent)
        assert address.offset >= 0 and address.offset + count <= len(self.memory[address.tensor]), (
            address.offset,
            count,
            len(self.memory[address.tensor]),
        )
        self.trace.append((address.tensor.name, address.offset, count, access))

    def stmt(self, stmt):
        if isinstance(stmt, ir.SeqStmt):
            for child in stmt.seq:
                self.stmt(child)
        elif isinstance(stmt, ir.RegionStmt):
            self.stmt(stmt.body)
        elif isinstance(stmt, ir.For):
            for index in range(int(self.expr(stmt.begin)), int(self.expr(stmt.end)), int(self.expr(stmt.step))):
                self.values[stmt.var] = index
                self.stmt(stmt.body)
        elif isinstance(stmt, ir.If):
            branch = stmt.then_branch if self.expr(stmt.cond) else stmt.else_branch
            if branch is not None:
                self.stmt(branch)
        elif isinstance(stmt, ir.Bind):
            self.values[stmt.var] = self.expr(stmt.value)
        elif isinstance(stmt, ir.Evaluate):
            self.expr(stmt.value)
        else:
            raise AssertionError(f"Unmodeled statement: {stmt}")


def execute(function, memory, tmp_path, scalars=None):
    oracle = InstructionOracle(memory, scalars)
    oracle.stmt(function.body)
    (tmp_path / "memory-trace.txt").write_text("\n".join(map(str, oracle.trace)) + "\n")
    return oracle


@pytest.mark.parametrize("dtype", ["float32", "float16", "int16", "int32"])
def test_parallel_nonzero_begin_preserves_scalar_computation(dtype, tmp_path):
    lanes = 2048 // ir.PrimType(dtype).dtype.bits
    source, output = tensor("source", (3 + 2 * lanes + 1,), dtype), tensor("output", (3 + 2 * lanes + 1,), dtype)
    i = ir.Var("i", "int32")
    value = ir.TensorLoad(source, [i]) * ir.const(2, dtype) + ir.const(1, dtype)
    function = apply([source, output], [vf([parallel(i, 3, 3 + 2 * lanes, [ir.TensorStore(output, [i], value)])])], tmp_path)
    data = (np.arange(3 + 2 * lanes + 1) % 19 - 9).astype(dtype)
    result = np.full_like(data, -91)
    expected = result.copy()
    expected[3:-1] = data[3:-1] * 2 + 1
    execute(function, {source: data, output: result}, tmp_path)
    np.testing.assert_array_equal(result, expected)
    assert not any(loop.kind == ir.ForKind.PARALLEL for loop in objects(function, ir.For))


@pytest.mark.parametrize("outer_is_vector", [False, True])
def test_parallel_two_dimensions_keep_begin_step_broadcast_and_identity(outer_is_vector, tmp_path):
    a, b, out = tensor("a", (7, 134)), tensor("b", (7,)), tensor("out", (7, 134))
    row, col = ir.Var("i", "int64"), ir.Var("i", "int64")
    value = ir.TensorLoad(a, [row, col]) + ir.TensorLoad(b, [row])
    store = ir.TensorStore(out, [row, col], value)
    nested = (
        parallel(col, 3, 131, [parallel(row, 2, 7, [store], 2)])
        if outer_is_vector
        else parallel(row, 2, 7, [parallel(col, 3, 131, [store])], 2)
    )
    function = apply([a, b, out], [vf([nested])], tmp_path)
    data = np.arange(7 * 134, dtype="float32").reshape(7, 134) / 8
    bias = np.arange(7, dtype="float32")
    result = np.full((7, 134), -91, dtype="float32")
    expected = result.copy()
    expected[2:7:2, 3:131] = data[2:7:2, 3:131] + bias[2:7:2, None]
    execute(function, {a: data, b: bias, out: result}, tmp_path)
    np.testing.assert_array_equal(result, expected)
    assert len({loop.var for loop in objects(function, ir.For)}) == 2


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("operator", ["exp", "log", "sqrt", "fabs"])
def test_parallel_unary_arithmetic_matches_numpy(dtype, operator, tmp_path):
    lanes = 2048 // ir.PrimType(dtype).dtype.bits
    src, dst = tensor("src", (lanes,), dtype), tensor("dst", (lanes,), dtype)
    i = ir.Var("i", "int32")
    expression = ir.Call("prim." + operator, [ir.TensorLoad(src, [i])], ty=ir.PrimType(dtype))
    function = apply([src, dst], [vf([parallel(i, 0, lanes, [ir.TensorStore(dst, [i], expression)])])], tmp_path)
    data = np.linspace(0.25, 2, lanes, dtype=dtype)
    result = np.zeros_like(data)
    execute(function, {src: data, dst: result}, tmp_path)
    expected = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "fabs": np.abs}[operator](data)
    np.testing.assert_allclose(result, expected, rtol=2e-3 if dtype == "float16" else 1e-6, atol=0)


@pytest.mark.parametrize("version", [0, 1])
def test_fill_multibuffer_region_has_exact_address_and_untouched_neighbors(version, tmp_path):
    dst = tensor("dst", (2, 3, 128))
    slot = ir.Var("slot", "int64")
    function = apply([slot, dst], [vf([tileop("fill", region(dst, [slot, 0, 0], [1, 3, 128]), ir.const(7, "float32"))])], tmp_path)
    result = np.full((2, 3, 128), -91, dtype="float32")
    expected = result.copy()
    expected[version] = 7
    execute(function, {dst: result}, tmp_path, {slot: version})
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("to_dtype", ["float32", "float16"])
@pytest.mark.parametrize("op", ["copy", "ascend_copy"])
def test_full_copy_and_cast_keep_exact_element_footprint(to_dtype, op, tmp_path):
    src, dst = tensor("src", (2, 128)), tensor("dst", (4, 64), to_dtype)
    function = apply([src, dst], [vf([tileop(op, region(src), region(dst))])], tmp_path)
    data = np.linspace(-5, 5, 256, dtype="float32").reshape(2, 128)
    result = np.full((4, 64), -91, dtype=to_dtype)
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result, data.reshape(4, 64).astype(to_dtype))
    assert all(str(call.args[3].value) == ("PK_B32" if to_dtype == "float16" else "NORM_B32") for call in calls(function, "tl.simd.vsts"))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("operation", ["sum", "max", "min", "abssum", "absmax", "bitand", "bitor", "bitxor"])
@pytest.mark.parametrize("clear,keepdims", [(True, False), (False, True)])
def test_non_square_reduce_obeys_axis_operator_clear_and_output_shape(dim, operation, clear, keepdims, tmp_path):
    dtype = "int32" if operation.startswith("bit") else "float32"
    shape = (1, 128) if dim == 0 else (3, 1)
    if not keepdims:
        shape = (128,) if dim == 0 else (3,)
    src, dst = tensor("src", (3, 128), dtype), tensor("dst", shape, dtype)
    function = apply(
        [src, dst],
        [vf([tileop("reduce", region(src), region(dst), ir.StringImm(operation), integer(dim), ir.const(clear, "bool"))])],
        tmp_path,
    )
    rng = np.random.default_rng(271828)
    data = rng.integers(-8, 9, size=(3, 128)).astype(dtype)
    initial = rng.integers(-3, 4, size=shape).astype(dtype)
    result = initial.copy()
    source = np.abs(data) if operation.startswith("abs") else data
    binary = {
        "sum": np.add,
        "abssum": np.add,
        "max": np.maximum,
        "absmax": np.maximum,
        "min": np.minimum,
        "bitand": np.bitwise_and,
        "bitor": np.bitwise_or,
        "bitxor": np.bitwise_xor,
    }[operation]
    expected = binary.reduce(source, axis=dim, keepdims=keepdims)
    if not clear:
        expected = binary(expected, initial)
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result, expected)
    assert not calls(function, "tl.tileop.reduce")


@pytest.mark.parametrize("dtype", ["float16", "int16"])
def test_dim1_halfwidth_accumulate_broadcasts_one_element(dtype, tmp_path):
    src, dst = tensor("src", (3, 256), dtype), tensor("dst", (3,), dtype)
    function = apply(
        [src, dst], [vf([tileop("reduce", region(src), region(dst), ir.StringImm("sum"), integer(1), ir.const(False, "bool"))])], tmp_path
    )
    data = (np.arange(3 * 256).reshape(3, 256) % 7 - 3).astype(dtype)
    result = np.array([4, -2, 7], dtype=dtype)
    expected = np.sum(data, axis=1, dtype=dtype) + result
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result, expected)
    assert any(call.args[1].value == "BRC_B16" for call in calls(function, "tl.simd.vld"))


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize("clear", [True, False])
def test_dim1_halfwidth_integer_reduce_preserves_wrap(dtype, clear, tmp_path):
    src, dst = tensor("src", (3, 256), dtype), tensor("dst", (3,), dtype)
    function = apply(
        [src, dst], [vf([tileop("reduce", region(src), region(dst), ir.StringImm("sum"), integer(1), ir.const(clear, "bool"))])], tmp_path
    )
    limits = np.iinfo(dtype)
    values = [limits.max, limits.min, -129 if limits.min < 0 else 257]
    data = np.array([[value] * 256 for value in values], dtype=dtype)
    result = np.array([limits.max, limits.min, 127], dtype=dtype)
    expected = []
    for index, row in enumerate(data):
        total = sum(int(value) for value in row)
        if not clear:
            total += int(result[index])
        wrapped = total % (1 << 16)
        if limits.min < 0 and wrapped >= 1 << 15:
            wrapped -= 1 << 16
        expected.append(wrapped)
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result, np.array(expected, dtype=dtype))


def test_parallel_metadata_inside_leaf_is_preserved(tmp_path):
    src, dst = tensor("src", (128,)), tensor("dst", (128,))
    i = ir.Var("i", "int32")
    function = apply(
        [src, dst], [vf([parallel(i, 0, 128, [attribute("trace", i, [ir.TensorStore(dst, [i], ir.TensorLoad(src, [i]))])])])], tmp_path
    )
    trace_regions = []
    for node in objects(function, ir.RegionStmt):
        metadata = _scope_fields(node)
        if metadata["key"] == "trace":
            trace_regions.append(metadata)
    assert len(trace_regions) == 1
    assert any(var.name == "vchunk" for var in objects(trace_regions[0]["value"], ir.Var))


@pytest.mark.parametrize(
    "case",
    [
        "odd_extent",
        "nonunit_step",
        "strided_load",
        "third_dimension",
        "dynamic_extent",
        "fill_alignment",
        "fill_noncontiguous",
        "copy_partial",
        "copy_transform",
        "reduce_shape",
    ],
)
def test_unsupported_native_simdvf_contracts_fail_before_codegen(case, tmp_path):
    src, dst = tensor("src", (256,)), tensor("dst", (256,))
    i, j, k, n = [ir.Var(name, "int32") for name in ("i", "j", "k", "n")]
    body = [parallel(i, 0, 64, [ir.TensorStore(dst, [i], ir.TensorLoad(src, [i]))])]
    params = [src, dst]
    match = "SIMD_VF"
    if case == "odd_extent":
        body = [parallel(i, 0, 63, [ir.TensorStore(dst, [i], ir.TensorLoad(src, [i]))])]
    elif case == "nonunit_step":
        body = [parallel(i, 0, 128, [ir.TensorStore(dst, [i], ir.TensorLoad(src, [i]))], 2)]
    elif case == "strided_load":
        body = [parallel(i, 0, 64, [ir.TensorStore(dst, [i], ir.TensorLoad(src, [2 * i]))])]
    elif case == "third_dimension":
        body = [parallel(i, 0, 1, [parallel(j, 0, 1, [parallel(k, 0, 64, [ir.TensorStore(dst, [k], ir.TensorLoad(src, [k]))])])])]
    elif case == "dynamic_extent":
        body = [ir.For(i, integer(0), n, integer(1), ir.ForKind.PARALLEL, body[0].body)]
        params.append(n)
    elif case == "fill_alignment":
        body = [tileop("fill", region(dst, [1], [64]), ir.const(1, "float32"))]
        match = "32-byte-aligned"
    elif case == "fill_noncontiguous":
        dst = tensor("dst2", (2, 70))
        params = [dst]
        body = [tileop("fill", region(dst, [0, 0], [2, 64]), ir.const(1, "float32"))]
    elif case == "copy_partial":
        body = [tileop("copy", region(src, [0], [64]), region(dst, [0], [64]))]
    elif case == "copy_transform":
        body = [tileop("ascend_copy", region(src), region(dst), metadata={"transpose": True})]
    elif case == "reduce_shape":
        src, dst = tensor("matrix", (3, 128)), tensor("bad", (4,))
        params = [src, dst]
        body = [tileop("reduce", region(src), region(dst), ir.StringImm("sum"), integer(1), ir.const(True, "bool"))]
        match = "dst_total"
    with pytest.raises((ValueError, TypeError), match=match):
        apply(params, [vf(body)], tmp_path)


def test_simt_and_outside_regions_retain_parallel_loops(tmp_path):
    src, dst = tensor("src", (64,)), tensor("dst", (64,))
    i, j = ir.Var("i", "int32"), ir.Var("i", "int32")
    function = apply(
        [src, dst],
        [
            parallel(i, 0, 64, [ir.TensorStore(dst, [i], ir.TensorLoad(src, [i]))]),
            vf([parallel(j, 0, 64, [ir.TensorStore(dst, [j], ir.TensorLoad(src, [j]))])], simd=False),
        ],
        tmp_path,
    )
    assert len([loop for loop in objects(function, ir.For) if loop.kind == ir.ForKind.PARALLEL]) == 2
    assert not calls(function, "tl.simd.vld")


@pytest.mark.parametrize("operation", ["parallel", "fill", "copy", "cast_copy", "reduce"])
def test_real_cce_source_consumes_native_simdvf_lowering(operation, tmp_path):
    raw = ir.Var("raw", ir.PtrType(ir.PrimType("float32"), "shared"))
    src, dst = (
        tensor("src", (3, 128)),
        tensor("dst", (3,) if operation == "reduce" else (3, 128), "float16" if operation == "cast_copy" else "float32"),
    )
    raw_dst = ir.Var("raw_dst", ir.PtrType(dst.ty.dtype, "shared"))
    definitions = [
        ir.Bind(src, ir.Call("tirx.decl_tensor", [raw], ty_args=[src.ty])),
        ir.Bind(dst, ir.Call("tirx.decl_tensor", [raw_dst], ty_args=[dst.ty])),
    ]
    if operation == "parallel":
        i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
        body = [
            parallel(i, 0, 3, [parallel(j, 0, 128, [ir.TensorStore(dst, [i, j], ir.TensorLoad(src, [i, j]) + ir.const(1, "float32"))])])
        ]
    elif operation == "fill":
        body = [tileop("fill", region(dst), ir.const(1, "float32"))]
    elif operation in ("copy", "cast_copy"):
        body = [tileop("copy", region(src), region(dst))]
    else:
        body = [tileop("reduce", region(src), region(dst), ir.StringImm("sum"), integer(1), ir.const(True, "bool"))]
    function = apply([raw, raw_dst], definitions + [vf(body)], tmp_path).with_attr("calling_conv", 2)
    module = tvm_tirx.IRModule({"simdvf_native": function})
    with tvm_tirx.target.Target("ascend"):
        module = tirx.transform.LowerTensorLayouts()(module)
        module = tirx.transform.FlattenTensor()(module)
    module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(module)
    (tmp_path / "codegen-input.json").write_text(ir.save_json(module))
    codegen = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(module, tvm_tirx.target.Target("ascend"))
    source = codegen.inspect_source("asc")
    (tmp_path / "native_simdvf.asc").write_text(source)
    assert "__simd_vf__" in source and "simd_inst::vsts_" in source
    assert "tl.tileop" not in source
    if operation == "cast_copy":
        assert "vsts_pack_b32" in source and "vcvt" in source


def test_parallel_between_loop_metadata_and_annotations_survive(tmp_path):
    src, dst = tensor("src", (2, 128)), tensor("dst", (2, 128))
    row, col = ir.Var("row", "int32"), ir.Var("col", "int32")
    inner = ir.For(
        col,
        integer(0),
        integer(128),
        integer(1),
        ir.ForKind.PARALLEL,
        [ir.TensorStore(dst, [row, col], ir.TensorLoad(src, [row, col]))],
        {"vector_contract": 23},
    )
    outer = ir.For(
        row, integer(0), integer(2), integer(1), ir.ForKind.PARALLEL, [attribute("trace_between", row, [inner])], {"serial_contract": 17}
    )
    function = apply([src, dst], [vf([outer])], tmp_path)
    metadata = [_scope_fields(node) for node in objects(function, ir.RegionStmt)]
    (between,) = [item for item in metadata if item["key"] == "trace_between"]
    assert tvm_tirx.sym.Analyzer().simplify(between["value"]).name == "vserial"
    assert sorted(int(value) for node in objects(function, ir.For) for value in node.annotations.values()) == [17, 23]
    data = np.arange(256, dtype="float32").reshape(2, 128)
    result = np.zeros_like(data)
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result, data)


def test_nested_simt_vf_is_not_peeled_as_plain_metadata(tmp_path):
    src, dst = tensor("src", (2, 64)), tensor("dst", (2, 64))
    row, col = ir.Var("row", "int32"), ir.Var("col", "int32")
    inner = parallel(col, 0, 64, [ir.TensorStore(dst, [row, col], ir.TensorLoad(src, [row, col]))])
    body = vf([parallel(row, 0, 2, [vf([inner], simd=False)])])
    function = apply([src, dst], [body], tmp_path)
    assert len([node for node in objects(function, ir.For) if node.kind == ir.ForKind.PARALLEL]) == 2
    assert not calls(function, "tl.simd.vld")


def test_narrow_loop_index_uses_a_counter_that_can_represent_register_width(tmp_path):
    src, dst = tensor("src", (128,), "float16"), tensor("dst", (128,), "float16")
    i = ir.Var("i", "int8")
    index = i + integer(64, "int8")
    function = apply([src, dst], [vf([parallel(i, -64, 64, [ir.TensorStore(dst, [index], ir.TensorLoad(src, [index]))])])], tmp_path)
    (counter,) = objects(function, ir.For)
    assert counter.var.ty.dtype.bits >= 32
    data = np.arange(128, dtype="float16")
    result = np.full_like(data, -1)
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("byte_offset", [4, 32])
def test_fill_alignment_includes_native_int64_byte_offset(byte_offset, tmp_path):
    raw = ir.Var("raw", ir.PtrType(ir.PrimType("float32"), "shared"))
    view = tensor("view", (64,))
    pointer = ir.Call("tirx.ptr_byte_offset", [raw, integer(byte_offset, "int64")], ty=raw.ty)
    body = [
        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
        vf([tileop("fill", region(view), ir.const(1, "float32"))]),
    ]
    if byte_offset == 4:
        with pytest.raises(ValueError, match="32-byte-aligned"):
            apply([raw], body, tmp_path)
    else:
        function = apply([raw], body, tmp_path)
        assert len(calls(function, "tl.simd.vsts")) == 1
        assert calls(function, "tirx.ptr_byte_offset")[0].args[1].value == 32


def test_narrow_machine_wrap_is_not_mistaken_for_contiguous_memory(tmp_path):
    src, dst = tensor("src", (128,), "float16"), tensor("dst", (128,), "float16")
    i = ir.Var("i", "int8")
    index = i + integer(120, "int8")
    with pytest.raises(ValueError, match="contiguous physical vector dimension"):
        apply([src, dst], [vf([parallel(i, -64, 64, [ir.TensorStore(dst, [index], ir.TensorLoad(src, [index]))])])], tmp_path)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("operation", ["sum", "max", "min", "abssum", "absmax", "bitand", "bitor", "bitxor"])
@pytest.mark.parametrize("clear", [False, True])
def test_empty_reduction_uses_shared_identity_without_source_reads(dim, operation, clear, tmp_path):
    dtype = "int32" if operation.startswith("bit") else "float32"
    shape = (0, 128) if dim == 0 else (3, 0)
    output_shape = (128,) if dim == 0 else (3,)
    src, dst = tensor("src", shape, dtype), tensor("dst", output_shape, dtype)
    function = apply(
        [src, dst],
        [vf([tileop("reduce", region(src), region(dst), ir.StringImm(operation), integer(dim), ir.const(clear, "bool"))])],
        tmp_path,
    )
    result = np.arange(np.prod(output_shape), dtype=dtype).reshape(output_shape)
    expected = result.copy()
    if clear:
        identity = {"sum": 0, "abssum": 0, "absmax": 0, "max": -np.inf, "min": np.inf, "bitand": -1, "bitor": 0, "bitxor": 0}[operation]
        expected.fill(identity)
    execute(function, {src: np.empty(shape, dtype=dtype), dst: result}, tmp_path)
    np.testing.assert_array_equal(result, expected)
    assert not calls(function, "tl.simd.vld")


def test_unary_negation_keeps_signed_zero_instruction_semantics(tmp_path):
    src, dst = tensor("src", (64,)), tensor("dst", (64,))
    i = ir.Var("i", "int32")
    value = ir.prim.Sub(ir.const(0, "float32"), ir.TensorLoad(src, [i]))
    function = apply([src, dst], [vf([parallel(i, 0, 64, [ir.TensorStore(dst, [i], value)])])], tmp_path)
    data = np.resize(np.array([-0.0, 0.0, 1.0, -1.0], dtype="float32"), 64)
    result = np.empty_like(data)
    execute(function, {src: data, dst: result}, tmp_path)
    np.testing.assert_array_equal(result.view("uint32"), np.negative(data).view("uint32"))
    assert calls(function, "tl.simd.vneg")
