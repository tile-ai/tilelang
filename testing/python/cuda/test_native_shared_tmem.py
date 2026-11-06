"""Native TMEM arenas preserve b32 addresses, alias origins and release paths."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, columns, dtype="float32", rows=128, scope="shared.tmem"):
    return ir.Var(name, tirx.TensorType((rows, columns), dtype, scope=scope))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def intrinsic(name, *args):
    return ir.Call(name, args, ty_args=[ir.PrimType("void")])


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def calls(value, name):
    return [item for item in objects(value, ir.Call) if isinstance(item.op, ir.Op) and item.op.name == name]


def statements(values):
    return [ir.Evaluate(ir.TensorLoad(value, [0, 0])) for value in values]


def allocated_columns(value):
    return [int(call.args[1]) for call in calls(value, "tl.ptx_init_tensor_memory")]


def addresses(value):
    return [
        node.value
        for node in objects(value, ir.Evaluate)
        if ir.is_prim_expr(node.value) and node.value.ty.dtype == "uint32" and not isinstance(node.value, ir.Call)
    ]


def evaluate_integer(expression, *, variables=None, word_values=None):
    """Evaluate integer IR independently of the compiler's proof heuristics."""
    variables = {} if variables is None else variables
    word_values = {} if word_values is None else word_values
    if isinstance(expression, ir.IntImm):
        value = int(expression.value)
    elif isinstance(expression, ir.Var):
        value = variables[expression]
    elif isinstance(expression, ir.TensorLoad):
        assert all(evaluate_integer(index, variables=variables) == 0 for index in expression.indices)
        value = word_values[expression.source]
    elif isinstance(expression, ir.prim.Cast):
        value = evaluate_integer(expression.value, variables=variables, word_values=word_values)
    else:
        operations = {
            ir.prim.Add: lambda a, b: a + b,
            ir.prim.Sub: lambda a, b: a - b,
            ir.prim.Mul: lambda a, b: a * b,
            ir.prim.FloorDiv: lambda a, b: a // b,
            ir.prim.FloorMod: lambda a, b: a % b,
            ir.prim.LShift: lambda a, b: a << b,
            ir.prim.RShift: lambda a, b: a >> b,
            ir.prim.BitwiseAnd: lambda a, b: a & b,
            ir.prim.BitwiseOr: lambda a, b: a | b,
        }
        operation = operations[type(expression)]
        value = operation(
            evaluate_integer(expression.a, variables=variables, word_values=word_values),
            evaluate_integer(expression.b, variables=variables, word_values=word_values),
        )
    dtype = expression.ty.dtype
    value %= 1 << dtype.bits
    if str(dtype).startswith("int") and value >= 1 << (dtype.bits - 1):
        value -= 1 << dtype.bits
    return value


def check_address(expression, offset, word=None, variables=None):
    (load,) = objects(expression, ir.TensorLoad)
    assert int(load.indices[0]) == 0
    assert load.source.ty.dtype == "uint32"
    assert load.source.ty.scope.name == "shared"
    if word is not None:
        assert load.source.same_as(word)
    # PTX encodes datapath in bits [22:16] and b32 column in the low bits.
    # Vary the arena base separately from each logical tensor's origin.
    for base in (0, 32, 128):
        actual = evaluate_integer(expression, variables=variables, word_values={load.source: base})
        assert actual == base + offset
    return load.source


def apply(body, tmp_path, *, params=(), thread=None, two_cta=False):
    thread = ir.Var("tx", "int32") if thread is None else thread
    if two_cta:
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
            {"node": None, "key": "tl.kernel", "value": {"use_2cta": integer(1), "cluster_dims": [2, 1, 1]}}
        )
        body = [ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)]
    region = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(128)], [thread], {}, body)
    function = (
        tirx.Function(list(params), [region])
        .with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}))
        .with_attr("global_symbol", "main")
    )
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    operation = tvm_ffi.get_global_func("tl.cuda.transform.LowerSharedTmem")()
    module = operation(tvm_tirx.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(operation(module), module)
    assert ir.save_json(function) == before
    result = module["main"]
    assert not any(value.ty.scope.name == "shared.tmem" for value in objects(result, ir.Var) if isinstance(value.ty, tirx.TensorType))
    assert not calls(result, "tl.deallocate_tmem")
    (tmp_path / "before.json").write_text(before)
    (tmp_path / "after.json").write_text(ir.save_json(result))
    return result


def test_scale_buffers_pack_after_a_wide_accumulator(tmp_path):
    values = [tensor("accumulator", 384), *(tensor(name, 4, "uint32") for name in ("sfa", "sfb", "sfc"))]
    result = apply([*(allocate(value) for value in values), *statements(values)], tmp_path)
    assert allocated_columns(result) == [512]
    assert [int(call.args[1]) for call in calls(result, "tl.ptx_deallocate_tensor_memory")] == [512]
    expressions = addresses(result)
    assert len(expressions) == 4
    word = check_address(expressions[0], 0)
    for expression, offset in zip(expressions[1:], (384, 388, 392), strict=True):
        check_address(expression, offset, word)
    assert len(objects(result, ir.Bind)) == 1


@pytest.mark.parametrize("columns", [[128, 4, 4], [384, 4, 4, 4], [128, 128, 128], [32, 32, 32, 32, 32], [256, 8], [64, 64, 4]])
def test_arena_packing_never_increases_the_column_budget(columns, tmp_path):
    values = [tensor("tmem" + str(index), count) for index, count in enumerate(columns)]
    result = apply([*(allocate(value) for value in values), *statements(values)], tmp_path)
    actual = allocated_columns(result)
    separate = sum(max(32, 1 << (count - 1).bit_length()) for count in columns)
    assert sum(actual) <= separate
    assert actual == sorted(actual, reverse=True)
    assert all(count >= 32 and count <= 512 and count & (count - 1) == 0 for count in actual)


def test_equal_cost_keeps_allocations_separate_and_widest_first(tmp_path):
    values = [tensor("narrow", 32), tensor("wide", 256)]
    result = apply([*(allocate(value) for value in values), *statements(values)], tmp_path)
    assert allocated_columns(result) == [256, 32]
    assert not check_address(addresses(result)[0], 0).same_as(check_address(addresses(result)[1], 0))


@pytest.mark.parametrize(
    "dtype,bits,columns", [("float32", 32, 64), ("float16", 16, 128), ("uint8", 8, 256), ("int4", 4, 512), ("float64", 64, 32)]
)
def test_physical_coordinates_match_the_ptx_address_oracle(dtype, bits, columns, tmp_path):
    value = tensor("tmem", columns, dtype)
    alignment = max(1, 32 // bits)
    coordinates = [(row, column) for row in (0, 1, 31, 127) for column in (0, alignment, columns - alignment)]
    result = apply([allocate(value), *(ir.Evaluate(ir.TensorLoad(value, [row, column])) for row, column in coordinates)], tmp_path)
    assert allocated_columns(result) == [64]
    for expression, (row, column) in zip(addresses(result), coordinates, strict=True):
        check_address(expression, (row << 16) | (column * bits // 32))


def test_a_partial_final_column_is_rounded_up(tmp_path):
    value = tensor("packed4", 257, "int4")
    result = apply([allocate(value), ir.Evaluate(ir.TensorLoad(value, [0, 256]))], tmp_path)
    assert allocated_columns(result) == [64]
    check_address(addresses(result)[0], 32)


def test_unaligned_subword_address_is_rejected(tmp_path):
    value = tensor("half", 128, "float16")
    with pytest.raises(ValueError, match="must start on a b32 column"):
        apply([allocate(value), ir.Evaluate(ir.TensorLoad(value, [0, 1]))], tmp_path)


def test_dynamic_coordinates_keep_the_arena_origin(tmp_path):
    accumulator, scale = tensor("accumulator", 384), tensor("scale", 8, "uint32")
    thread = ir.Var("tx", "int32")
    result = apply(
        [allocate(accumulator), allocate(scale), ir.Evaluate(ir.TensorLoad(scale, [thread // 32, (thread % 2) * 4]))],
        tmp_path,
        thread=thread,
    )
    expression = addresses(result)[0]
    for tx in range(128):
        check_address(expression, ((tx // 32) << 16) + (tx % 2) * 4 + 384, variables={thread: tx})


def test_native_cp_base_offset_pair_is_rebased(tmp_path):
    accumulator, scale = tensor("accumulator", 384), tensor("scale", 8, "uint32")
    shared = tensor("shared", 16, "uint32", rows=1, scope="shared")
    issue = intrinsic("tl.ptx_tcgen05_cp_warpx4", shared.data, scale.data, integer(4))
    result = apply([allocate(accumulator), allocate(scale), allocate(shared), ir.Evaluate(issue)], tmp_path)
    (call,) = calls(result, "tl.ptx_tcgen05_cp_warpx4")
    assert call.args[1].op.name == "tirx.tensor_data_ptr"
    assert call.args[1].args[0].ty.scope.name == "shared"
    assert int(sym.Analyzer().simplify(call.args[2])) == 388


@pytest.mark.parametrize("kind", ["tensor", "pointer", "byte_offset", "access_ptr"])
def test_aliases_keep_one_backing_allocation_and_their_origins(kind, tmp_path):
    value = tensor("allocation", 128, "float32")
    alias = tensor("alias", 64, "float16")
    shared = tensor("source", 16, "uint32", rows=1, scope="shared")
    pointer = value.data
    expected = 0
    if kind == "byte_offset":
        pointer = ir.Call("tirx.ptr_byte_offset", [pointer, integer(16)])
        expected = 4
    elif kind == "access_ptr":
        pointer = ir.Call("tl.ptr_access", [pointer, integer(8), integer(4), integer(1)])
        expected = 8
    body = [allocate(value), allocate(shared)]
    if kind == "pointer":
        ptr = ir.Var("base", pointer.ty)
        body.append(ir.Bind(ptr, pointer))
        pointer = ptr
    if kind != "pointer":
        body.append(ir.Bind(alias, ir.Call("tirx.decl_tensor", [pointer], ty_args=[alias.ty])))
        pointer = alias.data
    body.append(ir.Evaluate(intrinsic("tl.ptx_tcgen05_cp_warpx4", shared.data, pointer, integer(0))))
    result = apply(body, tmp_path)
    assert allocated_columns(result) == [128]
    call = calls(result, "tl.ptx_tcgen05_cp_warpx4")[0]
    assert evaluate_integer(call.args[2]) == expected


@pytest.mark.parametrize("kind", ["direct", "first_warp", "returns", "one_of_two"])
def test_explicit_release_preserves_the_source_lifetime_contract(kind, tmp_path):
    first, second = tensor("first", 128), tensor("second", 64)
    thread = ir.Var("tx", "int32")
    release = ir.Evaluate(intrinsic("tl.deallocate_tmem", first.data))
    body = [allocate(first)]
    if kind == "one_of_two":
        body.append(allocate(second))
    if kind in ("first_warp", "returns"):
        branch = [release]
        if kind == "returns":
            branch.append(ir.Evaluate(ir.Call("tirx.gpu_thread_return", [])))
        body.append(ir.If(thread < 32, branch, None))
    else:
        body.append(release)
    result = apply(body, tmp_path, thread=thread)
    counts = [int(call.args[1]) for call in calls(result, "tl.ptx_deallocate_tensor_memory")]
    expected = {"direct": [128], "first_warp": [128], "returns": [128, 128], "one_of_two": [128, 64]}
    assert counts == expected[kind]


def test_explicit_release_prevents_packing_other_live_tensors(tmp_path):
    accumulator, scale = tensor("accumulator", 384), tensor("scale", 4, "uint32")
    result = apply(
        [
            allocate(accumulator),
            allocate(scale),
            *statements([accumulator, scale]),
            ir.Evaluate(intrinsic("tl.deallocate_tmem", scale.data)),
        ],
        tmp_path,
    )
    assert allocated_columns(result) == [512, 32]
    first = check_address(addresses(result)[0], 0)
    assert not check_address(addresses(result)[1], 0).same_as(first)


def test_two_cta_allocation_and_release_keep_matching_metadata(tmp_path):
    value = tensor("two_cta", 128)
    result = apply([allocate(value), *statements([value])], tmp_path, two_cta=True)
    for name in ("tl.ptx_init_tensor_memory", "tl.ptx_deallocate_tensor_memory"):
        (call,) = calls(result, name)
        assert len(call.args) == 2
        assert int(call.attrs["use_2cta"]) == 1
        assert "annotation_schema" not in call.attrs
    assert len(calls(result, "tl.cluster_sync")) == 1


def test_unpacked_raw_base_cannot_silently_address_another_tensor(tmp_path):
    accumulator, scale = tensor("accumulator", 384), tensor("scale", 4, "uint32")
    with pytest.raises(ValueError, match="Packed or offset TMEM base"):
        apply([allocate(accumulator), allocate(scale), ir.Evaluate(scale.data)], tmp_path)


@pytest.mark.parametrize("problem", ["rows", "columns", "budget", "rank"])
def test_invalid_physical_footprints_fail_before_codegen(problem, tmp_path):
    if problem == "rank":
        values = [ir.Var("rank1", tirx.TensorType((128,), "float32", scope="shared.tmem"))]
    else:
        values = [tensor("first", 513 if problem == "columns" else 384, rows=129 if problem == "rows" else 128)]
        if problem == "budget":
            values.append(tensor("second", 384))
    with pytest.raises(ValueError, match="TMEM|datapath|b32"):
        apply([*(allocate(value) for value in values)], tmp_path)


def test_volatile_view_preserves_the_shared_address_word_contract(tmp_path):
    value = tensor("storage", 128)
    view = tensor("view", 64, "float16")
    declaration = ir.Call("tirx.decl_tensor", [value.data], attrs={"tirx.volatile": True}, ty_args=[view.ty])
    result = apply([allocate(value), ir.Bind(view, declaration), *statements([view])], tmp_path)
    (binding,) = objects(result, ir.Bind)
    assert bool(binding.value.attrs["tirx.volatile"])


def test_foreign_allocation_attributes_are_not_merged_away(tmp_path):
    values = [tensor("accumulator", 128), tensor("policy", 4), tensor("plain", 4)]
    body = [
        allocate(values[0]),
        ir.Bind(values[1], ir.Call("tirx.alloc_tensor", [], attrs={"external_policy": 7}, ty_args=[values[1].ty])),
        allocate(values[2]),
        *statements(values),
    ]
    result = apply(body, tmp_path)
    assert allocated_columns(result) == [128, 32, 32]
    (policy,) = [node for node in objects(result, ir.Bind) if node.var.name == "policy"]
    assert int(policy.value.attrs["external_policy"]) == 7


def test_source_origin_is_consumed_before_tmem_arena_packing(tmp_path):
    values = [tensor("accumulator", 384), tensor("scale", 4, "uint32")]
    allocations = [
        ir.Bind(value, ir.Call("tirx.alloc_tensor", [], attrs={"tl.source_block_alloc": integer(1)}, ty_args=[value.ty]))
        for value in values
    ]
    result = apply([*allocations, *statements(values)], tmp_path)
    assert allocated_columns(result) == [512]
    word = check_address(addresses(result)[0], 0)
    check_address(addresses(result)[1], 384, word)
    assert all("tl.source_block_alloc" not in call.attrs for call in calls(result, "tirx.alloc_tensor"))


@pytest.mark.parametrize("marker", [integer(0), ir.StringImm("unexpected")])
def test_invalid_source_allocation_origin_is_rejected(marker, tmp_path):
    value = tensor("value", 32)
    allocation = ir.Bind(value, ir.Call("tirx.alloc_tensor", [], attrs={"tl.source_block_alloc": marker}, ty_args=[value.ty]))
    with pytest.raises(ValueError, match="tl.source_block_alloc must be the source allocation marker"):
        apply([allocation, *statements([value])], tmp_path)


def test_unlowered_tmem_store_is_rejected(tmp_path):
    value = tensor("storage", 128)
    with pytest.raises(ValueError, match="TMEM stores must be lowered"):
        apply([allocate(value), ir.TensorStore(value, [0, 0], ir.FloatImm("float32", 1.0))], tmp_path)


@pytest.mark.parametrize("placement", ["annotation", "attribute", "kernel"])
@pytest.mark.parametrize("include_shared", [False, True])
def test_consumed_tmem_layout_metadata_does_not_retype_other_storage(placement, include_shared, tmp_path):
    from tilelang.layout import Layout

    value = tensor("tmem", 64)
    shared = tensor("ordinary_shared", 64, scope="shared")
    layout = Layout([128, 64], lambda i, j: [i, j])
    mapping = {value: layout}
    safe_values = {value: ir.FloatImm("float32", 0.0)}
    if include_shared:
        mapping[shared] = layout
        safe_values[shared] = ir.FloatImm("float32", 2.0)
    metadata = {"layout_map": mapping, "safe_value_map": safe_values}
    body = [allocate(value), allocate(shared)]
    if placement == "annotation":
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(metadata)
        body.append(ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema})))
        body.extend(statements([value]))
    else:
        nested = statements([value])
        if placement == "kernel":
            records = [("tl.kernel", {**metadata, "cluster_dims": [1, 1, 1]})]
        else:
            records = list(metadata.items())
        for key, contents in records:
            schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": key, "value": contents})
            nested = [ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, nested)]
        body.extend(nested)
    result = apply(body, tmp_path)
    remaining = []
    for call in calls(result, "tl.layout_map"):
        remaining.append({"layout_map": call.attrs["layout_map"]})
    for call in calls(result, "tl.annotation"):
        remaining.append(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"]))
    for region in objects(result, ir.RegionStmt):
        if region.op.name == "tl.attr":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[-1], region.attrs["annotation_schema"])
            if fields["key"] == "tl.kernel":
                remaining.append(fields["value"])
                assert list(fields["value"]["cluster_dims"]) == [1, 1, 1]
            else:
                remaining.append({fields["key"]: fields["value"]})
    maps = {key: item[key] for item in remaining for key in ("layout_map", "safe_value_map") if key in item}
    if include_shared:
        assert len(maps) == 2
        assert all(len(item) == 1 and next(iter(item)).same_as(shared) for item in maps.values())
        ir.assert_structural_equal(maps["layout_map"][shared], metadata["layout_map"][shared])
        assert float(maps["safe_value_map"][shared].value) == 2.0
    else:
        assert not maps


def test_cuda_sdk_compiles_a_real_native_tmem_arena(tmp_path):
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess

    values = [tensor("accumulator", 384), tensor("scale", 4, "uint32")]
    result = apply([*(allocate(value) for value in values), *statements(values)], tmp_path).with_attr("global_symbol", "tmem_arena")
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="c")
    result = result.with_attr("target", target)
    split = tirx.transform.SplitHostDevice()(tvm_tirx.IRModule({"main": result}))
    assert tirx.analysis.check_well_formed(split)
    kernels = {symbol: value for symbol, value in split.functions.items() if int(value.attrs.get("calling_conv", 0)) == 2}
    assert len(kernels) == 1
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm_tirx.IRModule(kernels))
    assert tirx.analysis.check_well_formed(lowered)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        lowered, tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    source = module.inspect_source("cuda")
    (tmp_path / "tmem.cu").write_text(source)
    assert source.count("tl::tmem_allocate(") == 1
    assert source.count("tl::tmem_deallocate(") == 1
    nvcc = shutil.which("nvcc")
    root = Path(__file__).resolve().parents[3]
    cutlass = Path(os.environ.get("TILELANG_TEST_CUTLASS_INCLUDE", root / "3rdparty/cutlass/include"))
    if nvcc is None or not (cutlass / "cute").is_dir():
        pytest.skip("CUDA SDK and fixed CUTLASS headers are needed for device compilation")
    command = [
        nvcc,
        "-std=c++17",
        "--cubin",
        "-arch=sm_100a",
        f"-I{root / 'src'}",
        f"-I{cutlass}",
        str(tmp_path / "tmem.cu"),
        "-o",
        str(tmp_path / "tmem.cubin"),
    ]
    compilation = subprocess.run(command, text=True, capture_output=True, check=False)
    (tmp_path / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": compilation.returncode}, indent=2))
    (tmp_path / "nvcc.log").write_text(compilation.stdout + compilation.stderr)
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
