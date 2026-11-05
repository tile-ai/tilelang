"""Integer SizeVar source defaults are local TileLang ABI policy on native Vars."""

import gc

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang
from tilelang import language as T
from tilelang.jit.abi import FunctionABIPlan, ShapeBindings
from tilelang.language.eager.builder import substitute_primfunc


def defaults(function):
    return list(tvm_ffi.get_global_func("tl.CollectSizeVarDefaults")(function))


def source_default(variable):
    return tvm_ffi.get_global_func("tl.HasSizeVarDefault")(variable)


@pytest.mark.parametrize(
    "name",
    [
        "bool",
        "int8",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "float8_e4m3fn",
        "int8x4",
        "int32x4",
        "uint8x4",
        "float16x2",
        "float32x4",
        "bfloat16x2",
    ],
)
def test_source_sizevar_constructor_is_an_ordinary_native_var(name):
    value = getattr(T, name)(is_size_var=True)
    assert type(value) is ir.Var
    assert value.ty.dtype == name
    assert not hasattr(value, "nonnegative")
    assert source_default(value) is (value.ty.dtype.is_integer and value.ty.dtype.lanes == 1)
    plain = getattr(T, name)()
    assert not source_default(plain)
    # Private source provenance intentionally belongs to a completed Function,
    # not to the standalone serialization of an upstream Var.
    restored = ir.load_json(ir.save_json(value))
    assert not source_default(restored)
    assert tvm_ffi.structural_equal(value.ty, restored.ty)


def test_sizevar_default_does_not_enter_a_native_analyzer_or_outlive_an_attempt():
    value = T.int32(is_size_var=True)
    assert not sym.Analyzer().can_prove(value >= 0)
    binding = ShapeBindings(size_var_defaults=[value])
    binding.bind_linear(T.max(value, 0), 7, "shape[0]")
    assert binding.evaluate(value, "source variable") == 7
    assert all(isinstance(entry, sym.BindingConstraint) for entry in binding.context)
    explicit = ShapeBindings(size_var_defaults=[value])
    explicit.bind(value, -2, "scalar parameter")
    assert explicit.evaluate(T.max(value, 0), "dimension") == 0
    assert explicit.evaluate(value, "source variable") == -2
    assert not sym.Analyzer().can_prove(value >= 0)


def test_sizevar_keyword_does_not_change_supplied_values_or_vector_operands():
    negative = T.int32(-3, is_size_var=True)
    assert isinstance(negative, ir.IntImm)
    assert negative.value == -3
    assert sym.Analyzer().const_int_bound(negative).max_value == -3
    plain = T.int32()
    assert T.int32(plain, is_size_var=True).same_as(plain)
    assert not source_default(plain)
    casted = T.int64(plain, is_size_var=True)
    assert isinstance(casted, ir.Expr)
    assert casted.ty.dtype == "int64"
    assert not isinstance(casted, ir.Var)
    packed = T.int32x2(T.int32(1), T.int32(2), is_size_var=True)
    assert packed.ty.dtype == "int32x2"
    assert source_default(T.int32(None, is_size_var=True))


@pytest.mark.parametrize("dtype", [None, "int32", "float32", "tensormap"])
def test_source_handle_accepts_size_keyword_without_numeric_domain(dtype):
    plain = T.handle(dtype)
    marked = T.handle(dtype, is_size_var=True)
    assert type(marked) is ir.Var
    assert isinstance(marked.ty, ir.PtrType)
    assert tvm_ffi.structural_equal(plain.ty, marked.ty)
    assert not source_default(marked)
    with pytest.raises(TypeError):
        sym.Analyzer().const_int_bound(marked)


def test_source_pointer_alias_and_void_accept_size_keyword():
    pointer = T.ptr("int32", "shared", is_size_var=True)
    assert tvm_ffi.structural_equal(pointer.ty, T.ptr("int32", "shared").ty)
    assert not source_default(pointer)
    void = T.void(is_size_var=True)
    assert type(void) is ir.Var
    assert tvm_ffi.structural_equal(void.ty, T.void().ty)
    assert not source_default(void)


def test_complete_function_json_and_renew_def_keep_current_parameter_identities():
    n = T.int32(is_size_var=True)

    @T.prim_func
    def kernel(N: T.int32(is_size_var=True), A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        for i in T.serial(n):
            B[i] = A[i] + T.max(N, 0)

    for function in [kernel, ir.load_json(ir.save_json(kernel)), tirx.renew_def(kernel)]:
        marked = set(defaults(function))
        assert marked == {function.params[0], function.params[1].ty.shape[0]}
        assert function.params[1].ty.shape[0].same_as(function.params[2].ty.shape[0])
        assert tirx.analysis.check_well_formed(function)
        attr_vars = []
        tvm_ffi.structural_walk(function.attrs, (ir.Var, attr_vars.append), order="pre")
        assert attr_vars == []


def test_temporary_python_wrapper_does_not_lose_native_source_provenance():
    dimension = T.max(T.int32(is_size_var=True), 0)
    gc.collect()

    @T.prim_func
    def kernel(A: T.Tensor((dimension,), "int32")):
        T.evaluate(0)

    marked = defaults(kernel)
    assert len(marked) == 1
    plan = FunctionABIPlan(kernel)
    _, bindings = plan.bind_inputs([np.zeros(7, dtype="int32")])
    assert bindings.evaluate(marked[0], "temporary source variable") == 7


def test_attaching_defaults_does_not_turn_a_declaration_into_a_definition():
    n = T.int32(is_size_var=True)
    declaration = tirx.Function([n], [], attrs=ir.DictAttrs({"declaration": True}))
    result = tvm_ffi.get_global_func("tl.AttachSizeVarDefaults")(declaration)
    assert result.same_as(declaration)
    assert len(result.body.seq) == 0
    assert defaults(result) == []
    # This verifier requires executable definitions, so preserving a native
    # declaration also preserves its diagnostic instead of manufacturing a
    # function body to satisfy the checker.
    for function in (declaration, result):
        with pytest.raises(Exception, match="is a declaration, but a definition is required"):
            tirx.analysis.check_well_formed(function)


def test_packed_pass_keeps_native_declarations_and_their_calling_convention():
    from tvm_tirx.target import Target

    variable = ir.Var("n", "int32")
    target = Target("c")
    declaration = tirx.Function(
        [variable],
        [],
        ret_type=ir.PrimType("int32"),
        attrs=ir.DictAttrs({"declaration": True, "global_symbol": "external", "target": target.with_host(target)}),
    )
    module = tilelang.transform.MakePackedAPI()(ir.IRModule({"external": declaration}))
    result = module["external"]
    assert result.params[0].same_as(variable)
    assert len(result.params) == 1
    assert len(result.body.seq) == 0
    assert result.attrs["declaration"]
    assert "calling_conv" not in result.attrs
    assert "tl.source_host_module" not in module.attrs


def test_substitute_remaps_carrier_without_marking_unrelated_same_name_vars():
    n = T.int32(is_size_var=True)
    m = T.int32()
    assert n.name == m.name

    @T.prim_func
    def kernel(A: T.Tensor((T.max(n, 0),), "int32"), B: T.Tensor((T.max(m, 0),), "int32")):
        T.evaluate(0)

    assert set(defaults(kernel)) == {n}
    replacement = ir.Var("replacement", "int32")
    remapped = substitute_primfunc(kernel, {n: replacement})
    assert set(defaults(remapped)) == {replacement}
    specialized = substitute_primfunc(kernel, {n: 3})
    assert defaults(specialized) == []
    assert tirx.analysis.check_well_formed(remapped)
    assert tirx.analysis.check_well_formed(specialized)
    with pytest.raises(ValueError, match="Cannot resolve"):
        FunctionABIPlan(remapped).bind_inputs([np.zeros(3, dtype="int32"), np.zeros(3, dtype="int32")])
    assert not source_default(m)


def test_eager_reconstruction_keeps_size_defaults_per_function():
    n = T.int32(is_size_var=True)

    @T.prim_func(eager_jit=True)
    def kernel(A: T.Tensor((T.max(n, 0),), "int32"), B: T.Tensor((T.max(n, 0),), "int32")):
        for i in T.serial(A.shape[0]):
            B[i] = A[i] + 3

    kernel.set_mode("eager")
    for extent in [1, 7, 1]:
        arrays = [np.zeros(extent, dtype="int32"), np.zeros(extent, dtype="int32")]
        function = kernel.get_tir(*arrays)
        assert tirx.analysis.check_well_formed(function)
        _, bindings = FunctionABIPlan(function).bind_inputs(arrays)
        for variable in defaults(function):
            assert bindings.evaluate(variable, "eager shape") == extent


def test_source_sizevar_nonlinear_shape_executes_through_public_jit():
    import torch

    n = T.int32(is_size_var=True)

    @T.prim_func
    def kernel(A: T.Tensor((T.max(n, 0),), "int32"), B: T.Tensor((T.max(n, 0),), "int32")):
        for i in T.serial(A.shape[0]):
            B[i] = A[i] + 3

    compiled = tilelang.compile(kernel, target="c", target_host="c", execution_backend="tvm_ffi")
    for extent in [0, 1, 7]:
        source = torch.arange(extent, dtype=torch.int32)
        output = torch.full((extent,), -991, dtype=torch.int32)
        compiled(source, output)
        torch.testing.assert_close(output, source + 3, rtol=0, atol=0)


@pytest.mark.parametrize("pointer", [T.handle, T.ptr])
def test_source_sizevar_pointer_annotations_execute_through_public_jit(pointer):
    import torch

    @T.prim_func
    def kernel(a: pointer("int32", is_size_var=True), b: pointer("int32", is_size_var=True)):
        A = T.match_buffer(a, (8,), "int32")
        B = T.match_buffer(b, (8,), "int32")
        for i in T.serial(8):
            B[i] = A[i] + 5

    compiled = tilelang.compile(kernel, target="c", target_host="c", execution_backend="tvm_ffi")
    source = torch.arange(8, dtype=torch.int32)
    output = torch.full((8,), -991, dtype=torch.int32)
    compiled(source, output)
    torch.testing.assert_close(output, source + 5, rtol=0, atol=0)
