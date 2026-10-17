from dataclasses import dataclass

import pytest
import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def _lazy_kernel_factory(hidden: int, token_stride: int, sf_only: bool = False, cast_only: bool = False):
    @T.prim_func
    def kernel():
        T.evaluate(0)

    return kernel


def test_lazy_jit_cache_key_canonicalizes_defaults_and_call_forms():
    positional_defaults_key, _ = _lazy_kernel_factory.func.parse_args(512, 512)
    positional_explicit_key, _ = _lazy_kernel_factory.func.parse_args(512, 512, False, False)
    kwargs_key, _ = _lazy_kernel_factory.func.parse_args(
        hidden=512,
        token_stride=512,
        sf_only=False,
        cast_only=False,
    )
    reordered_kwargs_key, _ = _lazy_kernel_factory.func.parse_args(
        cast_only=False,
        sf_only=False,
        token_stride=512,
        hidden=512,
    )

    expected = ((512, 512, False, False), None)
    assert positional_defaults_key == expected
    assert positional_explicit_key == expected
    assert kwargs_key == expected
    assert reordered_kwargs_key == expected


def test_lazy_jit_call_form_cache_skips_rebinding(monkeypatch):
    sentinel_kernel = object()
    canonical_key, _ = _lazy_kernel_factory.func.parse_args(
        hidden=512,
        token_stride=512,
    )
    _lazy_kernel_factory.mode = "lazy"
    _lazy_kernel_factory.func.set_mode("lazy")
    _lazy_kernel_factory._kernel_cache[canonical_key] = sentinel_kernel
    _lazy_kernel_factory._call_form_cache.clear()

    assert _lazy_kernel_factory(hidden=512, token_stride=512) is sentinel_kernel

    def fail_parse_args(*args, **kwargs):
        raise AssertionError("call-form cache hit should skip parse_args")

    monkeypatch.setattr(_lazy_kernel_factory.func, "parse_args", fail_parse_args)
    assert _lazy_kernel_factory(hidden=512, token_stride=512) is sentinel_kernel


def test_lazy_jit_call_form_cache_requires_hashable_arguments():
    _lazy_kernel_factory.mode = "lazy"
    _lazy_kernel_factory.func.set_mode("lazy")
    _lazy_kernel_factory._call_form_cache.clear()

    with pytest.raises(TypeError, match="unhashable"):
        _lazy_kernel_factory(hidden=512, token_stride=512, opts=[1, 2])


def test_jit_argument_binding_reports_python_call_errors():
    with pytest.raises(TypeError, match="multiple values"):
        _lazy_kernel_factory.func.parse_args(512, hidden=512, token_stride=512)

    with pytest.raises(TypeError, match="missing a required argument"):
        _lazy_kernel_factory.func.parse_args(hidden=512)

    extra_const_key, _ = _lazy_kernel_factory.func.parse_args(
        hidden=512,
        token_stride=512,
        M=1024,
    )
    assert extra_const_key == ((512, 512, False, False, (("M", 1024),)), None)


@tilelang.jit
def _lazy_var_keyword_factory(M: int, **kwargs):
    @T.prim_func
    def kernel():
        T.evaluate(0)

    return kernel


def test_lazy_jit_var_keyword_kwargs_are_flattened_and_hashable():
    key, kernel_args = _lazy_var_keyword_factory.func.parse_args(
        128,
        N=64,
        opts=(1, 2),
    )
    hash(key)
    assert kernel_args == {}

    p1_key, tensor_args, compile_kwargs = _lazy_var_keyword_factory.func._parse_phase1_key(
        128,
        N=64,
        opts=(1, 2),
    )
    hash(p1_key)
    assert tensor_args == {}
    assert compile_kwargs == {"M": 128, "N": 64, "opts": (1, 2)}
    assert "kwargs" not in compile_kwargs


def test_lazy_jit_varargs_are_forwarded_to_factory():
    @tilelang.jit
    def make_kernel(*values, scale=1):
        value = sum(values) * scale

        @T.prim_func
        def kernel():
            T.evaluate(value)

        return kernel

    first = make_kernel.get_tir(1, 2)
    assert first.body.value.value == 3
    assert make_kernel.get_tir(1, 2, scale=1) is first
    assert make_kernel.get_tir(1, 2, scale=3).body.value.value == 9
    assert make_kernel.get_tir().body.value.value == 0


def test_lazy_jit_varargs_and_kwargs_preserve_specializations():
    @tilelang.jit
    def make_kernel(first, *values, scale=1, **metadata):
        value = (first + sum(values) + metadata.get("offset", 0)) * scale

        @T.prim_func
        def kernel():
            T.evaluate(value)

        return kernel

    first = make_kernel.get_tir(1, 2, scale=3, offset=4)
    assert first.body.value.value == 21
    second = make_kernel.get_tir(1, 2, 3, scale=2, offset=4)
    assert second.body.value.value == 20
    assert make_kernel.get_tir(1, 2, scale=3, offset=4) is first


@dataclass
class _UnhashableConfig:
    axis: int


_UNHASHABLE_VALUES = [
    pytest.param({"value": 1}, id="dict"),
    pytest.param([1, 2], id="list"),
    pytest.param({1, 2}, id="set"),
    pytest.param(({"value": 1},), id="nested-tuple"),
    pytest.param(_UnhashableConfig(axis=0), id="dataclass"),
]


@pytest.mark.parametrize("default", _UNHASHABLE_VALUES)
@pytest.mark.parametrize("keyword_only", [False, True])
def test_jit_rejects_unhashable_defaults(default, keyword_only):
    if keyword_only:

        def make_kernel(*, options=default):
            @T.prim_func
            def kernel():
                T.evaluate(0)

            return kernel

    else:

        def make_kernel(options=default):
            @T.prim_func
            def kernel():
                T.evaluate(0)

            return kernel

    with pytest.raises(TypeError, match="Default value for JIT compile-time parameter 'options' must be hashable"):
        tilelang.jit(make_kernel)


@pytest.mark.parametrize("value", _UNHASHABLE_VALUES)
@pytest.mark.parametrize("call_form", ["positional", "keyword", "tuning", "parse_args", "get_tir"])
def test_lazy_jit_rejects_unhashable_arguments(value, call_form):
    @tilelang.jit
    def make_kernel(options=None):
        raise AssertionError("unhashable arguments must be rejected before calling the factory")

    with pytest.raises(TypeError, match="JIT compile-time arguments must be hashable"):
        if call_form == "positional":
            make_kernel(value)
        elif call_form == "keyword":
            make_kernel(options=value)
        elif call_form == "tuning":
            make_kernel(__tune_params={"options": value})
        elif call_form == "parse_args":
            make_kernel.func.parse_args(value)
        else:
            make_kernel.get_tir(value)


def test_lazy_jit_rejects_unhashable_varargs_and_kwargs():
    @tilelang.jit
    def make_kernel(*options):
        @T.prim_func
        def kernel():
            T.evaluate(0)

        return kernel

    for parse in (make_kernel.func.parse_args, make_kernel.func._parse_phase1_key):
        with pytest.raises(TypeError, match="JIT compile-time arguments must be hashable"):
            parse([1, 2])

    for parse in (_lazy_var_keyword_factory.func.parse_args, _lazy_var_keyword_factory.func._parse_phase1_key):
        with pytest.raises(TypeError, match="JIT compile-time arguments must be hashable"):
            parse(128, opts={"axis": [1, 2]})


def test_lazy_jit_call_form_cache_checks_hashability_before_equality(monkeypatch):
    class UnhashableInt(int):
        __hash__ = None

    @tilelang.jit
    def make_kernel(value):
        @T.prim_func
        def kernel():
            T.evaluate(value)

        return kernel

    monkeypatch.setattr(make_kernel, "compile", make_kernel.get_tir)
    make_kernel(1)
    with pytest.raises(TypeError, match="JIT compile-time arguments must be hashable"):
        make_kernel(UnhashableInt(1))


@dataclass(frozen=True)
class _HashableConfig:
    value: int


@pytest.mark.parametrize("config_type", [tuple, _HashableConfig])
def test_lazy_jit_hashable_defaults_reuse_specializations(monkeypatch, config_type):
    first_options = (1,) if config_type is tuple else _HashableConfig(1)
    second_options = (2,) if config_type is tuple else _HashableConfig(2)

    @tilelang.jit
    def make_kernel(options=first_options):
        value = options[0] if config_type is tuple else options.value

        @T.prim_func
        def kernel():
            T.evaluate(value)

        return kernel

    # Skip backend compilation to keep this test portable.
    monkeypatch.setattr(make_kernel, "compile", make_kernel.get_tir)
    first = make_kernel()
    assert first.body.value.value == 1
    assert make_kernel(first_options) is first
    assert make_kernel(options=first_options) is first

    second = make_kernel(second_options)
    assert second.body.value.value == 2
    assert make_kernel(options=second_options) is second
    assert make_kernel() is first


@tilelang.jit
def _copy_kernel(A, B, block: int = 64, dtype: T.dtype = T.float32):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], dtype]
    B: T.Tensor[[M, N], dtype]

    with T.Kernel(T.ceildiv(M, block), T.ceildiv(N, block), threads=128) as (pid_m, pid_n):
        T.copy(A[pid_m * block, pid_n * block], B[pid_m * block, pid_n * block])


def test_eager_jit_phase1_key_excludes_tensor_args_and_applies_defaults():
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    p1_key, tensor_args, compile_kwargs = _copy_kernel.func._parse_phase1_key(a, b)

    assert p1_key == (64, T.float32)
    assert tensor_args == {"A": a, "B": b}
    assert compile_kwargs == {"block": 64, "dtype": T.float32}


@tilelang.jit
def _copy_kernel_with_var_keywords(A, B, block: int = 64, **metadata):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]

    with T.Kernel(T.ceildiv(M, block), T.ceildiv(N, block), threads=128) as (pid_m, pid_n):
        T.copy(A[pid_m * block, pid_n * block], B[pid_m * block, pid_n * block])


def test_eager_jit_var_keyword_compile_kwargs_are_flattened():
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    p1_key, tensor_args, compile_kwargs = _copy_kernel_with_var_keywords.func._parse_phase1_key(
        a,
        b,
        opts=(1, 2),
    )

    hash(p1_key)
    assert tensor_args == {"A": a, "B": b}
    assert compile_kwargs == {"block": 64, "opts": (1, 2)}
    assert "metadata" not in compile_kwargs


@pytest.mark.parametrize("value", _UNHASHABLE_VALUES)
def test_eager_jit_rejects_unhashable_compile_time_arguments(value):
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    with pytest.raises(TypeError, match="JIT compile-time arguments must be hashable"):
        _copy_kernel(a, b, block=value)
    with pytest.raises(TypeError, match="JIT compile-time arguments must be hashable"):
        _copy_kernel_with_var_keywords(a, b, opts=value)


def test_eager_jit_excludes_unhashable_runtime_tensor_defaults():
    class UnhashableTensor(torch.Tensor):
        __hash__ = None

    default = torch.empty(8, 8).as_subclass(UnhashableTensor)

    @tilelang.jit
    def kernel(A=default, block=64):
        A: T.Tensor[(8, 8), T.float32]
        T.evaluate(0)

    p1_key, tensor_args, compile_kwargs = kernel.func._parse_phase1_key()
    assert p1_key == (64,)
    assert tensor_args["A"] is default
    assert compile_kwargs == {"block": 64}

    explicit = torch.empty(8, 8).as_subclass(UnhashableTensor)
    explicit_key, tensor_args, _ = kernel.func._parse_phase1_key(explicit)
    assert explicit_key == p1_key
    assert tensor_args["A"] is explicit
