import importlib
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm


@T.prim_func
def tiny_cpu_kernel(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        T.copy(A, B)


@T.prim_func
def annotated_cpu_kernel(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
    T.annotate_pass_configs(
        {
            tilelang.PassConfigKey.TL_CPU_ARCH: "riscv",
            tilelang.PassConfigKey.TL_FORCE_LET_INLINE: True,
        }
    )
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        T.copy(A, B)


@T.prim_func
def annotated_cpu_kernel_aarch64(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
    T.annotate_pass_configs(
        {
            tilelang.PassConfigKey.TL_CPU_ARCH: "aarch64",
            tilelang.PassConfigKey.TL_FORCE_LET_INLINE: True,
        }
    )
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        T.copy(A, B)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_repo_file(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text()


def _capture_lower_context(monkeypatch, func=tiny_cpu_kernel, pass_configs=None, outer_config=None, instruments=None):
    observed_configs = []
    lower_module = importlib.import_module("tilelang.engine.lower")
    original_lower_and_legalize = lower_module.LowerAndLegalize

    def capture_pass_context(mod, target):
        observed_configs.append(dict(tvm.transform.PassContext.current().config))
        return original_lower_and_legalize(mod, target)

    monkeypatch.setattr(lower_module, "LowerAndLegalize", capture_pass_context)

    with tvm.transform.PassContext(config=outer_config or {}, instruments=instruments or []), tvm.target.Target("c"):
        artifact = tilelang.lower(
            func,
            target="c",
            target_host="c",
            pass_configs=pass_configs,
        )

    return artifact, observed_configs


def test_lower_accepts_cpu_arch_pass_config():
    with tvm.target.Target("c"):
        artifact = tilelang.lower(
            tiny_cpu_kernel,
            target="c",
            target_host="c",
            pass_configs={"tl.cpu_arch": "riscv"},
        )

    assert artifact.kernel_source is not None
    assert artifact.kernel_source.strip()


def test_lower_without_cpu_arch_still_works():
    with tvm.target.Target("c"):
        artifact = tilelang.lower(tiny_cpu_kernel, target="c", target_host="c")

    assert artifact.kernel_source is not None
    assert artifact.kernel_source.strip()


def test_lower_keeps_runtime_only_positional_argument_compatibility():
    with tvm.target.Target("c"):
        artifact = tilelang.lower(tiny_cpu_kernel, "c", "c", True)

    assert artifact.kernel_source is not None
    assert artifact.kernel_source.strip()
    assert artifact.params is None


def test_cpu_arch_pass_config_does_not_change_scalar_cpu_codegen():
    with tvm.target.Target("c"):
        default_artifact = tilelang.lower(tiny_cpu_kernel, target="c", target_host="c")
        configured_artifact = tilelang.lower(
            tiny_cpu_kernel,
            target="c",
            target_host="c",
            pass_configs={"tl.cpu_arch": "riscv"},
        )

    assert default_artifact.kernel_source == configured_artifact.kernel_source


def test_lower_preserves_outer_pass_context_instruments_and_merges_config(monkeypatch):
    trace = []

    @tvm.instrument.pass_instrument
    class Probe:
        def enter_pass_ctx(self):
            trace.append("enter")

        def exit_pass_ctx(self):
            trace.append("exit")

        def run_before_pass(self, mod, info):
            if info.name == "tl.LowerTileOp":
                trace.append("before-lower")

        def run_after_pass(self, mod, info):
            if info.name == "tl.LowerTileOp":
                trace.append("after-lower")

    outer_config = {tilelang.PassConfigKey.TL_FORCE_LET_INLINE: True}
    artifact, observed_configs = _capture_lower_context(
        monkeypatch,
        pass_configs={"tl.cpu_arch": "riscv"},
        outer_config=outer_config,
        instruments=[Probe()],
    )

    assert artifact.kernel_source is not None
    assert observed_configs
    assert trace == ["enter", "before-lower", "after-lower", "exit"]
    seen_config = observed_configs[-1]
    assert bool(seen_config["tl.force_let_inline"])
    assert seen_config["tl.cpu_arch"] == "riscv"


def test_target_get_cpu_arch_is_observable_during_lowering_and_cpu_only(monkeypatch):
    lower_module = importlib.import_module("tilelang.engine.lower")
    original_lower_and_legalize = lower_module.LowerAndLegalize
    target_get_cpu_arch = tvm.ffi.get_global_func("tl.TargetGetCPUArch")
    observed_arches = []

    def capture_cpu_arch(mod, target):
        observed_arches.append(target_get_cpu_arch(target))
        return original_lower_and_legalize(mod, target)

    monkeypatch.setattr(lower_module, "LowerAndLegalize", capture_cpu_arch)

    with tvm.target.Target("c"):
        tilelang.lower(
            tiny_cpu_kernel,
            target="c",
            target_host="c",
            pass_configs={"tl.cpu_arch": "riscv"},
        )
    assert observed_arches[-1] == "riscv"

    with tvm.target.Target("c"):
        tilelang.lower(tiny_cpu_kernel, target="c", target_host="c")
    assert observed_arches[-1] is None

    assert target_get_cpu_arch(tvm.target.Target("cuda")) is None


def test_lower_honors_function_level_pass_configs_and_explicit_overrides(monkeypatch):
    _, observed_configs = _capture_lower_context(monkeypatch, func=annotated_cpu_kernel)
    assert observed_configs
    assert observed_configs[-1]["tl.cpu_arch"] == "riscv"
    assert bool(observed_configs[-1]["tl.force_let_inline"])

    _, override_configs = _capture_lower_context(
        monkeypatch,
        func=annotated_cpu_kernel,
        pass_configs={"tl.cpu_arch": "aarch64"},
    )
    assert override_configs
    assert override_configs[-1]["tl.cpu_arch"] == "aarch64"
    assert bool(override_configs[-1]["tl.force_let_inline"])


def test_lower_honors_function_level_pass_configs_for_irmodule_input(monkeypatch):
    mod = tvm.IRModule({"main": annotated_cpu_kernel})

    _, observed_configs = _capture_lower_context(monkeypatch, func=mod)
    assert observed_configs
    assert observed_configs[-1]["tl.cpu_arch"] == "riscv"
    assert bool(observed_configs[-1]["tl.force_let_inline"])

    _, override_configs = _capture_lower_context(
        monkeypatch,
        func=mod,
        pass_configs={"tl.cpu_arch": "aarch64"},
    )
    assert override_configs
    assert override_configs[-1]["tl.cpu_arch"] == "aarch64"
    assert bool(override_configs[-1]["tl.force_let_inline"])


def test_lower_honors_function_level_pass_configs_for_irmodule_with_unannotated_helper(monkeypatch):
    mod = tvm.IRModule(
        {
            "main": annotated_cpu_kernel,
            "helper": tiny_cpu_kernel,
        }
    )

    _, observed_configs = _capture_lower_context(monkeypatch, func=mod)
    assert observed_configs
    assert observed_configs[-1]["tl.cpu_arch"] == "riscv"
    assert bool(observed_configs[-1]["tl.force_let_inline"])


def test_lower_allows_explicit_override_for_conflicting_multifunc_irmodule(monkeypatch):
    mod = tvm.IRModule(
        {
            "main": annotated_cpu_kernel,
            "other": annotated_cpu_kernel_aarch64,
        }
    )

    _, observed_configs = _capture_lower_context(
        monkeypatch,
        func=mod,
        pass_configs={"tl.cpu_arch": "x86_64"},
    )
    assert observed_configs
    assert observed_configs[-1]["tl.cpu_arch"] == "x86_64"
    assert bool(observed_configs[-1]["tl.force_let_inline"])


def test_lower_rejects_conflicting_function_level_pass_configs_for_multifunc_irmodule():
    mod = tvm.IRModule(
        {
            "main": annotated_cpu_kernel,
            "other": annotated_cpu_kernel_aarch64,
        }
    )

    with (
        pytest.raises(
            ValueError,
            match="Conflicting function-level tilelang_pass_configs found in IRModule",
        ),
        tvm.target.Target("c"),
    ):
        tilelang.lower(mod, target="c", target_host="c")


def test_lower_rejects_unregistered_pass_config_key():
    with pytest.raises(AttributeError, match="Invalid config option 'tl.not_registered'"), tvm.target.Target("c"):
        tilelang.lower(
            tiny_cpu_kernel,
            target="c",
            target_host="c",
            pass_configs={"tl.not_registered": "value"},
        )


def test_cpu_arch_plumbing_is_declared_and_registered():
    assert 'TL_CPU_ARCH = "tl.cpu_arch"' in _read_repo_file("tilelang/transform/pass_config.py")
    assert 'kCPUArch = "tl.cpu_arch"' in _read_repo_file("src/op/builtin.h")
    assert "TVM_REGISTER_PASS_CONFIG_OPTION(kCPUArch, String);" in _read_repo_file("src/op/builtin.cc")
