from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from tvm import IRModule
from tvm.target import Target

from tilelang import tvm

DeviceCodegenFunc = Callable[[IRModule, Target], IRModule]
TargetPredicate = Callable[[Target], bool]


def global_func_device_codegen(global_func_name: str) -> DeviceCodegenFunc:
    def build(mod: IRModule, target: Target) -> IRModule:
        return tvm.ffi.get_global_func(global_func_name)(mod, target)

    return build


@dataclass(frozen=True, slots=True)
class DeviceCodegen:
    """Device codegen entry points for one backend target variant."""

    name: str
    build: DeviceCodegenFunc | None = None
    build_without_compile: DeviceCodegenFunc | None = None
    supports_target: TargetPredicate | None = None

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)

    def lower(self, mod: IRModule, target: Target, *, compile_device: bool) -> IRModule:
        build_func = self.build if compile_device else self.build_without_compile
        if build_func is None:
            mode = "with compilation" if compile_device else "without compilation"
            raise ValueError(f"Device codegen '{self.name}' for target '{target.kind.name}' does not support lowering {mode}")
        return build_func(mod, target)


_DEVICE_CODEGENS: dict[str, list[DeviceCodegen]] = {}
_LAZY_DEVICE_CODEGENS: dict[str, str] = {}
_LOADED_DEVICE_CODEGENS: set[str] = set()


def register_device_codegen(
    target_kind: str,
    codegen: DeviceCodegen,
    *,
    override: bool = False,
) -> DeviceCodegen:
    codegens = _DEVICE_CODEGENS.setdefault(target_kind, [])
    if override:
        codegens[:] = [item for item in codegens if item.name != codegen.name]
    elif any(item.name == codegen.name for item in codegens):
        raise ValueError(f"Device codegen {codegen.name!r} is already registered for target kind {target_kind!r}")
    codegens.append(codegen)
    return codegen


def register_lazy_device_codegen(target_kind: str, import_path: str) -> None:
    _LAZY_DEVICE_CODEGENS[target_kind] = import_path
    _LOADED_DEVICE_CODEGENS.discard(target_kind)


def _ensure_device_codegens_loaded(target_kind: str) -> None:
    if target_kind in _LOADED_DEVICE_CODEGENS:
        return
    import_path = _LAZY_DEVICE_CODEGENS.get(target_kind)
    if import_path is not None:
        import_module(import_path)
    _LOADED_DEVICE_CODEGENS.add(target_kind)


def _matching_device_codegens(target: Target) -> list[DeviceCodegen]:
    target_kind = target.kind.name
    _ensure_device_codegens_loaded(target_kind)
    return [codegen for codegen in _DEVICE_CODEGENS.get(target_kind, ()) if codegen.matches(target)]


def allowed_device_codegens_for_target(target: Target) -> list[str]:
    return [codegen.name for codegen in _matching_device_codegens(target)]


def resolve_device_codegen(target: Target) -> DeviceCodegen:
    matches = _matching_device_codegens(target)
    if not matches:
        target_kind = target.kind.name
        available = [codegen.name for codegen in _DEVICE_CODEGENS.get(target_kind, ())]
        options = ", ".join(available) if available else "<none>"
        raise ValueError(f"No device codegen registered for target '{target_kind}'. Available: {options}.")
    return matches[0]
