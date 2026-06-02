from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tvm import IRModule
from tvm.target import Target

TargetLike = str | dict[str, object] | Target
LowerFunc = Callable[[IRModule, Target], IRModule]
CodegenFunc = Callable[[IRModule, Target], IRModule]
TargetPredicate = Callable[[Target], bool]
TargetNormalizer = Callable[[TargetLike], Target | None]
FeatureQuery = Callable[[Target], object]
AvailabilityCheck = Callable[[], bool]
CallbackRegistrar = Callable[[], None]
ExecutionBackendSelector = Callable[[Target], str]


def _always_available() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class ExecutionBackendSpec:
    """JIT execution policy owned by a compiler backend."""

    name: str
    adapter: str | None = None
    is_available: AvailabilityCheck = _always_available
    supports_target: TargetPredicate | None = None
    enable_host_codegen: bool = False
    enable_device_compile: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExecutionBackendSpec.name must not be empty")
        if self.adapter is None:
            object.__setattr__(self, "adapter", self.name)

    def matches(self, target: Target) -> bool:
        if self.supports_target is not None:
            return self.supports_target(target)
        return True


@dataclass(frozen=True, slots=True)
class Backend:
    """Compiler backend descriptor.

    A backend owns the target-specific lowering pipeline, codegen hooks,
    callback registration, and feature queries for one or more TVM target kinds.
    """

    name: str
    target_kinds: tuple[str, ...]
    import_path: str | None = None

    pipeline: object | LowerFunc | None = None
    supports_target: TargetPredicate | None = None
    normalize_target: TargetNormalizer | None = None
    is_available: AvailabilityCheck = _always_available
    features: Mapping[str, FeatureQuery] = field(default_factory=dict)

    device_codegen: CodegenFunc | None = None
    device_codegen_without_compile: CodegenFunc | None = None
    host_pre_codegen: CodegenFunc | None = None
    register_callbacks: CallbackRegistrar | None = None

    execution_backends: Mapping[str, ExecutionBackendSpec] = field(default_factory=dict)
    default_execution_backend: str | ExecutionBackendSelector | None = None
    jit_execution_backends: tuple[str, ...] = ()
    cmake_name: str | None = None
    language_modules: tuple[str, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Backend.name must not be empty")
        if not self.target_kinds:
            raise ValueError(f"Backend {self.name!r} must register at least one target kind")
        object.__setattr__(self, "target_kinds", tuple(self.target_kinds))
        execution_backends = dict(self.execution_backends)
        if not execution_backends and self.jit_execution_backends:
            execution_backends = {name: ExecutionBackendSpec(name=name, adapter=name) for name in tuple(self.jit_execution_backends)}
        object.__setattr__(self, "execution_backends", execution_backends)
        object.__setattr__(self, "jit_execution_backends", tuple(execution_backends))
        object.__setattr__(self, "language_modules", tuple(self.language_modules))

    def matches(self, target: Target) -> bool:
        if self.supports_target is not None:
            return self.supports_target(target)
        return target.kind.name in self.target_kinds

    def lower(self, mod: IRModule, target: Target) -> IRModule:
        if self.pipeline is None:
            raise ValueError(f"Backend {self.name!r} has no pass pipeline")
        if hasattr(self.pipeline, "lower"):
            return self.pipeline.lower(mod, target)  # type: ignore[no-any-return, attr-defined]
        return self.pipeline(mod, target)  # type: ignore[misc, no-any-return]

    def codegen(self, mod: IRModule, target: Target, *, compile: bool) -> IRModule:
        hook = self.device_codegen if compile else self.device_codegen_without_compile
        if hook is None:
            mode = "compiled" if compile else "source"
            raise ValueError(f"Backend {self.name!r} has no {mode} device codegen hook")
        return hook(mod, target)

    def allowed_execution_backends(self, target: Target, *, include_unavailable: bool = True) -> tuple[str, ...]:
        """Return execution backend names that this backend allows for *target*."""

        specs = [spec for spec in self.execution_backends.values() if spec.matches(target)]
        if not include_unavailable:
            specs = [spec for spec in specs if spec.is_available()]
        specs.sort(key=lambda spec: spec.priority, reverse=True)
        return tuple(spec.name for spec in specs)

    def resolve_execution_backend(self, requested: str | None, target: Target) -> ExecutionBackendSpec:
        """Resolve a requested JIT execution backend under this compiler backend."""

        requested_name = None if requested in (None, "auto") else requested
        if requested_name is None:
            if callable(self.default_execution_backend):
                requested_name = self.default_execution_backend(target)
            else:
                requested_name = self.default_execution_backend

        if requested_name is None:
            available = self.allowed_execution_backends(target, include_unavailable=False)
            if not available:
                raise ValueError(f"Backend {self.name!r} has no available execution backend for target {target}")
            requested_name = available[0]

        spec = self.execution_backends.get(requested_name)
        if spec is None or not spec.matches(target):
            allowed = ", ".join(self.allowed_execution_backends(target)) or "<none>"
            raise ValueError(
                f"Invalid execution backend {requested_name!r} for Backend {self.name!r} "
                f"and target {target.kind.name!r}. Allowed: {allowed}. Tip: use execution_backend='auto'."
            )

        if not spec.is_available():
            available = ", ".join(self.allowed_execution_backends(target, include_unavailable=False)) or "<none>"
            raise ValueError(f"Execution backend {requested_name!r} for Backend {self.name!r} is unavailable. Available: {available}.")

        return spec

    def preprocess_host_codegen(self, mod: IRModule, target: Target) -> IRModule:
        if self.host_pre_codegen is None:
            return mod
        return self.host_pre_codegen(mod, target)

    def feature(self, name: str, target: Target, default: object = None) -> object:
        query = self.features.get(name)
        return default if query is None else query(target)

    def ensure_callbacks_registered(self) -> None:
        if self.register_callbacks is not None:
            self.register_callbacks()
