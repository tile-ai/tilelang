"""Behavioral capabilities of one resolved TileLang backend context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from tvm.target import Target


@dataclass(frozen=True, slots=True)
class MatrixInstruction:
    m: int
    n: int
    k: int
    input_dtype: str
    accumulation_dtype: str


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Target-neutral facts that may legally affect program lowering."""

    subgroup_width: int
    max_threads_per_group: int
    shared_memory_bytes: int
    supported_dtypes: frozenset[str]
    matrix_instructions: tuple[MatrixInstruction, ...] = ()
    features: frozenset[str] = frozenset()
    native_multi_launch: bool = False
    native_argument_binding: bool = False
    max_kernels_per_program: int | None = None

    def __post_init__(self) -> None:
        if self.subgroup_width <= 0 or self.max_threads_per_group <= 0:
            raise ValueError("backend thread geometry must be positive")
        if self.shared_memory_bytes < 0 or not self.supported_dtypes:
            raise ValueError("backend storage capabilities are invalid")
        if self.max_kernels_per_program is not None and self.max_kernels_per_program <= 0:
            raise ValueError("max_kernels_per_program must be positive")

    def supports(self, feature: str) -> bool:
        return feature in self.features

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.subgroup_width,
                self.max_threads_per_group,
                self.shared_memory_bytes,
                tuple(sorted(self.supported_dtypes)),
                self.matrix_instructions,
                tuple(sorted(self.features)),
                self.native_multi_launch,
                self.native_argument_binding,
                self.max_kernels_per_program,
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def with_execution(
        self,
        *,
        native_multi_launch: bool,
        native_argument_binding: bool,
        max_kernels_per_program: int | None,
    ) -> BackendCapabilities:
        return replace(
            self,
            native_multi_launch=native_multi_launch,
            native_argument_binding=native_argument_binding,
            max_kernels_per_program=max_kernels_per_program,
        )


DEFAULT_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
    }
)


def target_limits(
    target: Target,
    *,
    subgroup_width: int,
    matrix_instructions: tuple[MatrixInstruction, ...] = (),
    features: frozenset[str] = frozenset(),
    supported_dtypes: frozenset[str] = DEFAULT_DTYPES,
) -> BackendCapabilities:
    """Read common limits from a normalized target with backend-owned fallbacks."""

    attrs = target.attrs
    return BackendCapabilities(
        subgroup_width=int(attrs.get("thread_warp_size", subgroup_width)),
        max_threads_per_group=int(attrs.get("max_threads_per_block", attrs.get("max_num_threads", 1))),
        shared_memory_bytes=int(attrs.get("max_shared_memory_per_block", 0)),
        supported_dtypes=supported_dtypes,
        matrix_instructions=matrix_instructions,
        features=features,
    )
