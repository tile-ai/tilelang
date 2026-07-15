"""CUDA Tile IR backend: structured TileLang PrimFunc -> CUDA Tile IR lowering.

This package owns the structured lowering from the TileLang Semantic IR to CUDA
Tile IR, together with toolchain detection and assembly. The runtime adapter
that turns a lowered kernel into a torch-callable lives in
``tilelang.jit.adapter.tileir``.
"""

from .artifact import (  # noqa: F401
    TileIRArgumentRef,
    TileIRArtifactCompatibility,
    TileIRLaunchMetadata,
    TileIRLoweringResult,
    TileIRTemporaryBuffer,
)
from .checks import (  # noqa: F401
    TileIRDependencyError,
    check_tileir_available,
    find_tileiras,
    is_tileir_available,
)
from .errors import (  # noqa: F401
    TileIRAssemblyError,
    TileIRLoweringError,
    TileIRLoweringNotImplementedError,
)
from .launch import extract_launch_metadata  # noqa: F401
from .lowering import lower_primfunc_to_tileir  # noqa: F401
from .semantic import (  # noqa: F401
    SemanticBuffer,
    SemanticKernel,
    SemanticProgram,
    SemanticRegion,
    SemanticStmt,
    TileLangSemanticError,
    extract_semantic_program,
)

# Register the TileIR target normalizer with the backend target registry (imported for
# its side effect). The execution backend is registered in `tilelang.cuda.execution_backend`
# (it must be ordered after the plain-CUDA backends for `auto` resolution).
from . import target as target  # noqa: F401,E402

__all__ = [
    "SemanticBuffer",
    "SemanticKernel",
    "SemanticProgram",
    "SemanticRegion",
    "SemanticStmt",
    "TileIRAssemblyError",
    "TileIRArgumentRef",
    "TileIRArtifactCompatibility",
    "TileIRDependencyError",
    "TileIRLaunchMetadata",
    "TileIRLoweringError",
    "TileIRLoweringNotImplementedError",
    "TileIRLoweringResult",
    "TileIRTemporaryBuffer",
    "TileLangSemanticError",
    "check_tileir_available",
    "extract_launch_metadata",
    "extract_semantic_program",
    "find_tileiras",
    "is_tileir_available",
    "lower_primfunc_to_tileir",
]
