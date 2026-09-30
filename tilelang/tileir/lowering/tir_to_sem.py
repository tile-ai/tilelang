"""TIR → SemanticIR front door for the TileIR lowering pipeline.

This module is the **sole HIR entry point** for the pipeline:

    TIR → SemanticIR (this module) → TileIR → passes → MLIR

All downstream lowering stages (``sem_to_ir`` onwards) consume
only the structured ``SemanticProgram``/``SemanticStmt`` representation
produced here.  There is no raw-TIR-AST fallback in the pipeline.

If the PrimFunc contains constructs that ``extract_semantic_program`` cannot
represent as SemanticIR, a ``TileLangSemanticError`` is raised.  That is the
single, explicit rejection point for unsupported TIR.
"""

from __future__ import annotations

from tvm import tirx

from tilelang.tileir.semantic import SemanticProgram, extract_semantic_program


def tir_to_sem(prim_func: tirx.PrimFunc) -> SemanticProgram:
    """Convert a TVM TIR ``PrimFunc`` to a ``SemanticProgram`` (HIR).

    Delegates entirely to :func:`tilelang.tileir.semantic.extract_semantic_program`.
    Any ``TileLangSemanticError`` raised by the extractor propagates
    unchanged — it is the canonical signal that a TIR construct is not yet
    representable in SemanticIR.

    Parameters
    ----------
    prim_func:
        A TVM TIR ``PrimFunc`` produced by the TileLang frontend (i.e. traced
        via ``T.prim_func`` / ``tilelang.jit``).

    Returns
    -------
    SemanticProgram
        The structured SemanticIR representation of the kernel(s) in
        *prim_func*, ready for consumption by downstream lowering stages.
    """
    return extract_semantic_program(prim_func)
