"""Lexical source blocks, axes, and definition-safe annotation ownership."""

from __future__ import annotations

from contextlib import ExitStack
from contextvars import ContextVar

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx import tirx as _tirx
from tvm_tirx.ir import Range
from tvm_tirx.ir import prim as _prim
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.script.ir_builder import frame as _shared_frame
from tvm_tirx.tirx.script import ir_builder as _builder

from ._construction import as_expr, current_location, unwrap
from ._annotation_codec import emit_annotation
from .frame import LoopFrame


def _source_annotation_owner(*, required=True):
    """Find the innermost active source SBlock, including Kernel's block."""
    from .kernel import KernelLaunchFrame

    kernel = KernelLaunchFrame.Current()
    blocks = _sblock_stack.get()
    # BlockFrame uses a nested IRBuilder. Checking active native frames avoids
    # confusing an outer kernel with a block in that nested builder, and also
    # handles a kernel entered inside a source block.
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if kernel is not None and frame.same_as(kernel._frame):
                return kernel
            for block in reversed(blocks):
                if frame.same_as(block._frame):
                    return block
    if not required:
        return None
    raise ValueError("T.sblock_attr must be called at the top of a T.sblock(), but it occurred outside of any T.sblock() frame")


def _source_annotation_values(value):
    value = unwrap(value)
    if isinstance(value, dict | tvm_ffi.Map):
        return {_source_annotation_values(key): _source_annotation_values(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_source_annotation_values(item) for item in value]
    return value


def _source_sblock_attr(annotations, source_annotations=None):
    """Check source declarations and retain definition-safe tensor references.

    A helper may supply data-identity keys for its source comparison. Emission
    uses tensor identities: layout_map owns typed attrs, other metadata uses
    operands. The short-lived owner state never enters serialized compiler IR.
    """
    owner = _source_annotation_owner()
    previous = getattr(owner, "_source_annotations", None)
    if previous is None:
        previous = _source_annotation_values(getattr(owner, "annotations", {}))
    declarations = _source_annotation_values(annotations if source_annotations is None else source_annotations)
    owner._source_annotations = tvm_ffi.get_global_func("tl.MergeSourceAnnotations")(declarations, previous)
    # Each declaration stays at its definition-safe source position. Consumers
    # collect marked declarations for this entire nearest source owner; they
    # do not give ordinary native metadata this whole-block interpretation.
    return emit_annotation({**annotations, "tl.source_block_annotation": True})


def sblock_attr(annotations):
    """Recursively merge annotations on the current source block.

    Conflicting leaves raise ValueError during construction, using FFI value
    and object-identity equality. Loops and conditionals do not create a new
    annotation owner; an explicit SBlock or Kernel does.
    """
    return _source_sblock_attr(annotations)


def _allocation_annotation(annotations):
    """Attach internal metadata to a just-emitted allocation.

    Reducer and barrier helpers also work at function scope, where the old
    implicit allocation owner is not an explicit source SBlock. Keep their
    metadata next to the allocation as ordinary native operands. Public block
    annotation helpers continue to require a real SBlock or Kernel, and an
    allocation inside one still participates in that owner's merge checks.
    """
    if _source_annotation_owner(required=False) is None:
        return emit_annotation(annotations)
    return _source_sblock_attr(annotations)


_ASCEND_BLOCK_NAMES = frozenset({"CUBE", "VECTOR", "SIMT_VF", "SIMD_VF"})


def _ascend_block(name, annotations=None):
    """Build an Ascend execution domain using the native region builder.

    Only execution configuration belongs in the header. Tensor annotations
    stay in the body, after their definitions, via their own annotation API.
    """
    return _builder.region("tl.ascend_block", [], attrs={"name": name, **(annotations or {})})


_sblock_stack = ContextVar("tilelang_sblock_stack", default=())


class BlockFrame:
    """Collect a lexical block so its predicate guards the complete body.

    A nested *native* builder keeps the finished body available until ``where``
    and axis metadata have been resolved. The emitted IR contains ordinary
    Bind, If and native regions, with no legacy SBlock objects.
    """

    def __init__(self, name="", no_realize=False):
        self.name = str(name)
        self.no_realize = no_realize
        self.axes = []
        self.predicates = []
        self._builder = IRBuilder()
        self._frame = None
        self._token = None

    @property
    def stmts(self):
        return self._frame.stmts if self._frame is not None else []

    def __enter__(self):
        self._builder.__enter__()
        try:
            if self.name in _ASCEND_BLOCK_NAMES:
                # Axis bindings and predicates are already explicit native
                # statements; no_realize is not an execution-region attribute.
                self._frame = _ascend_block(self.name)
            else:
                self._frame = _builder.region("tl.lexical_scope", [], attrs={})
            self._frame.__enter__()
        except BaseException:
            import sys

            self._builder.__exit__(*sys.exc_info())
            raise
        self._token = _sblock_stack.set((*_sblock_stack.get(), self))
        return None

    def __exit__(self, exc_type, exc, trace):
        try:
            self._frame.__exit__(exc_type, exc, trace)
        finally:
            self._builder.__exit__(exc_type, exc, trace)
            _sblock_stack.reset(self._token)
        if exc_type is None:
            body = self._builder.get()
            # Block predicates belong to the enclosing scope. Axis aliases are
            # explicit Binds inside the block and must be replaced here.
            replacements = {var: binding for _, var, _, binding in self.axes}
            predicates = [
                tvm_ffi.structural_map(predicate, (ir.Var, lambda var: replacements.get(var, var))) for predicate in self.predicates
            ]
            if predicates:
                condition = predicates[0]
                for predicate in predicates[1:]:
                    condition = _prim.And(condition, predicate)
                body = ir.If(condition, body, None, current_location())
            _builder.add_to_parent(body)


def sblock(name="", no_realize=False):
    return BlockFrame(name, no_realize)


block = sblock


def _current_sblock():
    stack = _sblock_stack.get()
    if not stack:
        raise ValueError("This operation must appear inside T.sblock")
    return stack[-1]


def where(predicate):
    _current_sblock().predicates.append(as_expr(predicate))


def _region(value):
    if isinstance(value, ir.TensorRegion):
        return value
    if isinstance(value, ir.TensorLoad):
        return ir.TensorRegion(
            value.source, [Range.from_min_extent(index, 1) for index in value.indices], ir.TensorRegionType(), current_location()
        )
    if _tirx.is_tensor_var(value):
        return ir.TensorRegion(
            value, [Range.from_min_extent(0, extent) for extent in value.ty.shape], ir.TensorRegionType(), current_location()
        )
    raise TypeError("Read/write regions require a tensor or a tensor subscript")


def reads(*regions):
    return sblock_attr({"tl.reads": [_region(value) for value in regions]})


def writes(*regions):
    return sblock_attr({"tl.writes": [_region(value) for value in regions]})


class InitFrame:
    """Execute a reduction initializer only on the first reduction coordinates."""

    def __init__(self):
        owner = _current_sblock()
        conditions = [var == domain.begin for kind, var, domain, _ in owner.axes if kind == "R"]
        self.condition = as_expr(True)
        for condition in conditions:
            self.condition = _prim.And(self.condition, as_expr(condition))
        self._stack = None

    @property
    def stmts(self):
        return self._frame.stmts

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.__enter__()
        try:
            self._stack.enter_context(_builder.if_(self.condition, loc=current_location()))
            self._frame = _builder.then_(loc=current_location())
            self._stack.enter_context(self._frame)
        except BaseException:
            import sys

            self._stack.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def init():
    return InitFrame()


def _axis_domain(domain):
    if isinstance(domain, ir.Range):
        return domain
    if isinstance(domain, tuple | list | tvm_ffi.Array):
        if len(domain) != 2:
            raise ValueError("An axis domain must have begin/end bounds")
        return ir.Range(*domain)
    return ir.Range(0, domain)


class axis:
    """Block-axis annotations carried by actual native scalar bindings."""

    @staticmethod
    def _bind(kind, domain, binding, dtype="int32"):
        owner = _current_sblock()
        domain = _axis_domain(domain)
        value = as_expr(binding)
        var = ir.Var("axis", ir.PrimType(dtype), current_location())
        _builder.add_to_parent(ir.Bind(var, value, current_location()))
        owner.axes.append((kind, var, domain, value))
        emit_annotation({"tl.axis": {"kind": kind, "var": var, "begin": domain.begin, "extent": domain.extent}})
        return var

    @staticmethod
    def spatial(domain, binding, dtype="int32"):
        return axis._bind("S", domain, binding, dtype)

    @staticmethod
    def reduce(domain, binding, dtype="int32"):
        return axis._bind("R", domain, binding, dtype)

    @staticmethod
    def scan(domain, binding, dtype="int32"):
        return axis._bind("scan", domain, binding, dtype)

    @staticmethod
    def opaque(domain, binding, dtype="int32"):
        return axis._bind("opaque", domain, binding, dtype)

    @staticmethod
    def remap(kinds, bindings, dtype="int32"):
        from .eager.builder import Builder

        if len(kinds) != len(bindings) or any(kind not in "SR" for kind in kinds):
            raise ValueError("axis.remap requires one S/R kind per loop variable")
        builder = Builder.current()
        if builder is None:
            raise ValueError("axis.remap requires an active TileLang loop builder")
        domains = {}
        for current in builder.frames:
            frames = current.frames if isinstance(current, LoopFrame) else [current]
            for frame in frames:
                if isinstance(frame, _shared_frame.ForFrame):
                    domains.update(zip(frame.vars, frame.doms))
        result = []
        for kind, value in zip(kinds, bindings):
            if value not in domains:
                raise ValueError("axis.remap must refer to enclosing loop variables")
            result.append(axis._bind(kind, domains[value], value, dtype))
        return result[0] if len(result) == 1 else result

    S = spatial
    R = reduce
