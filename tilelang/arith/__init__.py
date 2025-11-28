from typing import TYPE_CHECKING, overload
import tvm_ffi
from tvm.ir.expr import PrimExpr, Range
from tvm.tir.expr import Var
from . import _ffi_api
from typing import Literal

@tvm_ffi.register_object("tl.arith.Z3Analyzer")
class Z3Analyzer(tvm_ffi.Object):
    def __init__(self, use_int=True, bv_padding_bits=4):
        """Initialize a Z3Analyzer object."""
        self.__init_handle_by_constructor__(_ffi_api.Z3Analyzer, use_int, bv_padding_bits)
    @overload
    def bind(self, var: Var, value: PrimExpr, /, allow_override: bool=False): ...
    @overload
    def bind(self, var: Var, range: Range, /, allow_override: bool=False): ...
    
    def bind(self, var, data, /, allow_override: bool=False):
        self._Bind(var, data, allow_override)

    def set_param(self, param: str, value: str | int | float | bool):
        self._SetParam(param, value)

    def __repr__(self):
        return self.get_smtlib2()

    if TYPE_CHECKING:
        def can_prove(self, expr: PrimExpr) -> bool: ...
        def get_smtlib2(self) -> str: ...
        def enter_with_scope(self): ...
        def exit_with_scope(self): ...
        def add_constraint(self, expr: PrimExpr): ...
        def add_assume(self, expr: PrimExpr): ...
        def set_timeout_ms(self, timeout: int): ...
        def set_max_step(self, max_step: int): ...