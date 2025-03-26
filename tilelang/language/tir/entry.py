# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import Callable, Optional, Union

import tvm.script
from tvm.tir.function import PrimFunc
import tvm.script.parser.tir.entry as _tir_entry
import inspect
from tvm.script.parser._core import parse, scan_macro, utils 

def prim_func(
    func: Optional[Callable] = None, private: bool = False, check_well_formed=True
) -> Union[PrimFunc, Callable]:
    """The parsing method for tir prim func, by using `@prim_func` as decorator.

    Parameters
    ----------
    func : Callable
        The function to be parsed as prim func.
        (Listed as optional to allow the decorator to be used
        without arguments, like `@prim_func`,
        or with an argument, `@prim_func(private=True)`)

    private : bool, optional
        Whether the function should be treated as private.
        A private function has no global symbol attribute;
        if the function is not private, it will have a global symbol
        matching the function name.

    Returns
    -------
    res : Union[PrimFunc, Callable]
        The parsed tir prim func.
    """

    return _tir_entry.prim_func(func, private, check_well_formed)

setattr(prim_func, "dispatch_token", "tir")

def macro(*args, hygienic: bool = True) -> Callable:
    """Decorator for macro definitions.

    Parameters
    ----------
    hygienic: bool
        Specifies whether the macro is hygienic or not.
        A macro is hygienic if all symbols used in the macro's body are resolved
        to values from the location of the macro definition. A non-hygienic macro
        will have its symbols resolved to values at the time of the macro's use.

        Example:
        ```
        import tvm
        from tvm.script import tir as T

        x_value = 128

        @T.macro(hygienic=True)
        def static_capture(A, B):
            B[()] = A[x_value]          ### x_value binds to 128

        @T.macro(hygienic=False)
        def dynamic_capture(A, B):
            B[()] = A[x_value]          ### x_value will bind at the time of use


        @T.prim_func
        def use1(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
            for x_value in T.serial(10):
                static_capture(A, B)    ### Produces B[()] = A[128]

        @T.prim_func
        def use2(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
            for x_value in T.serial(10):
                dynamic_capture(A, B)   ### Produces B[()] = A[x_value]
        ```
    """

    return _tir_entry.macro(*args, hygienic=hygienic)

setattr(macro, "dispatch_token", "tir")
