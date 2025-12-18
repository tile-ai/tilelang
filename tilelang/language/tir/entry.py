from __future__ import annotations
import inspect
from typing import Callable

import tvm.script.parser.tir.entry as _tir_entry
from tvm.tir.function import PrimFunc
from tvm.script.parser._core import parse, scan_macro, utils


def prim_func(func: Callable | None = None, private: bool = False, check_well_formed: bool = False) -> PrimFunc | Callable:
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
    # pylint: disable=unused-argument
    # (private will be used in the parser, but not immediately)

    # need to capture this var outside the wrapper because the wrapper
    # adds to the stack
    outer_stack = inspect.stack()

    def decorator_wrapper(func):
        if not inspect.isfunction(func):
            raise TypeError(f"Expect a function, but got: {func}")
        nonlocal outer_stack
        if utils.is_defined_in_class(outer_stack, func):
            outer_stack = None
            return func
        outer_stack = None
        f = parse(func, utils.inspect_function_capture(func), check_well_formed=check_well_formed)
        setattr(f, "__name__", func.__name__)  # noqa: B010
        return f

    if func is not None:
        # no optional args given => use wrapper directly
        return decorator_wrapper(func)
    else:
        # if there is an optional arg given, return a new decorator
        # that will then be invoked
        setattr(decorator_wrapper, "dispatch_token", "tir")  # noqa: B010
        return decorator_wrapper


setattr(prim_func, "dispatch_token", "tir")  # noqa: B010


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
        def use1(A: T.Buffer((1024,), T.int32), B: T.Buffer((), T.int32)) -> None:
            for x_value in T.serial(10):
                static_capture(A, B)    ### Produces B[()] = A[128]

        @T.prim_func
        def use2(A: T.Buffer((1024,), T.int32), B: T.Buffer((), T.int32)) -> None:
            for x_value in T.serial(10):
                dynamic_capture(A, B)   ### Produces B[()] = A[x_value]
        ```
    """

    def _decorator(func: Callable) -> _tir_entry.TIRMacro:
        source, closure_vars = scan_macro(func, utils.inspect_function_capture(func))
        obj = _tir_entry.TIRMacro(source, closure_vars, func, hygienic)
        obj.__name__ = func.__name__
        return obj

    if len(args) == 0:
        return _decorator
    if len(args) == 1 and inspect.isfunction(args[0]):
        return _decorator(args[0])

    raise ValueError("Invalid use of T.macro. Usage: @T.macro, @T.macro(), @T.macro(hygienic=[True|False])")


setattr(macro, "dispatch_token", "tir")  # noqa: B010
