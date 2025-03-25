# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.


def deprecated(
    method_name: str,
    new_method_name: str,
):
    """A decorator to indicate that a method is deprecated

    Parameters
    ----------
    method_name : str
        The name of the method to deprecate
    new_method_name : str
        The name of the new method to use instead
    """
    import functools  # pylint: disable=import-outside-toplevel
    import warnings  # pylint: disable=import-outside-toplevel

    def _deprecate(func):

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            warnings.warn(
                f"{method_name} is deprecated, use {new_method_name} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return _wrapper

    return _deprecate
